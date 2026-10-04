"""Local attention diagnostics for frozen-prefill compression policies.

The routines here deliberately evaluate on queries *after* the probe window.
That keeps the allocation signal and the diagnostic signal distinct and makes
static-score drift measurable without running a full generation campaign.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Mapping

import torch
import torch.nn.functional as F

from .quantization import affine_roundtrip
from .rd_allocator import allocate_lagrangian


@dataclass(frozen=True)
class LocalAttentionMetrics:
    attention_tv: float
    attention_js: float
    missed_mass: float
    top1_flip_rate: float
    output_relative_mse: float
    output_cosine: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def probe_token_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    prefix_tokens: int,
    observation_window: int,
    pooling_kernel: int = 5,
) -> torch.Tensor:
    """SnapKV/RDKV attention-mass weights in ``[B, KVH, prefix]`` layout."""

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query/key must have [B,H,T,D] layout")
    batch, query_heads, total_tokens, head_dim = query.shape
    if key.shape[0] != batch or key.shape[2:] != (total_tokens, head_dim):
        raise ValueError("query/key shapes are incompatible")
    kv_heads = key.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV heads")
    if not observation_window <= prefix_tokens <= total_tokens:
        raise ValueError("invalid prefix/probe lengths")

    start = prefix_tokens - observation_window
    probe_query = query[:, :, start:prefix_tokens, :].float()
    prefix_key = key[:, :, :prefix_tokens, :]
    groups = query_heads // kv_heads
    repeated_key = prefix_key.repeat_interleave(groups, dim=1).float()
    logits = torch.matmul(probe_query, repeated_key.transpose(-1, -2)) / sqrt(head_dim)
    query_positions = torch.arange(start, prefix_tokens, device=query.device)
    key_positions = torch.arange(prefix_tokens, device=query.device)
    causal = key_positions.view(1, 1, 1, -1) > query_positions.view(1, 1, -1, 1)
    logits.masked_fill_(causal, torch.finfo(logits.dtype).min)
    probabilities = logits.softmax(dim=-1)
    weights = probabilities.sum(dim=-2)
    weights = weights.reshape(batch, kv_heads, groups, prefix_tokens).sum(dim=2)
    if pooling_kernel > 1:
        if pooling_kernel % 2 == 0:
            raise ValueError("pooling_kernel must be odd")
        flat = weights.reshape(batch * kv_heads, 1, prefix_tokens)
        flat = F.avg_pool1d(
            flat,
            kernel_size=pooling_kernel,
            stride=1,
            padding=pooling_kernel // 2,
        )
        weights = flat.reshape(batch, kv_heads, prefix_tokens)
    return weights


def single_eviction_output_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    prefix_tokens: int,
    observation_window: int,
    pooling_kernel: int = 5,
) -> torch.Tensor:
    """Exact one-token eviction error on the probe attention outputs.

    For one query with full output ``o`` and attention mass ``a_t``, evicting
    token ``t`` and renormalizing the remaining attention changes the output by

    ``a_t / (1-a_t) * (o-v_t)``.

    This score sums its squared norm over the observation queries and grouped
    query heads.  Unlike attention-mass ranking, it sees value geometry and
    cancellation.  It remains a diagonal/single-removal proxy: simultaneous
    eviction interactions are intentionally left for a separate diagnostic.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must have [B,H,T,D] layout")
    batch, query_heads, total_tokens, head_dim = query.shape
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if key.shape[0] != batch or key.shape[2:] != (total_tokens, head_dim):
        raise ValueError("query/key/value shapes are incompatible")
    kv_heads = key.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV heads")
    if not observation_window <= prefix_tokens <= total_tokens:
        raise ValueError("invalid prefix/probe lengths")

    start = prefix_tokens - observation_window
    groups = query_heads // kv_heads
    probe_query = query[:, :, start:prefix_tokens, :].float()
    prefix_key = key[:, :, :prefix_tokens, :].repeat_interleave(groups, dim=1).float()
    prefix_value = value[:, :, :prefix_tokens, :].repeat_interleave(groups, dim=1).float()
    logits = torch.matmul(probe_query, prefix_key.transpose(-1, -2)) / sqrt(head_dim)
    query_positions = torch.arange(start, prefix_tokens, device=query.device)
    key_positions = torch.arange(prefix_tokens, device=query.device)
    causal = key_positions.view(1, 1, 1, -1) > query_positions.view(1, 1, -1, 1)
    logits.masked_fill_(causal, torch.finfo(logits.dtype).min)
    attention = logits.softmax(dim=-1)
    output = torch.matmul(attention, prefix_value)

    # ||v_t-o_tau||^2 without materializing [B,H,W,T,D].
    value_norm = prefix_value.square().sum(dim=-1).unsqueeze(-2)
    output_norm = output.square().sum(dim=-1).unsqueeze(-1)
    cross = torch.matmul(output, prefix_value.transpose(-1, -2))
    residual_norm = (value_norm + output_norm - 2.0 * cross).clamp_min_(0.0)
    scale = attention / (1.0 - attention).clamp_min(1e-6)
    scores = (scale.square() * residual_norm).sum(dim=-2)
    scores = scores.reshape(batch, kv_heads, groups, prefix_tokens).sum(dim=2)
    if pooling_kernel > 1:
        if pooling_kernel % 2 == 0:
            raise ValueError("pooling_kernel must be odd")
        flat = scores.reshape(batch * kv_heads, 1, prefix_tokens)
        flat = F.avg_pool1d(
            flat,
            kernel_size=pooling_kernel,
            stride=1,
            padding=pooling_kernel // 2,
        )
        scores = flat.reshape(batch, kv_heads, prefix_tokens)
    return scores


def single_token_pair_action_distortions(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_candidates: torch.Tensor,
    value_candidates: torch.Tensor,
    *,
    prefix_tokens: int,
    observation_window: int,
    output_projections: torch.Tensor | None = None,
    token_chunk_size: int = 256,
) -> torch.Tensor:
    """Exact probe-output distortion for one-token K/V replacement actions.

    Candidate tensors use ``[A,B,KVH,T,D]`` layout.  The returned table has
    shape ``[B,KVH,prefix,A+1]``: action zero evicts the token, and each later
    action replaces that token's K/V pair with the corresponding candidate.
    Every table entry is exact when *only that token* changes; using the table
    additively across many tokens is the diagonal approximation consumed by a
    rate--distortion allocator.

    For full attention output ``o``, token probability ``p`` and key-induced
    logit change ``delta``, the replacement output obeys

    ``o' - o = p/Z * [exp(delta)(v'-o) + (o-v)]``, where
    ``Z = 1 + p(exp(delta)-1)``.

    Eviction is the same expression at ``exp(delta)=0``.  This gives one
    common distortion surface for token removal and progressive vector
    quantization, while retaining the softmax renormalization and K/V coupling
    that separable norm-times-NMSE proxies discard.

    By default the error is the squared Euclidean norm of the attention-output
    perturbation.  ``output_projections`` optionally supplies empirical-Fisher
    vectors in ``[S,B,QH,W,D]`` (or one vector in ``[B,QH,W,D]``), where ``W``
    is the observation window.  In that mode each entry is

    ``0.5 * mean_s sum_w (sum_group <g_s, delta_o>)**2``.

    Query heads belonging to one KV head are summed *before* squaring because
    replacing one KV pair perturbs the whole grouped-query block jointly.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must have [B,H,T,D] layout")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if key_candidates.ndim != 5 or value_candidates.shape != key_candidates.shape:
        raise ValueError("candidate K/V must share [A,B,KVH,T,D] layout")
    if key_candidates.shape[1:] != key.shape:
        raise ValueError("candidate and original KV geometry differs")
    batch, query_heads, total_tokens, head_dim = query.shape
    if key.shape[0] != batch or key.shape[2:] != (total_tokens, head_dim):
        raise ValueError("query and KV geometry differs")
    kv_heads = key.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if not observation_window <= prefix_tokens <= total_tokens:
        raise ValueError("invalid prefix/probe lengths")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")

    groups = query_heads // kv_heads
    start = prefix_tokens - observation_window
    probe_query = query[:, :, start:prefix_tokens, :].float()
    original_key = key[:, :, :prefix_tokens, :].repeat_interleave(groups, dim=1).float()
    original_value = value[:, :, :prefix_tokens, :].repeat_interleave(groups, dim=1).float()
    logits = torch.matmul(probe_query, original_key.transpose(-1, -2)) / sqrt(head_dim)
    query_positions = torch.arange(start, prefix_tokens, device=query.device)
    key_positions = torch.arange(prefix_tokens, device=query.device)
    causal = key_positions.view(1, 1, 1, -1) > query_positions.view(1, 1, -1, 1)
    logits.masked_fill_(causal, torch.finfo(logits.dtype).min)
    probability = logits.softmax(dim=-1)
    output = torch.matmul(probability, original_value)

    projections = None
    if output_projections is not None:
        if output_projections.ndim == 4:
            output_projections = output_projections.unsqueeze(0)
        expected_projection_shape = (
            batch,
            query_heads,
            observation_window,
            head_dim,
        )
        if (
            output_projections.ndim != 5
            or tuple(output_projections.shape[1:]) != expected_projection_shape
        ):
            raise ValueError(
                "output projections must use [S,B,QH,W,D] and match the probe"
            )
        projections = output_projections.float()

    def reduce_projected_delta(projected_delta: torch.Tensor) -> torch.Tensor:
        samples = projected_delta.shape[0]
        grouped = projected_delta.reshape(
            samples,
            batch,
            kv_heads,
            groups,
            observation_window,
            prefix_tokens,
        ).sum(dim=3)
        return 0.5 * grouped.square().mean(dim=0).sum(dim=-2)

    if projections is not None:
        projected_value = torch.einsum(
            "sbhwd,bhtd->sbhwt", projections, original_value
        )
        projected_output = torch.einsum(
            "sbhwd,bhwd->sbhw", projections, output
        ).unsqueeze(-1)
        eviction_scale = probability / (1.0 - probability).clamp_min(1e-7)
        eviction_delta = eviction_scale.unsqueeze(0) * (
            projected_output - projected_value
        )
        action_errors = [reduce_projected_delta(eviction_delta)]
    else:
        # Do not expand ||o-v||^2 as ||o||^2+||v||^2-2o^Tv.  At deep
        # transformer layers those terms can be O(1e10) while the replacement
        # error is tiny; their FP32 cancellation previously overwhelmed the
        # actual 15/16/17-stage ordering.  Materialize only a bounded token
        # chunk of the difference vector and square that directly.
        eviction_error = torch.empty(
            (batch, query_heads, prefix_tokens),
            dtype=torch.float32,
            device=query.device,
        )
        for token_start in range(0, prefix_tokens, token_chunk_size):
            token_stop = min(prefix_tokens, token_start + token_chunk_size)
            local_probability = probability[..., token_start:token_stop]
            local_scale = local_probability / (1.0 - local_probability).clamp_min(1e-7)
            local_difference = (
                output.unsqueeze(-2)
                - original_value[:, :, None, token_start:token_stop, :]
            )
            eviction_error[..., token_start:token_stop] = (
                local_scale.unsqueeze(-1) * local_difference
            ).square().sum(dim=-1).sum(dim=-2)
        action_errors = [eviction_error]

    for action in range(key_candidates.shape[0]):
        candidate_key = key_candidates[action, :, :, :prefix_tokens, :]
        candidate_value = value_candidates[action, :, :, :prefix_tokens, :]
        candidate_key = candidate_key.repeat_interleave(groups, dim=1).float()
        candidate_value = candidate_value.repeat_interleave(groups, dim=1).float()
        logit_delta = torch.einsum(
            "bhwd,bhtd->bhwt", probe_query, candidate_key - original_key
        ) / sqrt(head_dim)
        ratio = logit_delta.clamp(min=-30.0, max=30.0).exp()
        normalizer = (1.0 + probability * (ratio - 1.0)).clamp_min(1e-7)

        one_minus_ratio = 1.0 - ratio
        scale = probability / normalizer
        if projections is not None:
            projected_candidate = torch.einsum(
                "sbhwd,bhtd->sbhwt", projections, candidate_value
            )
            projected_delta = scale.unsqueeze(0) * (
                ratio.unsqueeze(0) * projected_candidate
                - projected_value
                + one_minus_ratio.unsqueeze(0) * projected_output
            )
            action_errors.append(reduce_projected_delta(projected_delta))
        else:
            replacement_error = torch.empty(
                (batch, query_heads, prefix_tokens),
                dtype=torch.float32,
                device=query.device,
            )
            for token_start in range(0, prefix_tokens, token_chunk_size):
                token_stop = min(prefix_tokens, token_start + token_chunk_size)
                local_ratio = ratio[..., token_start:token_stop]
                local_scale = scale[..., token_start:token_stop]
                local_candidate = candidate_value[
                    :, :, None, token_start:token_stop, :
                ]
                local_original = original_value[
                    :, :, None, token_start:token_stop, :
                ]
                local_output = output.unsqueeze(-2)
                # Exact stable form of
                # p/Z * [exp(delta)(v'-o) + (o-v)].
                delta_output = local_scale.unsqueeze(-1) * (
                    local_ratio.unsqueeze(-1) * (local_candidate - local_output)
                    + (local_output - local_original)
                )
                replacement_error[..., token_start:token_stop] = (
                    delta_output.square().sum(dim=-1).sum(dim=-2)
                )
            action_errors.append(replacement_error)

    stacked = torch.stack(action_errors, dim=-1)
    if projections is not None:
        return stacked
    return stacked.reshape(
        batch, kv_heads, groups, prefix_tokens, stacked.shape[-1]
    ).sum(dim=2)


def external_query_pair_action_distortions(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_candidates: torch.Tensor,
    value_candidates: torch.Tensor,
) -> torch.Tensor:
    """One-token replacement errors for external queries over a past cache.

    This is the decode-time counterpart of
    :func:`single_token_pair_action_distortions`. ``query`` has independent
    length ``Q`` and attends every one of the ``T`` cache entries, so no causal
    alignment mask is applied. Candidate tensors use ``[A,B,KVH,T,D]`` and the
    result uses ``[B,KVH,T,A+1]`` with eviction at action zero.
    """

    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("query/key/value must use compatible 4D layouts")
    if key_candidates.ndim != 5 or value_candidates.shape != key_candidates.shape:
        raise ValueError("candidate K/V must share [A,B,KVH,T,D] layout")
    if key_candidates.shape[1:] != key.shape:
        raise ValueError("candidate and cache geometry differs")
    batch, query_heads, _queries, head_dim = query.shape
    if key.shape[0] != batch or key.shape[-1] != head_dim:
        raise ValueError("query and cache geometry differs")
    kv_heads = key.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")

    groups = query_heads // kv_heads
    probe_query = query.float()
    original_key = key.repeat_interleave(groups, dim=1).float()
    original_value = value.repeat_interleave(groups, dim=1).float()
    logits = torch.matmul(probe_query, original_key.transpose(-1, -2)) / sqrt(
        head_dim
    )
    probability = logits.softmax(dim=-1)
    output = torch.matmul(probability, original_value)
    value_norm = original_value.square().sum(dim=-1).unsqueeze(-2)
    output_norm = output.square().sum(dim=-1).unsqueeze(-1)
    output_dot_value = torch.matmul(output, original_value.transpose(-1, -2))
    eviction_residual = (
        value_norm + output_norm - 2.0 * output_dot_value
    ).clamp_min(0.0)
    eviction_scale = probability / (1.0 - probability).clamp_min(1e-7)
    action_errors = [(eviction_scale.square() * eviction_residual).sum(dim=-2)]

    for action in range(key_candidates.shape[0]):
        candidate_key = key_candidates[action].repeat_interleave(groups, dim=1).float()
        candidate_value = value_candidates[action].repeat_interleave(groups, dim=1).float()
        logit_delta = torch.einsum(
            "bhqd,bhtd->bhqt", probe_query, candidate_key - original_key
        ) / sqrt(head_dim)
        ratio = logit_delta.clamp(min=-30.0, max=30.0).exp()
        normalizer = (1.0 + probability * (ratio - 1.0)).clamp_min(1e-7)
        candidate_norm = candidate_value.square().sum(dim=-1).unsqueeze(-2)
        candidate_dot_value = (
            candidate_value * original_value
        ).sum(dim=-1).unsqueeze(-2)
        output_dot_candidate = torch.matmul(
            output, candidate_value.transpose(-1, -2)
        )
        one_minus_ratio = 1.0 - ratio
        residual_norm = (
            ratio.square() * candidate_norm
            + value_norm
            + one_minus_ratio.square() * output_norm
            - 2.0 * ratio * candidate_dot_value
            + 2.0 * ratio * one_minus_ratio * output_dot_candidate
            - 2.0 * one_minus_ratio * output_dot_value
        ).clamp_min(0.0)
        scale = probability / normalizer
        action_errors.append((scale.square() * residual_norm).sum(dim=-2))

    stacked = torch.stack(action_errors, dim=-1)
    return stacked.reshape(
        batch, kv_heads, groups, key.shape[2], stacked.shape[-1]
    ).sum(dim=2)


def future_token_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    prefix_tokens: int,
) -> torch.Tensor:
    """Oracle attention mass from held-out future queries (diagnostic only)."""

    query_heads = query.shape[1]
    kv_heads = key.shape[1]
    groups = query_heads // kv_heads
    future_query = query[:, :, prefix_tokens:, :].float()
    prefix_key = key[:, :, :prefix_tokens, :].repeat_interleave(groups, dim=1).float()
    logits = torch.matmul(future_query, prefix_key.transpose(-1, -2)) / sqrt(query.shape[-1])
    probabilities = logits.softmax(dim=-1)
    weights = probabilities.sum(dim=-2)
    return weights.reshape(query.shape[0], kv_heads, groups, prefix_tokens).sum(dim=2)


def key_channel_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    prefix_tokens: int,
) -> torch.Tensor:
    """RDKV/ThinK norm-product weights in ``[B, KVH, D]`` layout."""

    query_heads = query.shape[1]
    kv_heads = key.shape[1]
    groups = query_heads // kv_heads
    q = query[:, :, :prefix_tokens, :].float()
    q_norm = q.square().sum(dim=2).sqrt()
    q_norm = q_norm.reshape(q.shape[0], kv_heads, groups, q.shape[-1]).square().sum(dim=2).sqrt()
    k_norm = key[:, :, :prefix_tokens, :].float().square().sum(dim=2).sqrt()
    return q_norm * k_norm / sqrt(query.shape[-1])


def topk_with_recent(
    weights: torch.Tensor,
    *,
    capacity: int,
    recent_tokens: int,
) -> torch.Tensor:
    """Boolean mask retaining top weighted history plus an exact recent tail."""

    if not 0 < capacity <= weights.shape[-1]:
        raise ValueError("capacity is outside the token range")
    recent = min(recent_tokens, capacity)
    history_end = weights.shape[-1] - recent
    history_capacity = capacity - recent
    mask = torch.zeros_like(weights, dtype=torch.bool)
    if history_capacity:
        selected = weights[..., :history_end].topk(history_capacity, dim=-1).indices
        mask.scatter_(-1, selected, True)
    if recent:
        mask[..., -recent:] = True
    return mask


def allocate_rdkv_reference(
    token_weights: torch.Tensor,
    channel_weights: torch.Tensor,
    *,
    effective_bits: float,
    value_distortion: Mapping[int, float],
    key_distortion: Mapping[int, float],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """RDKV's sequential V-token then K-channel allocation.

    The requested effective rate is measured per original K/V scalar. Half of
    the raw-code budget is assigned to V and half to K, as in RDKV.
    """

    if token_weights.shape[:2] != channel_weights.shape[:2]:
        raise ValueError("token/channel weights must share batch and head axes")
    if effective_bits < 0:
        raise ValueError("effective_bits must be non-negative")
    batch, heads, tokens = token_weights.shape
    channels = channel_weights.shape[-1]
    value_bits = torch.empty_like(token_weights, dtype=torch.int16)
    key_bits = torch.empty_like(channel_weights, dtype=torch.int16)
    value_objective = 0.0
    key_objective = 0.0
    for batch_index in range(batch):
        for head_index in range(heads):
            value_result = allocate_lagrangian(
                token_weights[batch_index, head_index],
                target_mean_bits=effective_bits,
                distortion=value_distortion,
            )
            value_bits[batch_index, head_index] = value_result.bits
            value_objective += value_result.objective
            kept = max(1, int(value_result.bits.gt(0).sum().item()))
            key_target = min(16.0, effective_bits * tokens / kept)
            key_result = allocate_lagrangian(
                channel_weights[batch_index, head_index],
                target_mean_bits=key_target,
                distortion=key_distortion,
            )
            key_bits[batch_index, head_index] = key_result.bits
            key_objective += key_result.objective

    kept_mask = value_bits.gt(0)
    raw_bits = int((value_bits.sum() * channels).item())
    for batch_index in range(batch):
        for head_index in range(heads):
            kept = int(kept_mask[batch_index, head_index].sum().item())
            raw_bits += int(key_bits[batch_index, head_index].sum().item()) * kept
    denominator = batch * heads * tokens * channels * 2
    summary = {
        "raw_effective_bits": raw_bits / denominator,
        "retained_token_fraction": float(kept_mask.float().mean().item()),
        "value_mean_bits": float(value_bits.float().mean().item()),
        "key_mean_bits_on_channels": float(key_bits.float().mean().item()),
        "predicted_value_objective": value_objective,
        "predicted_key_objective": key_objective,
    }
    return value_bits, key_bits, summary


def _mixed_value_roundtrip(
    value: torch.Tensor,
    bits: torch.Tensor,
    *,
    group_size: int | None = None,
) -> torch.Tensor:
    replay = torch.zeros_like(value)
    for bit in (1, 2, 4, 8, 16):
        mask = bits.eq(bit)
        if not mask.any():
            continue
        rows = value[mask]
        if bit == 16:
            replay[mask] = rows
        else:
            quantized, _ = affine_roundtrip(
                rows,
                bits=bit,
                axis=-1,
                group_size=group_size or value.shape[-1],
            )
            replay[mask] = quantized
    return replay


def _mixed_key_roundtrip(
    key: torch.Tensor,
    channel_bits: torch.Tensor,
    retained: torch.Tensor,
    *,
    group_size: int | None = None,
) -> torch.Tensor:
    replay = torch.zeros_like(key)
    for batch_index in range(key.shape[0]):
        for head_index in range(key.shape[1]):
            token_mask = retained[batch_index, head_index]
            rows = key[batch_index, head_index, token_mask]
            if rows.numel() == 0:
                continue
            output = torch.zeros_like(rows)
            for bit in (1, 2, 4, 8, 16):
                channel_mask = channel_bits[batch_index, head_index].eq(bit)
                if not channel_mask.any():
                    continue
                columns = rows[:, channel_mask]
                if bit == 16:
                    output[:, channel_mask] = columns
                else:
                    quantized, _ = affine_roundtrip(
                        columns,
                        bits=bit,
                        axis=0,
                        group_size=group_size or rows.shape[0],
                    )
                    output[:, channel_mask] = quantized
            replay[batch_index, head_index, token_mask] = output
    return replay


def uniform_candidate(
    key: torch.Tensor,
    value: torch.Tensor,
    token_weights: torch.Tensor,
    *,
    bits: int,
    capacity: int,
    recent_tokens: int,
    group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    retained = topk_with_recent(
        token_weights, capacity=capacity, recent_tokens=recent_tokens
    )
    value_bits = retained.to(torch.int16) * bits
    channel_bits = torch.full(
        (*retained.shape[:2], key.shape[-1]),
        bits,
        dtype=torch.int16,
        device=key.device,
    )
    return (
        _mixed_key_roundtrip(key, channel_bits, retained, group_size=group_size),
        _mixed_value_roundtrip(value, value_bits, group_size=group_size),
        retained,
    )


def rdkv_candidate(
    key: torch.Tensor,
    value: torch.Tensor,
    value_bits: torch.Tensor,
    key_bits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    retained = value_bits.gt(0)
    return (
        _mixed_key_roundtrip(key, key_bits, retained),
        _mixed_value_roundtrip(value, value_bits),
        retained,
    )


def evaluate_local_attention(
    future_query: torch.Tensor,
    full_key: torch.Tensor,
    full_value: torch.Tensor,
    candidate_key: torch.Tensor,
    candidate_value: torch.Tensor,
    retained: torch.Tensor,
) -> LocalAttentionMetrics:
    """Compare a compressed prefix against full attention for held-out queries."""

    query_heads = future_query.shape[1]
    kv_heads = full_key.shape[1]
    groups = query_heads // kv_heads
    head_dim = future_query.shape[-1]
    full_k = full_key.repeat_interleave(groups, dim=1).float()
    full_v = full_value.repeat_interleave(groups, dim=1).float()
    cand_k = candidate_key.repeat_interleave(groups, dim=1).float()
    cand_v = candidate_value.repeat_interleave(groups, dim=1).float()
    expanded_mask = retained.repeat_interleave(groups, dim=1)

    full_logits = torch.matmul(future_query.float(), full_k.transpose(-1, -2)) / sqrt(head_dim)
    candidate_logits = torch.matmul(future_query.float(), cand_k.transpose(-1, -2)) / sqrt(head_dim)
    candidate_logits.masked_fill_(~expanded_mask.unsqueeze(-2), torch.finfo(candidate_logits.dtype).min)
    full_prob = full_logits.softmax(dim=-1)
    candidate_prob = candidate_logits.softmax(dim=-1)
    full_output = torch.matmul(full_prob, full_v)
    candidate_output = torch.matmul(candidate_prob, cand_v)

    eps = torch.finfo(torch.float32).eps
    midpoint = (full_prob + candidate_prob) / 2
    js = 0.5 * (
        (full_prob * ((full_prob + eps).log() - (midpoint + eps).log())).sum(dim=-1)
        + (candidate_prob * ((candidate_prob + eps).log() - (midpoint + eps).log())).sum(dim=-1)
    )
    error = candidate_output - full_output
    output_energy = full_output.square().mean().clamp_min(eps)
    cosine = F.cosine_similarity(full_output, candidate_output, dim=-1)
    return LocalAttentionMetrics(
        attention_tv=float((0.5 * (full_prob - candidate_prob).abs().sum(dim=-1)).mean().item()),
        attention_js=float(js.mean().item()),
        missed_mass=float((full_prob * (~expanded_mask).unsqueeze(-2)).sum(dim=-1).mean().item()),
        top1_flip_rate=float(full_prob.argmax(dim=-1).ne(candidate_prob.argmax(dim=-1)).float().mean().item()),
        output_relative_mse=float((error.square().mean() / output_energy).item()),
        output_cosine=float(cosine.mean().item()),
    )
