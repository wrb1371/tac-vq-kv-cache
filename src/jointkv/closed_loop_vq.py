"""Differentiable closed-loop nested VQ with an exact-rate depth projector.

The deployed forward path is entirely discrete: residual-VQ assignments are
nearest-neighbour indices and every cache block takes one of three progressive
prefixes.  Training uses straight-through estimators only in the backward
pass.  Consequently the forward cache and its byte ledger are identical to
the deployable representation, while final language-model loss can update
both the codewords and the precision-depth selector.

Action zero is intentionally absent.  The current research phase studies the
strict ``{low, anchor, high}`` precision exchange before reopening eviction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nested_vq import validate_stage_bits
from .packed_nested import (
    PackedLayoutLedger,
    estimate_block_shared_paired_progressive_layout,
)


@dataclass(frozen=True)
class ProjectedDepths:
    """Hard deployable depths plus their straight-through action weights."""

    block_depths: torch.Tensor
    action_weights: torch.Tensor
    ledgers: tuple[PackedLayoutLedger, ...]
    low_blocks_per_request: tuple[int, ...]
    anchor_blocks_per_request: tuple[int, ...]
    high_blocks_per_request: tuple[int, ...]


@dataclass(frozen=True)
class QuantizedKV:
    key: torch.Tensor
    value: torch.Tensor
    key_indices: torch.Tensor
    value_indices: torch.Tensor
    projected_depths: ProjectedDepths | None


def _nearest_indices(
    states: torch.Tensor,
    table: torch.Tensor,
    *,
    token_chunk_size: int,
) -> torch.Tensor:
    """Batched exact nearest neighbours for ``[H,N,D]`` and ``[H,K,D]``."""

    if states.ndim != 3 or table.ndim != 3:
        raise ValueError("nearest-neighbour inputs must have [H,N,D]/[H,K,D]")
    if states.shape[0] != table.shape[0] or states.shape[-1] != table.shape[-1]:
        raise ValueError("nearest-neighbour geometries differ")
    if token_chunk_size <= 0:
        raise ValueError("token chunk size must be positive")
    table_norm = table.square().sum(dim=-1)
    chunks = []
    for start in range(0, states.shape[1], token_chunk_size):
        local = states[:, start : start + token_chunk_size]
        distance = (
            local.square().sum(dim=-1, keepdim=True)
            + table_norm[:, None, :]
            - 2.0 * torch.bmm(local, table.transpose(-1, -2))
        )
        chunks.append(distance.argmin(dim=-1))
    return torch.cat(chunks, dim=1)


def _nearest_indices_and_topk(
    states: torch.Tensor,
    table: torch.Tensor,
    *,
    topk: int,
    token_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact hard NN indices and a hard-containing top-k shortlist.

    The distance arithmetic and ``argmin`` are deliberately identical to
    :func:`_nearest_indices`.  The shortlist is backward-only metadata; ties
    are repaired so the exact hard index is always represented.
    """

    if states.ndim != 3 or table.ndim != 3:
        raise ValueError("nearest-neighbour inputs must have [H,N,D]/[H,K,D]")
    if states.shape[0] != table.shape[0] or states.shape[-1] != table.shape[-1]:
        raise ValueError("nearest-neighbour geometries differ")
    if token_chunk_size <= 0:
        raise ValueError("token chunk size must be positive")
    if topk <= 1 or topk > table.shape[1]:
        raise ValueError("soft-backward top-k lies outside the codebook")
    table_norm = table.square().sum(dim=-1)
    hard_chunks = []
    candidate_chunks = []
    for start in range(0, states.shape[1], token_chunk_size):
        local = states[:, start : start + token_chunk_size]
        distance = (
            local.square().sum(dim=-1, keepdim=True)
            + table_norm[:, None, :]
            - 2.0 * torch.bmm(local, table.transpose(-1, -2))
        )
        hard = distance.argmin(dim=-1)
        candidates = torch.topk(
            distance, k=topk, dim=-1, largest=False, sorted=True
        ).indices
        # ``topk`` and ``argmin`` can select different representatives of an
        # exact tie.  Preserve the hard action by replacing only the last
        # shortlist entry in that exceptional case.
        missing = ~(candidates == hard[..., None]).any(dim=-1)
        candidates[..., -1] = torch.where(
            missing, hard, candidates[..., -1]
        )
        hard_chunks.append(hard)
        candidate_chunks.append(candidates)
    return torch.cat(hard_chunks, dim=1), torch.cat(candidate_chunks, dim=1)


def _gather_words(table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[H,N,...]`` word indices from one ``[H,K,D]`` table."""

    heads, cardinality, dimension = map(int, table.shape)
    if indices.shape[0] != heads:
        raise ValueError("word indices and table heads differ")
    flat = indices.reshape(heads, -1)
    gathered = torch.gather(
        table,
        1,
        flat[..., None].expand(-1, -1, dimension),
    )
    return gathered.reshape(*indices.shape, dimension)


class _HardForwardTopKSoftBackward(torch.autograd.Function):
    """Exact hard gather with a recomputed top-k soft-assignment backward.

    Forward returns the two hard-selected words directly, with no numerical
    correction term.  Backward recomputes the local soft action

    ``hard + beta * (soft - stop_gradient(soft))``

    and evaluates its VJP.  Recomputing avoids retaining ``[H,N,K,D]``
    candidate tensors for every residual stage.  ``raw_table`` supplies the
    decoded word, while ``search_table`` and ``search_residual`` define GSRQ's
    (possibly whitened) assignment geometry.
    """

    @staticmethod
    def forward(
        ctx,
        raw_table: torch.Tensor,
        search_table: torch.Tensor,
        search_residual: torch.Tensor,
        hard_indices: torch.Tensor,
        candidate_indices: torch.Tensor,
        temperature: float,
        beta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(
            raw_table,
            search_table,
            search_residual,
            hard_indices,
            candidate_indices,
        )
        ctx.temperature = float(temperature)
        ctx.beta = float(beta)
        # Returning the literal hard gathers is the bitwise deployment
        # contract.  The soft action exists only in ``backward`` below.
        return (
            _gather_words(raw_table, hard_indices),
            _gather_words(search_table, hard_indices),
        )

    @staticmethod
    def backward(ctx, grad_raw, grad_search):
        (
            saved_raw_table,
            saved_search_table,
            saved_search_residual,
            hard_indices,
            candidate_indices,
        ) = ctx.saved_tensors
        with torch.enable_grad():
            raw_table = saved_raw_table.detach().requires_grad_(True)
            search_table = saved_search_table.detach().requires_grad_(True)
            search_residual = saved_search_residual.detach().requires_grad_(True)
            raw_hard = _gather_words(raw_table, hard_indices)
            search_hard = _gather_words(search_table, hard_indices)
            raw_candidates = _gather_words(raw_table, candidate_indices)
            search_candidates = _gather_words(search_table, candidate_indices)
            distance = (
                search_residual[..., None, :] - search_candidates
            ).square().sum(dim=-1)
            probability = torch.softmax(-distance / ctx.temperature, dim=-1)
            raw_soft = (probability[..., None] * raw_candidates).sum(dim=-2)
            search_soft = (probability[..., None] * search_candidates).sum(dim=-2)
            # The detached half of the ST expression has zero derivative, so
            # this is its exact VJP while avoiding a fake soft forward value.
            raw_surrogate = raw_hard + ctx.beta * raw_soft
            search_surrogate = search_hard + ctx.beta * search_soft
            raw_gradient = (
                torch.zeros_like(raw_surrogate) if grad_raw is None else grad_raw
            )
            search_gradient = (
                torch.zeros_like(search_surrogate)
                if grad_search is None
                else grad_search
            )
            table_raw_grad, table_search_grad, residual_grad = torch.autograd.grad(
                outputs=(raw_surrogate, search_surrogate),
                inputs=(raw_table, search_table, search_residual),
                grad_outputs=(raw_gradient, search_gradient),
                allow_unused=False,
            )
        return (
            table_raw_grad,
            table_search_grad,
            residual_grad,
            None,
            None,
            None,
            None,
        )


def hard_residual_vq_exits(
    states: torch.Tensor,
    tables: torch.Tensor,
    *,
    prefixes: Sequence[int],
    assignment_whiteners: torch.Tensor | None = None,
    token_chunk_size: int = 1024,
    straight_through_states: bool = True,
    soft_backward_topk: int = 1,
    soft_backward_temperature: float = 0.0,
    soft_backward_beta: float = 0.0,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Encode hard residual-VQ indices and return differentiable exits.

    ``tables`` has ``[H,S,K,D]`` layout.  Index search is deliberately outside
    autograd.  Selected codewords remain in the graph, and an identity STE on
    the input states keeps the full transformer trajectory differentiable
    across successive quantized layers.  When ``soft_backward_beta > 0``, the
    literal hard gather remains the forward action while its backward is
    augmented by a top-k soft-assignment surrogate.  Setting beta to zero is
    the historical hard baseline and does not build the surrogate graph.
    """

    if states.ndim != 4 or tables.ndim != 4:
        raise ValueError("states/tables must have [B,H,T,D]/[H,S,K,D]")
    batch, heads, tokens, dimension = map(int, states.shape)
    if tables.shape[0] != heads or tables.shape[-1] != dimension:
        raise ValueError("state and table geometry differs")
    requested = tuple(sorted(set(int(depth) for depth in prefixes)))
    stages = int(tables.shape[1])
    if not requested or requested[0] <= 0 or requested[-1] > stages:
        raise ValueError("requested prefixes lie outside the VQ ladder")
    if assignment_whiteners is not None and tuple(assignment_whiteners.shape) != (
        heads,
        dimension,
        dimension,
    ):
        raise ValueError("assignment whitener has invalid geometry")
    if soft_backward_topk <= 0 or soft_backward_topk > tables.shape[2]:
        raise ValueError("soft-backward top-k lies outside the codebook")
    if not math.isfinite(soft_backward_beta) or soft_backward_beta < 0:
        raise ValueError("soft-backward beta must be finite and non-negative")
    soft_backward_enabled = soft_backward_beta > 0
    if soft_backward_enabled and soft_backward_topk <= 1:
        raise ValueError("positive soft-backward beta requires top-k > 1")
    if soft_backward_enabled and (
        not math.isfinite(soft_backward_temperature)
        or soft_backward_temperature <= 0
    ):
        raise ValueError("soft-backward temperature must be finite and positive")

    # Search follows the legacy SCFW/GSRQ recursion literally: transform the
    # original K once and then subtract transformed selected codewords.
    detached_states = states.detach().float()
    raw_residual = (
        detached_states.permute(1, 0, 2, 3)
        .reshape(heads, batch * tokens, dimension)
        .contiguous()
        .clone()
    )
    # The persistent artifact stores FP16 codewords.  Use quantization-aware
    # masters so training forward is exactly the FP16-roundtripped deployment
    # table while backward still updates an FP32 parameter.
    deployed_tables = tables + (tables.half().float() - tables).detach()
    detached_tables = deployed_tables.detach().float()
    selector_residual = raw_residual
    selector_tables = detached_tables
    live_selector_residual = None
    live_selector_tables = None
    if soft_backward_enabled:
        live_selector_residual = (
            states.float()
            .permute(1, 0, 2, 3)
            .reshape(heads, batch * tokens, dimension)
            .contiguous()
        )
        live_selector_tables = deployed_tables.float()
    if assignment_whiteners is not None:
        whiteners = assignment_whiteners.detach().float()
        selector_residual = (
            torch.einsum("hde,bhte->bhtd", whiteners, detached_states)
            .permute(1, 0, 2, 3)
            .reshape(heads, batch * tokens, dimension)
            .contiguous()
        )
        selector_tables = torch.einsum("hrke,hde->hrkd", detached_tables, whiteners)
        if soft_backward_enabled:
            live_selector_residual = (
                torch.einsum("hde,bhte->bhtd", whiteners, states.float())
                .permute(1, 0, 2, 3)
                .reshape(heads, batch * tokens, dimension)
                .contiguous()
            )
            live_selector_tables = torch.einsum(
                "hrke,hde->hrkd", deployed_tables.float(), whiteners
            )

    decoded = torch.zeros(
        (heads, batch * tokens, dimension),
        device=states.device,
        dtype=torch.float32,
    )
    indices = torch.empty(
        (batch, heads, tokens, stages),
        device=states.device,
        dtype=torch.uint8,
    )
    outputs: dict[int, torch.Tensor] = {}
    for stage in range(stages):
        search_table = selector_tables[:, stage]
        with torch.no_grad():
            if soft_backward_enabled:
                local_indices, candidate_indices = _nearest_indices_and_topk(
                    selector_residual,
                    search_table,
                    topk=soft_backward_topk,
                    token_chunk_size=token_chunk_size,
                )
            else:
                local_indices = _nearest_indices(
                    selector_residual,
                    search_table,
                    token_chunk_size=token_chunk_size,
                )
        # Gather from the live FP32 master table so final LM loss reaches every
        # selected codeword even though assignment itself is hard.
        local_table = deployed_tables[:, stage].float()
        if soft_backward_enabled:
            assert live_selector_tables is not None
            assert live_selector_residual is not None
            selected, live_selector_selected = _HardForwardTopKSoftBackward.apply(
                local_table,
                live_selector_tables[:, stage],
                live_selector_residual,
                local_indices,
                candidate_indices,
                soft_backward_temperature,
                soft_backward_beta,
            )
            live_selector_residual = live_selector_residual - live_selector_selected
        else:
            selected = torch.gather(
                local_table,
                1,
                local_indices[..., None].expand(-1, -1, dimension),
            )
        decoded = decoded + selected
        with torch.no_grad():
            raw_residual.sub_(selected.detach())
            if assignment_whiteners is not None:
                selector_selected = torch.gather(
                    search_table,
                    1,
                    local_indices[..., None].expand(-1, -1, dimension),
                )
                selector_residual.sub_(selector_selected)
            indices[..., stage] = (
                local_indices.reshape(heads, batch, tokens)
                .permute(1, 0, 2)
                .to(torch.uint8)
            )
        depth = stage + 1
        if depth in requested:
            reconstructed = (
                decoded.reshape(heads, batch, tokens, dimension)
                .permute(1, 0, 2, 3)
                .contiguous()
                .to(states.dtype)
            )
            if straight_through_states:
                # Forward value is exactly ``reconstructed``.  Backward has an
                # identity path to the incoming hidden trajectory as well as
                # the ordinary path to selected codewords.
                reconstructed = reconstructed + (states - states.detach())
            outputs[depth] = reconstructed
    return outputs, indices


def anchor_effect_residual_vq_exits(
    states: torch.Tensor,
    anchor_tables: torch.Tensor,
    effect_tables: torch.Tensor,
    *,
    prefixes: Sequence[int],
    assignment_whiteners: torch.Tensor | None = None,
    token_chunk_size: int = 1024,
    straight_through_states: bool = True,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Encode with frozen anchors and decode the same indices with effects.

    Both tables use ``[H,S,K,D]`` layout.  The anchor table alone determines
    every hard index and every residual passed to the next search stage:

    ``u_s = u_(s-1) - A_s[z_s]``.

    The transformer-visible reconstruction is instead
    ``sum_s B_s[z_s]``.  Anchor and effect tables are independently
    FP16-roundtripped so the forward matches the deployable encoder/decoder.
    No gradient path exists from the effect table to the current layer's
    anchor search; selected effect words and the identity state STE remain
    differentiable for full-model closed-loop training.
    """

    if states.ndim != 4 or anchor_tables.ndim != 4 or effect_tables.ndim != 4:
        raise ValueError("states/tables must have [B,H,T,D]/[H,S,K,D]")
    if anchor_tables.shape != effect_tables.shape:
        raise ValueError("anchor and effect table geometries differ")
    batch, heads, tokens, dimension = map(int, states.shape)
    if anchor_tables.shape[0] != heads or anchor_tables.shape[-1] != dimension:
        raise ValueError("state and table geometry differs")
    requested = tuple(sorted(set(int(depth) for depth in prefixes)))
    stages = int(anchor_tables.shape[1])
    if not requested or requested[0] <= 0 or requested[-1] > stages:
        raise ValueError("requested prefixes lie outside the VQ ladder")
    if assignment_whiteners is not None and tuple(assignment_whiteners.shape) != (
        heads,
        dimension,
        dimension,
    ):
        raise ValueError("assignment whitener has invalid geometry")

    detached_states = states.detach().float()
    raw_anchor_residual = (
        detached_states.permute(1, 0, 2, 3)
        .reshape(heads, batch * tokens, dimension)
        .contiguous()
        .clone()
    )
    # Anchors are frozen buffers; effects are FP32 masters with an FP16
    # deployment-forward / identity-backward bridge.
    deployed_anchor = anchor_tables.detach().half().float()
    deployed_effect = effect_tables + (
        effect_tables.half().float() - effect_tables
    ).detach()
    selector_residual = raw_anchor_residual
    selector_tables = deployed_anchor
    if assignment_whiteners is not None:
        whiteners = assignment_whiteners.detach().float()
        selector_residual = (
            torch.einsum("hde,bhte->bhtd", whiteners, detached_states)
            .permute(1, 0, 2, 3)
            .reshape(heads, batch * tokens, dimension)
            .contiguous()
        )
        selector_tables = torch.einsum(
            "hrke,hde->hrkd", deployed_anchor, whiteners
        )

    decoded = torch.zeros(
        (heads, batch * tokens, dimension),
        device=states.device,
        dtype=torch.float32,
    )
    indices = torch.empty(
        (batch, heads, tokens, stages),
        device=states.device,
        dtype=torch.uint8,
    )
    outputs: dict[int, torch.Tensor] = {}
    for stage in range(stages):
        search_table = selector_tables[:, stage]
        with torch.no_grad():
            local_indices = _nearest_indices(
                selector_residual,
                search_table,
                token_chunk_size=token_chunk_size,
            )
        anchor_word = torch.gather(
            deployed_anchor[:, stage],
            1,
            local_indices[..., None].expand(-1, -1, dimension),
        )
        effect_word = torch.gather(
            deployed_effect[:, stage].float(),
            1,
            local_indices[..., None].expand(-1, -1, dimension),
        )
        decoded = decoded + effect_word
        with torch.no_grad():
            raw_anchor_residual.sub_(anchor_word)
            if assignment_whiteners is not None:
                selector_word = torch.gather(
                    search_table,
                    1,
                    local_indices[..., None].expand(-1, -1, dimension),
                )
                selector_residual.sub_(selector_word)
            indices[..., stage] = (
                local_indices.reshape(heads, batch, tokens)
                .permute(1, 0, 2)
                .to(torch.uint8)
            )
        depth = stage + 1
        if depth in requested:
            reconstructed = (
                decoded.reshape(heads, batch, tokens, dimension)
                .permute(1, 0, 2, 3)
                .contiguous()
                .to(states.dtype)
            )
            if straight_through_states:
                reconstructed = reconstructed + (states - states.detach())
            outputs[depth] = reconstructed
    return outputs, indices


def block_selector_features(
    raw_key: torch.Tensor,
    raw_value: torch.Tensor,
    key_exits: dict[int, torch.Tensor],
    value_exits: dict[int, torch.Tensor],
    *,
    exits: Sequence[int],
    block_size: int,
    epsilon: float = 1e-8,
    feature_mode: str = "legacy10",
    attention_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return causal codec features in ``[B,H,N_blocks,F]`` layout.

    Features use only current block projections/reconstruction residuals and
    position, so the learned selector does not require teacher logits or
    labels at deployment.  They are detached intentionally: codewords learn
    from the final LM path, while selector parameters learn the exchange rule.
    """

    actions = tuple(int(depth) for depth in exits)
    if feature_mode not in {
        "legacy10",
        "absolute18",
        "attention26",
        "localrd34",
        "semantic530",
        "transition1554",
    }:
        raise ValueError("unknown precision-selector feature mode")
    if len(actions) != 3 or tuple(sorted(set(actions))) != actions:
        raise ValueError("selector requires three sorted exits")
    if raw_key.shape != raw_value.shape or raw_key.ndim != 4:
        raise ValueError("raw K/V geometry differs")
    if block_size <= 0:
        raise ValueError("block size must be positive")
    batch, heads, tokens, _dimension = map(int, raw_key.shape)
    if tokens % block_size:
        raise ValueError("closed-loop selector currently requires full blocks")
    blocks = tokens // block_size

    def energy(raw: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        per_token = (raw.detach().float() - reconstructed.detach().float()).square().mean(-1)
        return per_token.reshape(batch, heads, blocks, block_size).mean(-1)

    key_energy = [energy(raw_key, key_exits[depth]) for depth in actions]
    value_energy = [energy(raw_value, value_exits[depth]) for depth in actions]
    raw_key_norm = raw_key.detach().float().square().mean(-1).reshape(
        batch, heads, blocks, block_size
    ).mean(-1)
    raw_value_norm = raw_value.detach().float().square().mean(-1).reshape(
        batch, heads, blocks, block_size
    ).mean(-1)

    # Absolute anchor error, marginal value of the anchor stage, and marginal
    # value of the refinement stage.  Log ratios make the scale comparable
    # across layers and K/V heads.
    features = torch.stack(
        (
            torch.log(key_energy[1] + epsilon),
            torch.log(value_energy[1] + epsilon),
            torch.log((key_energy[0] + epsilon) / (key_energy[1] + epsilon)),
            torch.log((value_energy[0] + epsilon) / (value_energy[1] + epsilon)),
            torch.log((key_energy[1] + epsilon) / (key_energy[2] + epsilon)),
            torch.log((value_energy[1] + epsilon) / (value_energy[2] + epsilon)),
            torch.log(raw_key_norm + epsilon),
            torch.log(raw_value_norm + epsilon),
        ),
        dim=-1,
    )
    mean = features.mean(dim=2, keepdim=True)
    std = features.std(dim=2, keepdim=True, unbiased=False).clamp_min(1e-4)
    normalized = (features - mean) / std
    position = (
        (torch.arange(blocks, device=raw_key.device, dtype=torch.float32) + 0.5)
        / blocks
    )
    position = position.reshape(1, 1, blocks, 1).expand(batch, heads, -1, -1)
    core = (
        normalized
        if feature_mode == "legacy10"
        else torch.cat((normalized, features), dim=-1)
    )
    if feature_mode in {"attention26", "localrd34"}:
        prompt_dim = 8 if feature_mode == "attention26" else 16
        if attention_features is None or tuple(attention_features.shape) != (
            batch,
            heads,
            blocks,
            prompt_dim,
        ):
            raise ValueError("prompt-aware mode requires aligned [B,H,N,8] salience")
        core = torch.cat((core, attention_features), dim=-1)
    if feature_mode in {"semantic530", "transition1554"}:
        def semantic_moments(states: torch.Tensor) -> torch.Tensor:
            """Signed channel summaries that retain transition direction."""

            local = states.detach().float().reshape(
                batch, heads, blocks, block_size, -1
            )
            block_mean = local.mean(dim=3)
            block_rms = torch.log(local.square().mean(dim=3) + epsilon)
            moments = torch.cat((block_mean, block_rms), dim=-1)
            request_mean = moments.mean(dim=2, keepdim=True)
            request_std = moments.std(
                dim=2, keepdim=True, unbiased=False
            ).clamp_min(1e-4)
            return (moments - request_mean) / request_std

        core = torch.cat(
            (core, semantic_moments(raw_key), semantic_moments(raw_value)), dim=-1
        )
        if feature_mode == "transition1554":
            low, anchor, high = actions
            # The oracle margin is an inner product between an unknown LM
            # sensitivity and exactly these signed transition tensors.
            # Earlier features kept norms but discarded transition direction.
            # These summaries are causal and contain no teacher gradients.
            core = torch.cat(
                (
                    core,
                    semantic_moments(key_exits[low] - key_exits[anchor]),
                    semantic_moments(value_exits[low] - value_exits[anchor]),
                    semantic_moments(key_exits[high] - key_exits[anchor]),
                    semantic_moments(value_exits[high] - value_exits[anchor]),
                ),
                dim=-1,
            )
    return torch.cat((core, 2.0 * position - 1.0, position.square()), dim=-1)


@torch.no_grad()
def sampled_prompt_attention_features(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    block_size: int,
    checkpoints: int = 4,
    observation_window: int = 32,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Bounded prompt-aware salience in ``[B,KVH,N_blocks,8]`` layout.

    Four raw features are computed from queries already present in the prompt:
    total mass over checkpoints, maximum checkpoint mass, final-checkpoint
    mass, and within-block token concentration.  Their per-head block z-scores
    are concatenated with the absolute log values.  No future generation,
    target labels, loss gradients, or oracle quantities are used.
    """

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("prompt salience requires [B,H,T,D] query/key tensors")
    batch, query_heads, tokens, dimension = map(int, query.shape)
    if key.shape[0] != batch or key.shape[2:] != (tokens, dimension):
        raise ValueError("prompt salience query/key geometry differs")
    kv_heads = int(key.shape[1])
    if query_heads % kv_heads or tokens % block_size:
        raise ValueError("prompt salience requires GQA-compatible full blocks")
    if checkpoints <= 0 or observation_window <= 0:
        raise ValueError("prompt salience sampling geometry must be positive")
    groups = query_heads // kv_heads
    ends = sorted(
        {
            max(observation_window, round(tokens * (index + 1) / checkpoints))
            for index in range(checkpoints)
        }
    )
    block_rows = []
    token_rows = []
    for end in ends:
        width = min(observation_window, end)
        begin = end - width
        local_query = query[:, :, begin:end].float()
        prefix_key = key[:, :, :end].repeat_interleave(groups, dim=1).float()
        logits = torch.matmul(local_query, prefix_key.transpose(-1, -2)) / math.sqrt(
            dimension
        )
        query_positions = torch.arange(begin, end, device=query.device)
        key_positions = torch.arange(end, device=query.device)
        causal = key_positions.reshape(1, 1, 1, end) > query_positions.reshape(
            1, 1, width, 1
        )
        logits.masked_fill_(causal, torch.finfo(logits.dtype).min)
        mass = logits.softmax(dim=-1).sum(dim=-2)
        mass = mass.reshape(batch, kv_heads, groups, end).sum(dim=2)
        padded = torch.zeros(
            batch, kv_heads, tokens, dtype=torch.float32, device=query.device
        )
        padded[..., :end] = mass
        token_rows.append(padded)
        block_rows.append(
            padded.reshape(batch, kv_heads, tokens // block_size, block_size).sum(-1)
        )
    checkpoint_mass = torch.stack(block_rows, dim=-1)
    token_mass = torch.stack(token_rows, dim=-1).sum(dim=-1)
    total = checkpoint_mass.sum(dim=-1)
    maximum = checkpoint_mass.max(dim=-1).values
    final = checkpoint_mass[..., -1]
    peak = token_mass.reshape(
        batch, kv_heads, tokens // block_size, block_size
    ).max(dim=-1).values
    concentration = peak / total.clamp_min(epsilon)
    absolute = torch.stack(
        (
            torch.log(total + epsilon),
            torch.log(maximum + epsilon),
            torch.log(final + epsilon),
            torch.log(concentration + epsilon),
        ),
        dim=-1,
    )
    mean = absolute.mean(dim=2, keepdim=True)
    std = absolute.std(dim=2, keepdim=True, unbiased=False).clamp_min(1e-4)
    return torch.cat(((absolute - mean) / std, absolute), dim=-1)


@torch.no_grad()
def sampled_prompt_local_rd_features(
    query: torch.Tensor,
    anchor_key: torch.Tensor,
    anchor_value: torch.Tensor,
    key_candidates: torch.Tensor,
    value_candidates: torch.Tensor,
    *,
    block_size: int,
    checkpoints: int = 4,
    observation_window: int = 32,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Prompt-only local RD features in ``[B,KVH,N_blocks,16]``.

    Each sampled query window measures the exact attention-output distortion
    when one uniform-16 token is counterfactually replaced by a 15/16/17 VQ
    reconstruction.  The calculation includes key-induced softmax
    renormalization and the paired value change.  It is still deployable:
    all queries, keys, values, and candidates belong to the current prompt.
    """

    from .local_attention import single_token_pair_action_distortions

    if anchor_key.shape != anchor_value.shape or anchor_key.ndim != 4:
        raise ValueError("local RD anchor K/V geometry differs")
    if key_candidates.shape != value_candidates.shape or key_candidates.ndim != 5:
        raise ValueError("local RD candidates require [A,B,H,T,D]")
    if key_candidates.shape[0] != 3 or key_candidates.shape[1:] != anchor_key.shape:
        raise ValueError("local RD requires three aligned VQ actions")
    batch, heads, tokens, _dimension = anchor_key.shape
    if tokens % block_size or checkpoints <= 0 or observation_window <= 0:
        raise ValueError("invalid local RD sampling geometry")
    ends = sorted(
        {
            max(observation_window, round(tokens * (index + 1) / checkpoints))
            for index in range(checkpoints)
        }
    )
    total = torch.zeros(
        batch, heads, tokens, 3, dtype=torch.float64, device=query.device
    )
    square = torch.zeros_like(total)
    count = torch.zeros(batch, heads, tokens, 1, dtype=torch.float64, device=query.device)
    for end in ends:
        width = min(observation_window, end)
        local = single_token_pair_action_distortions(
            query[:, :, :end],
            anchor_key[:, :, :end],
            anchor_value[:, :, :end],
            key_candidates[:, :, :, :end],
            value_candidates[:, :, :, :end],
            prefix_tokens=end,
            observation_window=width,
        )[..., 1:].double() / width
        total[:, :, :end] += local
        square[:, :, :end] += local.square()
        count[:, :, :end] += 1
    mean = total / count.clamp_min(1)
    std = (square / count.clamp_min(1) - mean.square()).clamp_min(0).sqrt()
    blocks = tokens // block_size
    mean = mean.reshape(batch, heads, blocks, block_size, 3).mean(dim=3)
    std = std.reshape(batch, heads, blocks, block_size, 3).mean(dim=3)
    absolute = torch.stack(
        (
            torch.log(mean[..., 0] + epsilon),
            torch.log(mean[..., 1] + epsilon),
            torch.log(mean[..., 2] + epsilon),
            torch.log((mean[..., 0] + epsilon) / (mean[..., 1] + epsilon)),
            torch.log((mean[..., 1] + epsilon) / (mean[..., 2] + epsilon)),
            torch.log(std[..., 0] + epsilon),
            torch.log(std[..., 1] + epsilon),
            torch.log(std[..., 2] + epsilon),
        ),
        dim=-1,
    ).float()
    feature_mean = absolute.mean(dim=2, keepdim=True)
    feature_std = absolute.std(dim=2, keepdim=True, unbiased=False).clamp_min(1e-4)
    return torch.cat(((absolute - feature_mean) / feature_std, absolute), dim=-1)


def _fixed_quota_assignment(
    logits: torch.Tensor,
    *,
    low_count: int,
    high_count: int,
) -> torch.Tensor:
    """Greedy two-order solution for one fixed-quota three-action problem."""

    if logits.ndim != 2 or logits.shape[-1] != 3:
        raise ValueError("one-request logits must have [N,3] layout")
    units = int(logits.shape[0])
    if min(low_count, high_count) < 0 or low_count + high_count > units:
        raise ValueError("invalid low/high action quotas")
    low_advantage = logits[:, 0] - logits[:, 1]
    high_advantage = logits[:, 2] - logits[:, 1]

    def candidate(first_low: bool) -> torch.Tensor:
        actions = torch.ones(units, device=logits.device, dtype=torch.long)
        available = torch.ones(units, device=logits.device, dtype=torch.bool)
        order = ((0, low_count, low_advantage), (2, high_count, high_advantage))
        if not first_low:
            order = (order[1], order[0])
        for action, count, advantage in order:
            if count == 0:
                continue
            masked = advantage.masked_fill(~available, -torch.inf)
            chosen = torch.topk(masked, k=count, largest=True, sorted=False).indices
            actions[chosen] = action
            available[chosen] = False
        return actions

    low_first = candidate(True)
    high_first = candidate(False)

    def objective(actions: torch.Tensor) -> torch.Tensor:
        return logits.gather(1, actions[:, None]).sum()

    return high_first if objective(high_first) > objective(low_first) else low_first


def _exact_fixed_quota_assignment(
    logits: torch.Tensor,
    *,
    low_count: int,
    high_count: int,
) -> torch.Tensor:
    """Exact disjoint three-action assignment under fixed low/high quotas.

    This CPU dynamic program is the deployment counterpart of the offline
    oracle allocator.  It consumes only selector scores; no labels, gradients,
    or future tokens enter this function.
    """

    if logits.ndim != 2 or logits.shape[-1] != 3:
        raise ValueError("one-request logits must have [N,3] layout")
    units = int(logits.shape[0])
    if min(low_count, high_count) < 0 or low_count + high_count > units:
        raise ValueError("invalid low/high action quotas")
    # The fixed-cardinality three-action problem is equivalent to selecting
    # disjoint low/high sets.  For the small deployment quota, enumerate the
    # low membership of the only possible overlap between the independent
    # top sets; all non-overlap selections are fixed.  This is exact and avoids
    # a 512 x low_count x high_count Python DP during every validation row.
    local = logits.detach().double().cpu()
    low_gain = local[:, 0] - local[:, 1]
    high_gain = local[:, 2] - local[:, 1]
    if low_count == 0 or high_count == 0:
        actions = torch.ones(units, dtype=torch.long)
        action = 0 if low_count else 2
        score = low_gain if low_count else high_gain
        count = low_count or high_count
        actions[torch.topk(score, k=count, largest=True, sorted=False).indices] = action
        return actions.to(logits.device)
    low_order = torch.argsort(low_gain, descending=True)
    high_order = torch.argsort(high_gain, descending=True)
    # Any optimal low item must occur in the top low_count+high_count low
    # scores: at most high_count better candidates can be occupied by high.
    candidates = low_order[: low_count + high_count]
    best_value = -torch.inf
    best_low: torch.Tensor | None = None
    best_high: torch.Tensor | None = None
    import itertools

    if math.comb(low_count + high_count, low_count) <= 200_000:
        for chosen_tuple in itertools.combinations(candidates.tolist(), low_count):
            chosen_low = torch.tensor(chosen_tuple, dtype=torch.long)
            available_high = torch.ones(units, dtype=torch.bool)
            available_high[chosen_low] = False
            chosen_high = high_order[available_high[high_order]][:high_count]
            value = low_gain[chosen_low].sum() + high_gain[chosen_high].sum()
            if value > best_value:
                best_value = value
                best_low = chosen_low
                best_high = chosen_high
    if best_low is not None and best_high is not None:
        actions = torch.ones(units, dtype=torch.long)
        actions[best_low] = 0
        actions[best_high] = 2
        return actions.to(logits.device)

    # Defensive general-quota fallback (the loop above is nonempty for all
    # valid positive quotas).
    dp = torch.full((low_count + 1, high_count + 1), -torch.inf, dtype=torch.float64)
    dp[0, 0] = 0.0
    trace = torch.full(
        (units, low_count + 1, high_count + 1), -1, dtype=torch.int8
    )
    for unit in range(units):
        new = dp.clone()
        choice = torch.zeros_like(dp, dtype=torch.int8)
        low_candidate = dp[:-1] + low_gain[unit]
        use_low = low_candidate > new[1:]
        new[1:] = torch.where(use_low, low_candidate, new[1:])
        choice[1:] = torch.where(use_low, torch.ones_like(choice[1:]), choice[1:])
        high_candidate = dp[:, :-1] + high_gain[unit]
        use_high = high_candidate > new[:, 1:]
        new[:, 1:] = torch.where(use_high, high_candidate, new[:, 1:])
        choice[:, 1:] = torch.where(
            use_high, torch.full_like(choice[:, 1:], 2), choice[:, 1:]
        )
        choice[~torch.isfinite(new)] = -1
        trace[unit] = choice
        dp = new
    if not torch.isfinite(dp[low_count, high_count]):
        raise RuntimeError("exact quota target is unreachable")
    actions = torch.ones(units, dtype=torch.long)
    low = int(low_count)
    high = int(high_count)
    for unit in range(units - 1, -1, -1):
        choice = int(trace[unit, low, high])
        if choice == 0:
            continue
        if choice == 1:
            actions[unit] = 0
            low -= 1
        elif choice == 2:
            actions[unit] = 2
            high -= 1
        else:
            raise RuntimeError("invalid exact quota backpointer")
    if low or high:
        raise RuntimeError("exact quota backtracking did not discharge quotas")
    return actions.to(logits.device)


def project_logits_to_exact_rate(
    logits: torch.Tensor,
    *,
    stage_bits: Sequence[int],
    exits: Sequence[int],
    tokens_per_block: int,
    head_dim: int,
    effective_bits: float,
    trust_region_fraction: float,
    temperature: float = 1.0,
    assignment_solver: str = "greedy_two_order",
) -> ProjectedDepths:
    """Project selector logits to a strict per-request physical byte cap.

    The current parent uses equal full-size blocks.  The quota has the largest
    low/high exchange permitted by the trust region while retaining enough
    extra low blocks to pay the packed two-bit action labels.  Each request is
    projected and audited independently.
    """

    schedule = validate_stage_bits(stage_bits)
    actions = tuple(int(depth) for depth in exits)
    if len(actions) != 3 or tuple(sorted(set(actions))) != actions:
        raise ValueError("exact-rate projection requires three sorted exits")
    if logits.ndim != 4 or logits.shape[-1] != 3:
        raise ValueError("selector logits must have [B,H,N_blocks,3] layout")
    if tokens_per_block <= 0 or head_dim <= 0 or effective_bits <= 0:
        raise ValueError("invalid cache/rate geometry")
    if not 0.0 <= trust_region_fraction <= 1.0 or temperature <= 0:
        raise ValueError("invalid trust fraction or temperature")
    if assignment_solver not in {"greedy_two_order", "exact_dp"}:
        raise ValueError("unknown precision-depth assignment solver")
    if not (actions[0] + actions[2] == 2 * actions[1]):
        raise ValueError("the precision exchange requires symmetric exits")
    if any(bit != schedule[0] for bit in schedule):
        raise ValueError("the current exact quota requires equal stage widths")

    batch, heads, blocks, _ = map(int, logits.shape)
    units = heads * blocks
    reference_scalars = 2 * units * tokens_per_block * head_dim
    budget_bits = math.floor(effective_bits * reference_scalars)
    anchor_payload = 2 * units * tokens_per_block * sum(schedule[: actions[1]])
    allocated_label_bits = math.ceil((2 * units) / 8) * 8
    one_step_bits = 2 * tokens_per_block * schedule[actions[1] - 1]
    mandatory_net_low = max(
        0,
        math.ceil((anchor_payload + allocated_label_bits - budget_bits) / one_step_bits),
    )
    maximum_changed = max(
        mandatory_net_low,
        math.floor(trust_region_fraction * units),
    )
    if maximum_changed > units or mandatory_net_low > units:
        raise ValueError("rate cap cannot be met by the low exit")
    # low - high >= mandatory_net_low, maximize low+high under the trust cap.
    high_count = max(0, (maximum_changed - mandatory_net_low) // 2)
    low_count = high_count + mandatory_net_low

    hard_actions = []
    ledgers = []
    lows = []
    anchors = []
    highs = []
    token_counts = torch.full((blocks,), tokens_per_block, dtype=torch.int64)
    for request in range(batch):
        flat = logits[request].detach().reshape(units, 3)
        solver = (
            _exact_fixed_quota_assignment
            if assignment_solver == "exact_dp"
            else _fixed_quota_assignment
        )
        assigned = solver(
            flat,
            low_count=low_count,
            high_count=high_count,
        ).reshape(heads, blocks)
        depth = torch.empty_like(assigned)
        for action, value in enumerate(actions):
            depth[assigned.eq(action)] = value
        ledger = estimate_block_shared_paired_progressive_layout(
            depth[None].cpu(),
            token_counts,
            stage_bits=schedule,
            exits=actions,
            head_dim=head_dim,
        )
        if ledger.total_bits > budget_bits:
            raise RuntimeError(
                f"exact-rate projector exceeded request cap: {ledger.total_bits}>{budget_bits}"
            )
        hard_actions.append(assigned)
        ledgers.append(ledger)
        lows.append(int(assigned.eq(0).sum()))
        anchors.append(int(assigned.eq(1).sum()))
        highs.append(int(assigned.eq(2).sum()))
    hard_action_tensor = torch.stack(hard_actions)
    hard_one_hot = F.one_hot(hard_action_tensor, num_classes=3).to(logits.dtype)
    soft = F.softmax(logits / temperature, dim=-1)
    straight_through = hard_one_hot + (soft - soft.detach())
    block_depths = torch.empty_like(hard_action_tensor)
    for action, value in enumerate(actions):
        block_depths[hard_action_tensor.eq(action)] = value
    return ProjectedDepths(
        block_depths=block_depths,
        action_weights=straight_through,
        ledgers=tuple(ledgers),
        low_blocks_per_request=tuple(lows),
        anchor_blocks_per_request=tuple(anchors),
        high_blocks_per_request=tuple(highs),
    )


class PrecisionDepthSelector(nn.Module):
    """Small model-static selector shared across requests and token blocks."""

    feature_dim = 10

    def __init__(self, layers: int, heads: int) -> None:
        super().__init__()
        if layers <= 0 or heads <= 0:
            raise ValueError("selector layer/head geometry must be positive")
        self.weight = nn.Parameter(torch.zeros(layers, heads, self.feature_dim, 3))
        self.bias = nn.Parameter(torch.zeros(layers, heads, 3))
        with torch.no_grad():
            # Start from codec geometry, then let full-model CE learn how much
            # that geometry correlates with downstream importance.
            self.weight[..., 2, 0] = -0.5
            self.weight[..., 3, 0] = -0.5
            self.weight[..., 4, 2] = 0.5
            self.weight[..., 5, 2] = 0.5

    def forward(self, layer: int, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[-1] != self.feature_dim:
            raise ValueError("selector features must have [B,H,N,10] layout")
        if not 0 <= layer < self.weight.shape[0] or features.shape[1] != self.weight.shape[1]:
            raise ValueError("selector layer/head geometry differs")
        weight = self.weight[layer]
        bias = self.bias[layer]
        return torch.einsum("bhnf,hfa->bhna", features, weight) + bias[None, :, None]


class OracleDistilledPrecisionSelector(nn.Module):
    """Compact nonlinear student for signed oracle-marginal prediction.

    The two learned outputs are the gain assigned to depth 15 and depth 17;
    depth 16 is the zero reference.  Layer/head embeddings share statistical
    strength across the oracle dataset without discarding model geometry.
    """

    def __init__(
        self,
        layers: int,
        heads: int,
        *,
        hidden_dim: int = 32,
        layer_embedding_dim: int = 8,
        head_embedding_dim: int = 4,
        feature_dim: int = 10,
        local_linear: bool = False,
    ) -> None:
        super().__init__()
        if min(layers, heads, hidden_dim) <= 0:
            raise ValueError("oracle-distilled selector geometry must be positive")
        self.layers = int(layers)
        self.heads = int(heads)
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = int(feature_dim)
        self.local_linear = bool(local_linear)
        self.layer_embedding_dim = int(layer_embedding_dim)
        self.head_embedding_dim = int(head_embedding_dim)
        self.layer_embedding = nn.Embedding(layers, layer_embedding_dim)
        self.head_embedding = nn.Embedding(heads, head_embedding_dim)
        input_dim = self.feature_dim + layer_embedding_dim + head_embedding_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.layer_head_bias = nn.Parameter(torch.zeros(layers, heads, 2))
        if self.local_linear:
            self.layer_head_feature_weight = nn.Parameter(
                torch.zeros(layers, heads, feature_dim, 2)
            )

    def forward(self, layer: int, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[-1] != self.feature_dim:
            raise ValueError("selector features must have [B,H,N,10] layout")
        if not 0 <= layer < self.layers or features.shape[1] != self.heads:
            raise ValueError("oracle-distilled selector layer/head geometry differs")
        batch, heads, blocks, _ = features.shape
        layer_ids = torch.full(
            (batch, heads, blocks), layer, device=features.device, dtype=torch.long
        )
        head_ids = torch.arange(heads, device=features.device).reshape(1, heads, 1)
        head_ids = head_ids.expand(batch, -1, blocks)
        inputs = torch.cat(
            (
                features,
                self.layer_embedding(layer_ids),
                self.head_embedding(head_ids),
            ),
            dim=-1,
        )
        signed = self.network(inputs) + self.layer_head_bias[layer][None, :, None]
        if self.local_linear:
            signed = signed + torch.einsum(
                "bhnf,hfa->bhna",
                features,
                self.layer_head_feature_weight[layer],
            )
        anchor = torch.zeros_like(signed[..., :1])
        return torch.cat((signed[..., :1], anchor, signed[..., 1:]), dim=-1)


def mix_progressive_exits(
    exits_by_depth: dict[int, torch.Tensor],
    action_weights: torch.Tensor,
    *,
    exits: Sequence[int],
    block_size: int,
) -> torch.Tensor:
    """Expand block actions and mix three progressive exits."""

    actions = tuple(int(depth) for depth in exits)
    stacked = torch.stack([exits_by_depth[depth] for depth in actions], dim=-2)
    tokens = int(stacked.shape[2])
    weights = action_weights.repeat_interleave(block_size, dim=2)[:, :, :tokens]
    return (stacked * weights.unsqueeze(-1)).sum(dim=-2).to(stacked.dtype)


class ClosedLoopNestedVQ(nn.Module):
    """All-layer trainable codebooks and the learned exact-rate selector."""

    def __init__(
        self,
        key_tables: Sequence[torch.Tensor],
        value_tables: Sequence[torch.Tensor],
        key_whiteners: Sequence[torch.Tensor | None],
        *,
        stage_bits: Sequence[int],
        exits: Sequence[int],
        block_size: int,
        head_dim: int,
        effective_bits: float,
        trust_region_fraction: float,
        temperature: float = 1.0,
        token_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        if not (len(key_tables) == len(value_tables) == len(key_whiteners)):
            raise ValueError("closed-loop layer payloads differ")
        if not key_tables:
            raise ValueError("closed-loop codec has no layers")
        schedule = validate_stage_bits(stage_bits)
        actions = tuple(int(depth) for depth in exits)
        geometry = tuple(key_tables[0].shape)
        if len(geometry) != 4 or geometry[1] != len(schedule):
            raise ValueError("invalid direct codebook geometry")
        if any(tuple(table.shape) != geometry for table in (*key_tables, *value_tables)):
            raise ValueError("codebook layer geometries differ")
        if geometry[-1] != head_dim:
            raise ValueError("codebook dimension differs from head_dim")
        self.key_tables = nn.ParameterList(
            [nn.Parameter(table.detach().float().contiguous()) for table in key_tables]
        )
        self.value_tables = nn.ParameterList(
            [nn.Parameter(table.detach().float().contiguous()) for table in value_tables]
        )
        for layer, whitener in enumerate(key_whiteners):
            self.register_buffer(
                f"key_whitener_{layer:02d}",
                (
                    torch.empty(0, dtype=torch.float32)
                    if whitener is None
                    else whitener.detach().float().contiguous()
                ),
                persistent=True,
            )
        self.selector = PrecisionDepthSelector(len(key_tables), geometry[0])
        self.stage_bits = schedule
        self.exits = actions
        self.block_size = int(block_size)
        self.head_dim = int(head_dim)
        self.effective_bits = float(effective_bits)
        self.trust_region_fraction = float(trust_region_fraction)
        self.temperature = float(temperature)
        self.token_chunk_size = int(token_chunk_size)

    def whitener(self, layer: int) -> torch.Tensor | None:
        value = getattr(self, f"key_whitener_{layer:02d}")
        return None if value.numel() == 0 else value

    def forward(
        self,
        layer: int,
        raw_key: torch.Tensor,
        raw_value: torch.Tensor,
        *,
        mode: str = "mixed",
    ) -> QuantizedKV:
        if mode not in {"mixed", "uniform"}:
            raise ValueError("closed-loop mode must be mixed or uniform")
        key_exits, key_indices = hard_residual_vq_exits(
            raw_key,
            self.key_tables[layer],
            prefixes=self.exits,
            assignment_whiteners=self.whitener(layer),
            token_chunk_size=self.token_chunk_size,
        )
        value_exits, value_indices = hard_residual_vq_exits(
            raw_value,
            self.value_tables[layer],
            prefixes=self.exits,
            assignment_whiteners=None,
            token_chunk_size=self.token_chunk_size,
        )
        if mode == "uniform":
            anchor = self.exits[1]
            return QuantizedKV(
                key=key_exits[anchor],
                value=value_exits[anchor],
                key_indices=key_indices,
                value_indices=value_indices,
                projected_depths=None,
            )
        features = block_selector_features(
            raw_key,
            raw_value,
            key_exits,
            value_exits,
            exits=self.exits,
            block_size=self.block_size,
        )
        logits = self.selector(layer, features)
        projected = project_logits_to_exact_rate(
            logits,
            stage_bits=self.stage_bits,
            exits=self.exits,
            tokens_per_block=self.block_size,
            head_dim=self.head_dim,
            effective_bits=self.effective_bits,
            trust_region_fraction=self.trust_region_fraction,
            temperature=self.temperature,
        )
        return QuantizedKV(
            key=mix_progressive_exits(
                key_exits,
                projected.action_weights,
                exits=self.exits,
                block_size=self.block_size,
            ),
            value=mix_progressive_exits(
                value_exits,
                projected.action_weights,
                exits=self.exits,
                block_size=self.block_size,
            ),
            key_indices=key_indices,
            value_indices=value_indices,
            projected_depths=projected,
        )
