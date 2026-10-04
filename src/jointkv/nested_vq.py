"""Unequal-cardinality shared/factorized nested residual VQ.

Each stage owns a dictionary shared by all KV heads in one layer.  A small
head-specific low-rank residual adapts that dictionary:

    C[h, s] = shared[s] + entry_factor[h, s] @ dimension_factor[h, s]

Stage cardinalities may differ, while a prefix of the index stream remains a
valid reconstruction.  This module deliberately has no eviction action; an
optional erasure mask belongs to the training loss, not to the codec rate or
inference action set.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math

import torch
import torch.nn.functional as F


def validate_stage_bits(stage_bits: Sequence[int]) -> tuple[int, ...]:
    """Validate fixed-width stage indices and return an immutable schedule."""

    result = tuple(int(bits) for bits in stage_bits)
    if not result or any(bits <= 0 or bits > 8 for bits in result):
        raise ValueError("stage bits must lie in [1, 8]")
    return result


def prefix_index_bits(stage_bits: Sequence[int], prefixes: Iterable[int]) -> dict[int, int]:
    """Cumulative index bits for one vector at each requested prefix."""

    schedule = validate_stage_bits(stage_bits)
    requested = sorted({int(prefix) for prefix in prefixes})
    if not requested or requested[0] <= 0 or requested[-1] > len(schedule):
        raise ValueError("prefixes must lie inside the stage schedule")
    cumulative = 0
    result: dict[int, int] = {}
    requested_set = set(requested)
    for stage, bits in enumerate(schedule, start=1):
        cumulative += bits
        if stage in requested_set:
            result[stage] = cumulative
    return result


def paired_prefix_rates(stage_bits: Sequence[int], prefixes: Iterable[int]) -> dict[int, int]:
    """Payload bits for the K+V index streams of one token/head pair."""

    return {
        prefix: 2 * bits
        for prefix, bits in prefix_index_bits(stage_bits, prefixes).items()
    }


def _validate_factorized_stages(
    shared_codebooks: Sequence[torch.Tensor],
    entry_factors: Sequence[torch.Tensor],
    dimension_factors: Sequence[torch.Tensor],
    stage_bits: Sequence[int],
) -> tuple[int, int, tuple[int, ...]]:
    schedule = validate_stage_bits(stage_bits)
    if not (
        len(shared_codebooks)
        == len(entry_factors)
        == len(dimension_factors)
        == len(schedule)
    ):
        raise ValueError("factorized stage lists must match the bit schedule")
    heads = dimension = None
    cardinalities = []
    representation = None
    for stage, (shared, entry, projection, bits) in enumerate(
        zip(shared_codebooks, entry_factors, dimension_factors, schedule)
    ):
        if shared.ndim not in (2, 3) or entry.ndim != 3 or projection.ndim != 3:
            raise ValueError("invalid factorized codebook rank")
        if shared.ndim == 2:
            cardinality, local_dimension = shared.shape
            local_representation = "shared"
            shared_heads = None
        else:
            shared_heads, cardinality, local_dimension = shared.shape
            local_representation = "per_head"
        if cardinality != 1 << bits:
            raise ValueError(f"stage {stage} cardinality does not match its index bits")
        local_heads, entry_cardinality, rank = entry.shape
        projection_heads, projection_rank, projection_dimension = projection.shape
        if (
            entry_cardinality != cardinality
            or projection_heads != local_heads
            or projection_rank != rank
            or projection_dimension != local_dimension
        ):
            raise ValueError(f"stage {stage} factor geometry differs")
        if shared_heads is not None and shared_heads != local_heads:
            raise ValueError(f"stage {stage} per-head table geometry differs")
        if local_representation == "per_head" and rank != 0:
            raise ValueError("direct per-head tables cannot also have low-rank adapters")
        if rank < 0 or rank > min(cardinality, local_dimension):
            raise ValueError(f"stage {stage} factor rank is invalid")
        if representation is None:
            representation = local_representation
        elif representation != local_representation:
            raise ValueError("nested stages cannot mix shared and per-head representations")
        if heads is None:
            heads, dimension = local_heads, local_dimension
        elif (heads, dimension) != (local_heads, local_dimension):
            raise ValueError("head/dimension geometry must be constant across stages")
        if not all(
            torch.isfinite(tensor).all() for tensor in (shared, entry, projection)
        ):
            raise ValueError("factorized codebooks must be finite")
        cardinalities.append(cardinality)
    assert heads is not None and dimension is not None
    return heads, dimension, tuple(cardinalities)


def materialize_factorized_codebooks(
    shared_codebooks: Sequence[torch.Tensor],
    entry_factors: Sequence[torch.Tensor],
    dimension_factors: Sequence[torch.Tensor],
    stage_bits: Sequence[int],
) -> list[torch.Tensor]:
    """Materialize ragged per-head tables as ``[H,K_s,D]`` tensors."""

    _validate_factorized_stages(
        shared_codebooks, entry_factors, dimension_factors, stage_bits
    )
    result = []
    for shared, entry, projection in zip(
        shared_codebooks, entry_factors, dimension_factors
    ):
        if shared.ndim == 3:
            result.append(shared)
        else:
            correction = torch.einsum("hkr,hrd->hkd", entry, projection)
            result.append(shared.unsqueeze(0) + correction)
    return result


def factorize_head_codebooks(
    codebooks: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Low-rank factorize head-specific ``[H,K,D]`` tables around their mean."""

    if codebooks.ndim != 3 or not torch.isfinite(codebooks).all():
        raise ValueError("codebooks must have finite [H,K,D] layout")
    heads, cardinality, dimension = codebooks.shape
    requested_rank = int(rank)
    if requested_rank < 0:
        raise ValueError("factor rank must be non-negative")
    local_rank = min(requested_rank, cardinality, dimension)
    shared = codebooks.mean(dim=0)
    entry = torch.empty(
        (heads, cardinality, local_rank),
        dtype=codebooks.dtype,
        device=codebooks.device,
    )
    projection = torch.empty(
        (heads, local_rank, dimension),
        dtype=codebooks.dtype,
        device=codebooks.device,
    )
    for head in range(heads):
        if local_rank == 0:
            continue
        left, singular, right = torch.linalg.svd(
            codebooks[head] - shared, full_matrices=False
        )
        entry[head] = left[:, :local_rank] * singular[:local_rank]
        projection[head] = right[:local_rank]
    return shared, entry, projection


def _nearest_indices(
    states: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    assignment_whiteners: torch.Tensor | None = None,
    token_chunk_size: int = 4096,
) -> torch.Tensor:
    """Return nearest table entry for ``states=[H,N,D]`` and tables ``[H,K,D]``."""

    if states.ndim != 3 or codebooks.ndim != 3:
        raise ValueError("nearest-neighbor inputs must have [H,N,D]/[H,K,D]")
    heads, tokens, dimension = states.shape
    if codebooks.shape[0] != heads or codebooks.shape[2] != dimension:
        raise ValueError("nearest-neighbor geometry differs")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    if assignment_whiteners is not None:
        if assignment_whiteners.shape != (heads, dimension, dimension):
            raise ValueError("assignment whiteners must have [H,D,D] layout")
        selector_states = torch.matmul(states, assignment_whiteners.transpose(-1, -2))
        selector_tables = torch.matmul(
            codebooks, assignment_whiteners.transpose(-1, -2)
        )
    else:
        selector_states = states
        selector_tables = codebooks
    table_norm = selector_tables.square().sum(dim=-1)
    indices = torch.empty((heads, tokens), dtype=torch.long, device=states.device)
    for start in range(0, tokens, token_chunk_size):
        stop = min(tokens, start + token_chunk_size)
        chunk = selector_states[:, start:stop]
        distances = (
            chunk.square().sum(dim=-1, keepdim=True)
            + table_norm[:, None, :]
            - 2.0 * torch.bmm(chunk, selector_tables.transpose(-1, -2))
        )
        indices[:, start:stop] = distances.argmin(dim=-1)
    return indices


def _nearest_topk_indices(
    states: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    candidates: int,
    excluded_indices: torch.Tensor | None = None,
    token_chunk_size: int = 4096,
) -> torch.Tensor:
    """Return the nearest ``candidates`` entries, optionally excluding one per row."""

    if states.ndim != 3 or codebooks.ndim != 3:
        raise ValueError("nearest-neighbor inputs must have [H,N,D]/[H,K,D]")
    heads, tokens, dimension = states.shape
    if codebooks.shape[0] != heads or codebooks.shape[2] != dimension:
        raise ValueError("nearest-neighbor geometry differs")
    if candidates <= 0 or candidates >= codebooks.shape[1]:
        raise ValueError("top-k candidate count must lie in [1, cardinality)")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    if excluded_indices is not None and excluded_indices.shape != (heads, tokens):
        raise ValueError("excluded indices must have [H,N] layout")
    table_norm = codebooks.square().sum(dim=-1)
    result = torch.empty(
        (heads, tokens, candidates), dtype=torch.long, device=states.device
    )
    for start in range(0, tokens, token_chunk_size):
        stop = min(tokens, start + token_chunk_size)
        chunk = states[:, start:stop]
        distances = (
            chunk.square().sum(dim=-1, keepdim=True)
            + table_norm[:, None, :]
            - 2.0 * torch.matmul(chunk, codebooks.transpose(-1, -2))
        )
        if excluded_indices is not None:
            distances.scatter_(
                -1,
                excluded_indices[:, start:stop].long().unsqueeze(-1),
                torch.inf,
            )
        result[:, start:stop] = distances.topk(
            candidates, dim=-1, largest=False, sorted=True
        ).indices
    return result


@torch.no_grad()
def initialize_factorized_residual_vq(
    states: torch.Tensor,
    *,
    stage_bits: Sequence[int],
    factor_ranks: Sequence[int],
    kmeans_iterations: int = 4,
    seed: int = 0,
    assignment_whiteners: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    token_chunk_size: int = 4096,
    representation: str = "factorized",
) -> dict[str, object]:
    """Initialize nested VQ with per-head residual k-means, then factorize.

    This is a non-GSRQ initializer: centroids are learned directly from the
    supplied calibration states and no parent codebook is read.  Every stage is
    factorized before its reconstruction is subtracted, so later residuals see
    the actual deployable shared/factorized decoder rather than an oracle table.
    """

    schedule = validate_stage_bits(stage_bits)
    ranks = tuple(int(rank) for rank in factor_ranks)
    if len(ranks) != len(schedule) or any(rank < 0 for rank in ranks):
        raise ValueError("one non-negative factor rank is required per stage")
    if (
        states.ndim != 4
        or any(int(size) <= 0 for size in states.shape)
        or not torch.isfinite(states).all()
    ):
        raise ValueError("states must have finite [B,H,T,D] layout")
    if kmeans_iterations <= 0:
        raise ValueError("kmeans iterations must be positive")
    if representation not in {"factorized", "per_head"}:
        raise ValueError("representation must be factorized or per_head")
    batch, heads, tokens, dimension = states.shape
    residual = (
        states.permute(1, 0, 2, 3)
        .reshape(heads, batch * tokens, dimension)
        .float()
        .clone()
    )
    flattened_weights = None
    if sample_weights is not None:
        if sample_weights.shape != (batch, heads, tokens):
            raise ValueError("sample weights must have [B,H,T] layout")
        if not torch.isfinite(sample_weights).all() or (sample_weights < 0).any():
            raise ValueError("sample weights must be finite and non-negative")
        flattened_weights = (
            sample_weights.permute(1, 0, 2)
            .reshape(heads, batch * tokens)
            .to(device=states.device, dtype=torch.float32)
        )
        if (flattened_weights.gt(0).sum(dim=-1) < max(1 << bits for bits in schedule)).any():
            raise ValueError(
                "each head must have at least the maximum stage cardinality positive weights"
            )
    if assignment_whiteners is not None:
        assignment_whiteners = assignment_whiteners.to(
            device=states.device, dtype=torch.float32
        )
    generator = torch.Generator(device=states.device).manual_seed(int(seed))
    shared_stages: list[torch.Tensor] = []
    entry_stages: list[torch.Tensor] = []
    dimension_stages: list[torch.Tensor] = []
    residual_mse = []
    weighted_residual_mse = []
    for stage, (bits, rank) in enumerate(zip(schedule, ranks)):
        cardinality = 1 << bits
        if cardinality > residual.shape[1]:
            raise ValueError("stage cardinality exceeds calibration vectors per head")
        samples = []
        for head in range(heads):
            if flattened_weights is None:
                chosen = torch.randperm(
                    residual.shape[1], generator=generator, device=states.device
                )[:cardinality]
            else:
                chosen = torch.multinomial(
                    flattened_weights[head],
                    cardinality,
                    replacement=False,
                    generator=generator,
                )
            samples.append(residual[head].index_select(0, chosen))
        centroids = torch.stack(samples)
        for _ in range(kmeans_iterations):
            assignments = _nearest_indices(
                residual,
                centroids,
                assignment_whiteners=assignment_whiteners,
                token_chunk_size=token_chunk_size,
            )
            updated = centroids.clone()
            for head in range(heads):
                sums = torch.zeros_like(centroids[head])
                if flattened_weights is None:
                    weighted_residual = residual[head]
                    counts = torch.bincount(
                        assignments[head], minlength=cardinality
                    ).to(residual.dtype)
                else:
                    local_weights = flattened_weights[head]
                    weighted_residual = residual[head] * local_weights[:, None]
                    counts = torch.zeros(
                        cardinality, dtype=residual.dtype, device=residual.device
                    )
                    counts.index_add_(0, assignments[head], local_weights)
                sums.index_add_(0, assignments[head], weighted_residual)
                occupied = counts > 0
                updated[head, occupied] = sums[occupied] / counts[occupied, None]
            centroids = updated
        if representation == "factorized":
            shared, entry, projection = factorize_head_codebooks(centroids, rank)
            materialized = shared.unsqueeze(0) + torch.einsum(
                "hkr,hrd->hkd", entry, projection
            )
        else:
            shared = centroids
            entry = centroids.new_empty((heads, cardinality, 0))
            projection = centroids.new_empty((heads, 0, dimension))
            materialized = shared
        assignments = _nearest_indices(
            residual,
            materialized,
            assignment_whiteners=assignment_whiteners,
            token_chunk_size=token_chunk_size,
        )
        selected = torch.stack(
            [materialized[head].index_select(0, assignments[head]) for head in range(heads)]
        )
        residual.sub_(selected)
        shared_stages.append(shared)
        entry_stages.append(entry)
        dimension_stages.append(projection)
        residual_mse.append(float(residual.square().mean().item()))
        if flattened_weights is None:
            weighted_residual_mse.append(residual_mse[-1])
        else:
            denominator = flattened_weights.sum() * dimension
            weighted_residual_mse.append(
                float(
                    (residual.square() * flattened_weights[..., None])
                    .sum()
                    .div(denominator)
                    .item()
                )
            )
    return {
        "shared_codebooks": shared_stages,
        "entry_factors": entry_stages,
        "dimension_factors": dimension_stages,
        "residual_mse_by_stage": residual_mse,
        "weighted_residual_mse_by_stage": weighted_residual_mse,
        "representation": representation,
    }


@torch.no_grad()
def factorized_residual_vq_reconstructions(
    states: torch.Tensor,
    shared_codebooks: Sequence[torch.Tensor],
    entry_factors: Sequence[torch.Tensor],
    dimension_factors: Sequence[torch.Tensor],
    *,
    stage_bits: Sequence[int],
    prefixes: Iterable[int],
    token_chunk_size: int = 4096,
    return_indices: bool = False,
    assignment_whiteners: torch.Tensor | None = None,
    prevalidated_direct_tables: bool = False,
) -> tuple[dict[int, torch.Tensor], torch.Tensor | None]:
    """Encode once and reconstruct unequal-cardinality nested prefixes."""

    if prevalidated_direct_tables:
        schedule = validate_stage_bits(stage_bits)
        if not shared_codebooks or not (
            len(shared_codebooks)
            == len(entry_factors)
            == len(dimension_factors)
            == len(schedule)
        ):
            raise ValueError("prevalidated direct stage lists differ")
        first = shared_codebooks[0]
        if first.ndim != 3:
            raise ValueError("prevalidated tables must be direct [H,K,D] tensors")
        heads, dimension = int(first.shape[0]), int(first.shape[2])
        if any(entry.numel() or projection.numel() for entry, projection in zip(entry_factors, dimension_factors)):
            raise ValueError("prevalidated direct tables cannot have factors")
    else:
        heads, dimension, _ = _validate_factorized_stages(
            shared_codebooks, entry_factors, dimension_factors, stage_bits
        )
    if states.ndim != 4 or states.shape[1] != heads or states.shape[-1] != dimension:
        raise ValueError("states and factorized codec geometry differs")
    requested = sorted({int(prefix) for prefix in prefixes})
    if not requested or requested[0] <= 0 or requested[-1] > len(stage_bits):
        raise ValueError("prefixes must lie inside the stage schedule")
    batch, _heads, tokens, _dimension = states.shape
    maximum = requested[-1]
    tables = (
        list(shared_codebooks[:maximum])
        if prevalidated_direct_tables
        else materialize_factorized_codebooks(
            shared_codebooks[:maximum],
            entry_factors[:maximum],
            dimension_factors[:maximum],
            stage_bits[:maximum],
        )
    )
    outputs = {prefix: torch.empty_like(states) for prefix in requested}
    saved_indices = (
        torch.empty(
            (batch, heads, tokens, maximum), dtype=torch.uint8, device=states.device
        )
        if return_indices
        else None
    )
    residual = (
        states.permute(1, 0, 2, 3)
        .reshape(heads, batch * tokens, dimension)
        .float()
        .clone()
    )
    decoded = torch.zeros_like(residual)
    local_whiteners = (
        None
        if assignment_whiteners is None
        else assignment_whiteners.to(device=states.device, dtype=torch.float32)
    )
    # Preserve the deployed SCFW/GSRQ arithmetic path exactly.  Reapplying the
    # linear transform to the raw residual at every stage is algebraically
    # equivalent, but its different FP32 reduction order changes a small
    # number of nearest-neighbour ties.  The reference encoder transforms once
    # and then recursively subtracts transformed codewords.  Keep raw and
    # selector residuals in lockstep so assignments are identity-preserving
    # while reconstruction remains in the original coordinate system.
    selector_residual = None
    selector_tables: torch.Tensor | None = None
    if local_whiteners is not None:
        # Match the legacy SCFW/GSRQ encoder literally, including the einsum
        # layout and one full-window bmm.  Both layout and matrix size can
        # affect FP32 GEMM rounding at near-tied codewords.
        selector_residual = (
            torch.einsum("hde,bhte->bhtd", local_whiteners, states.float())
            .permute(1, 0, 2, 3)
            .reshape(heads, batch * tokens, dimension)
            .float()
            .clone()
        )
        raw_table_tensor = torch.stack(
            [table.to(device=states.device, dtype=torch.float32) for table in tables],
            dim=1,
        )
        selector_tables = torch.einsum(
            "hrke,hde->hrkd", raw_table_tensor, local_whiteners
        )
    for stage, table in enumerate(tables):
        local_table = table.to(device=states.device, dtype=torch.float32)
        selector_table = (
            local_table
            if selector_tables is None
            else selector_tables[:, stage]
        )
        indices = _nearest_indices(
            residual if selector_residual is None else selector_residual,
            selector_table,
            assignment_whiteners=None,
            token_chunk_size=(
                token_chunk_size
                if selector_residual is None
                else selector_residual.shape[1]
            ),
        )
        selected = torch.gather(
            local_table,
            1,
            indices[..., None].expand(-1, -1, dimension),
        )
        decoded.add_(selected)
        residual.sub_(selected)
        if selector_residual is not None:
            selector_selected = torch.gather(
                selector_table,
                1,
                indices[..., None].expand(-1, -1, dimension),
            )
            selector_residual.sub_(selector_selected)
        if saved_indices is not None:
            saved_indices[..., stage] = indices.reshape(heads, batch, tokens).permute(
                1, 0, 2
            ).to(torch.uint8)
        prefix = stage + 1
        if prefix in outputs:
            reconstructed = decoded.reshape(heads, batch, tokens, dimension).permute(
                1, 0, 2, 3
            )
            if local_whiteners is not None:
                # Legacy transformed-K decoding accumulates directly into a
                # contiguous [B,H,T,D] buffer before the dtype conversion.
                reconstructed = reconstructed.contiguous()
            outputs[prefix] = reconstructed.to(states.dtype).clone()
    return outputs, saved_indices


def _selected_head_codewords(
    table: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Gather ``[H,N]`` indices from one ``[H,K,D]`` stage table."""

    if table.ndim != 3 or indices.ndim != 2 or table.shape[0] != indices.shape[0]:
        raise ValueError("per-head codeword gather geometry differs")
    return torch.gather(
        table,
        1,
        indices.to(dtype=torch.long).unsqueeze(-1).expand(-1, -1, table.shape[-1]),
    )


@torch.no_grad()
def multi_exit_additive_coordinate_refinement(
    states: torch.Tensor,
    stage_tables: Sequence[torch.Tensor],
    initial_indices: torch.Tensor,
    *,
    prefixes: Iterable[int],
    exit_weights: Sequence[float],
    sweeps: int = 1,
    token_chunk_size: int = 4096,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, dict[str, object]]:
    """Refine additive indices against all deployable exits jointly.

    ``initial_indices`` normally comes from the ordinary greedy residual
    encoder.  A reverse coordinate sweep then updates one stage at a time.
    For stage ``s`` and fixed remaining indices, the exact Euclidean target is

    ``old_word + sum_m w_m * (state - reconstruction_m) / sum_m w_m``

    over exits ``m`` that include ``s``.  Consequently every coordinate step
    is non-increasing for the declared weighted multi-exit objective.  The
    index layout and decoder are unchanged; only online encoding work differs.
    """

    requested = tuple(sorted({int(prefix) for prefix in prefixes}))
    weights = tuple(float(weight) for weight in exit_weights)
    if states.ndim != 4 or not torch.isfinite(states).all():
        raise ValueError("states must have finite [B,H,T,D] layout")
    if not requested or requested[0] <= 0 or requested[-1] > len(stage_tables):
        raise ValueError("prefixes must lie inside the materialized stages")
    if len(weights) != len(requested) or any(
        not math.isfinite(weight) or weight < 0 for weight in weights
    ):
        raise ValueError("one finite non-negative weight is required per exit")
    if not any(weight > 0 for weight in weights):
        raise ValueError("at least one exit weight must be positive")
    if sweeps <= 0 or token_chunk_size <= 0:
        raise ValueError("coordinate sweeps and token chunks must be positive")
    batch, heads, tokens, dimension = states.shape
    maximum = requested[-1]
    if initial_indices.shape != (batch, heads, tokens, maximum):
        raise ValueError("initial indices differ from the requested stage geometry")
    local_tables = [
        table.to(device=states.device, dtype=torch.float32)
        for table in stage_tables[:maximum]
    ]
    for stage, table in enumerate(local_tables):
        if table.ndim != 3 or table.shape[0] != heads or table.shape[2] != dimension:
            raise ValueError(f"stage {stage} table geometry differs from states")
        if not torch.isfinite(table).all():
            raise ValueError(f"stage {stage} table is non-finite")
    flat_states = (
        states.permute(1, 0, 2, 3)
        .reshape(heads, batch * tokens, dimension)
        .float()
    )
    local_indices = (
        initial_indices.permute(1, 0, 2, 3)
        .reshape(heads, batch * tokens, maximum)
        .long()
        .clone()
    )
    cumulative = torch.zeros_like(flat_states)
    reconstructions: dict[int, torch.Tensor] = {}
    requested_set = set(requested)
    for stage, table in enumerate(local_tables):
        stage_indices = local_indices[..., stage]
        if int(stage_indices.min()) < 0 or int(stage_indices.max()) >= table.shape[1]:
            raise ValueError(f"stage {stage} index is outside its codebook")
        cumulative.add_(_selected_head_codewords(table, stage_indices))
        prefix = stage + 1
        if prefix in requested_set:
            reconstructions[prefix] = cumulative.clone()

    def objective() -> tuple[float, dict[str, float]]:
        per_exit = {
            str(prefix): float(
                (flat_states - reconstructions[prefix]).square().mean().item()
            )
            for prefix in requested
        }
        denominator = sum(weights)
        weighted = sum(
            weight * per_exit[str(prefix)]
            for prefix, weight in zip(requested, weights)
        ) / denominator
        return weighted, per_exit

    initial_objective, initial_per_exit = objective()
    sweep_history: list[dict[str, object]] = []
    for sweep in range(sweeps):
        changed_by_stage: dict[str, float] = {}
        for stage in range(maximum - 1, -1, -1):
            affected = [
                (prefix, weight)
                for prefix, weight in zip(requested, weights)
                if prefix > stage and weight > 0
            ]
            weight_sum = sum(weight for _prefix, weight in affected)
            if weight_sum <= 0:
                raise ValueError(f"stage {stage} is not covered by a positive exit")
            table = local_tables[stage]
            old_indices = local_indices[..., stage]
            old_words = _selected_head_codewords(table, old_indices)
            target = old_words.clone()
            residual = torch.zeros_like(target)
            for prefix, weight in affected:
                residual.add_(flat_states - reconstructions[prefix], alpha=weight)
            target.add_(residual, alpha=1.0 / weight_sum)
            new_indices = _nearest_indices(
                target, table, token_chunk_size=token_chunk_size
            )
            changed_by_stage[str(stage + 1)] = float(
                new_indices.ne(old_indices).float().mean().item()
            )
            new_words = _selected_head_codewords(table, new_indices)
            delta = new_words - old_words
            for prefix, _weight in affected:
                reconstructions[prefix].add_(delta)
            local_indices[..., stage] = new_indices
        local_objective, local_per_exit = objective()
        sweep_history.append(
            {
                "sweep": sweep + 1,
                "weighted_objective": local_objective,
                "per_exit_mse": local_per_exit,
                "changed_fraction_by_stage": changed_by_stage,
                "mean_changed_fraction": sum(changed_by_stage.values())
                / len(changed_by_stage),
            }
        )
    final_objective, final_per_exit = objective()
    outputs = {
        prefix: reconstruction.reshape(heads, batch, tokens, dimension)
        .permute(1, 0, 2, 3)
        .to(dtype=states.dtype)
        .contiguous()
        for prefix, reconstruction in reconstructions.items()
    }
    refined_indices = (
        local_indices.reshape(heads, batch, tokens, maximum)
        .permute(1, 0, 2, 3)
        .to(torch.uint8)
        .contiguous()
    )
    diagnostics: dict[str, object] = {
        "mode": "multi_exit_additive_coordinate",
        "coordinate_order": "reverse",
        "sweeps": sweeps,
        "prefixes": list(requested),
        "exit_weights": list(weights),
        "initial_weighted_objective": initial_objective,
        "final_weighted_objective": final_objective,
        "relative_objective_change": final_objective / max(initial_objective, 1e-30)
        - 1.0,
        "initial_per_exit_mse": initial_per_exit,
        "final_per_exit_mse": final_per_exit,
        "sweep_history": sweep_history,
    }
    return outputs, refined_indices, diagnostics


def _candidate_head_codewords(
    table: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Gather ``[H,N,L]`` candidate indices as ``[H,N,L,D]`` words."""

    if table.ndim != 3 or indices.ndim != 3 or table.shape[0] != indices.shape[0]:
        raise ValueError("per-head candidate gather geometry differs")
    return torch.stack(
        [F.embedding(indices[head].long(), table[head]) for head in range(table.shape[0])]
    )


@torch.no_grad()
def multi_exit_additive_block_refinement(
    states: torch.Tensor,
    stage_tables: Sequence[torch.Tensor],
    initial_indices: torch.Tensor,
    *,
    prefixes: Iterable[int],
    exit_weights: Sequence[float],
    sweeps: int = 1,
    shortlist_size: int = 4,
    token_chunk_size: int = 512,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, dict[str, object]]:
    """Sliding two-stage block-coordinate refinement for progressive AQ.

    Each reverse sweep visits overlapping pairs ``(S-1,S), ..., (1,2)``.
    The old index of each stage is always candidate zero; the remaining
    candidates are its nearest coordinate alternatives.  The Cartesian
    product is scored with the exact weighted error of every affected exit,
    so each block update is fail-closed and can change two indices together.
    """

    requested = tuple(sorted({int(prefix) for prefix in prefixes}))
    weights = tuple(float(weight) for weight in exit_weights)
    if states.ndim != 4 or not torch.isfinite(states).all():
        raise ValueError("states must have finite [B,H,T,D] layout")
    if not requested or requested[0] <= 0 or requested[-1] > len(stage_tables):
        raise ValueError("prefixes must lie inside the materialized stages")
    if len(weights) != len(requested) or any(
        not math.isfinite(weight) or weight < 0 for weight in weights
    ):
        raise ValueError("one finite non-negative weight is required per exit")
    if not any(weight > 0 for weight in weights):
        raise ValueError("at least one exit weight must be positive")
    if sweeps <= 0 or shortlist_size < 2 or token_chunk_size <= 0:
        raise ValueError("invalid block-refinement search geometry")
    batch, heads, tokens, dimension = states.shape
    maximum = requested[-1]
    if maximum < 2:
        raise ValueError("two-stage refinement requires at least two stages")
    if initial_indices.shape != (batch, heads, tokens, maximum):
        raise ValueError("initial indices differ from the requested stage geometry")
    local_tables = [
        table.to(device=states.device, dtype=torch.float32)
        for table in stage_tables[:maximum]
    ]
    for stage, table in enumerate(local_tables):
        if table.ndim != 3 or table.shape[0] != heads or table.shape[2] != dimension:
            raise ValueError(f"stage {stage} table geometry differs from states")
        if table.shape[1] < shortlist_size or not torch.isfinite(table).all():
            raise ValueError(f"stage {stage} cannot support the requested shortlist")
    flat_states = (
        states.permute(1, 0, 2, 3)
        .reshape(heads, batch * tokens, dimension)
        .float()
    )
    local_indices = (
        initial_indices.permute(1, 0, 2, 3)
        .reshape(heads, batch * tokens, maximum)
        .long()
        .clone()
    )
    cumulative = torch.zeros_like(flat_states)
    reconstructions: dict[int, torch.Tensor] = {}
    requested_set = set(requested)
    for stage, table in enumerate(local_tables):
        stage_indices = local_indices[..., stage]
        if int(stage_indices.min()) < 0 or int(stage_indices.max()) >= table.shape[1]:
            raise ValueError(f"stage {stage} index is outside its codebook")
        cumulative.add_(_selected_head_codewords(table, stage_indices))
        if stage + 1 in requested_set:
            reconstructions[stage + 1] = cumulative.clone()

    def objective() -> tuple[float, dict[str, float]]:
        per_exit = {
            str(prefix): float(
                (flat_states - reconstructions[prefix]).square().mean().item()
            )
            for prefix in requested
        }
        weighted = sum(
            weight * per_exit[str(prefix)]
            for prefix, weight in zip(requested, weights)
        ) / sum(weights)
        return weighted, per_exit

    initial_objective, initial_per_exit = objective()
    sweep_history: list[dict[str, object]] = []
    total_tokens = flat_states.shape[1]
    for sweep in range(sweeps):
        block_history: list[dict[str, object]] = []
        for first_stage in range(maximum - 2, -1, -1):
            second_stage = first_stage + 1
            block_stages = (first_stage, second_stage)
            affected = [
                (prefix, weight)
                for prefix, weight in zip(requested, weights)
                if prefix > first_stage and weight > 0
            ]
            changed_first = changed_second = 0
            for start in range(0, total_tokens, token_chunk_size):
                stop = min(total_tokens, start + token_chunk_size)
                state_chunk = flat_states[:, start:stop]
                old_indices = [
                    local_indices[:, start:stop, stage] for stage in block_stages
                ]
                old_words = [
                    _selected_head_codewords(local_tables[stage], index)
                    for stage, index in zip(block_stages, old_indices)
                ]
                shortlists = []
                candidate_words = []
                for stage, old_index, old_word in zip(
                    block_stages, old_indices, old_words
                ):
                    local_affected = [
                        (prefix, weight)
                        for prefix, weight in affected
                        if prefix > stage
                    ]
                    weight_sum = sum(weight for _prefix, weight in local_affected)
                    if weight_sum <= 0:
                        raise ValueError(f"stage {stage} lacks a positive exit")
                    residual = torch.zeros_like(old_word)
                    for prefix, weight in local_affected:
                        residual.add_(
                            state_chunk - reconstructions[prefix][:, start:stop],
                            alpha=weight,
                        )
                    target = old_word + residual / weight_sum
                    alternatives = _nearest_topk_indices(
                        target,
                        local_tables[stage],
                        candidates=shortlist_size - 1,
                        excluded_indices=old_index,
                        token_chunk_size=token_chunk_size,
                    )
                    shortlist = torch.cat((old_index.unsqueeze(-1), alternatives), dim=-1)
                    shortlists.append(shortlist)
                    candidate_words.append(
                        _candidate_head_codewords(local_tables[stage], shortlist)
                    )
                local_cost = torch.zeros(
                    (
                        heads,
                        stop - start,
                        shortlist_size,
                        shortlist_size,
                    ),
                    dtype=torch.float32,
                    device=states.device,
                )
                for prefix, weight in affected:
                    base = state_chunk - reconstructions[prefix][:, start:stop]
                    additions = []
                    for stage, old_word, candidates_words in zip(
                        block_stages, old_words, candidate_words
                    ):
                        if prefix > stage:
                            base = base + old_word
                            additions.append(candidates_words)
                        else:
                            additions.append(None)
                    if additions[1] is None:
                        candidate_sum = additions[0].unsqueeze(-2)
                    else:
                        candidate_sum = additions[0].unsqueeze(-2) + additions[1].unsqueeze(-3)
                    local_cost.add_(
                        (base.unsqueeze(-2).unsqueeze(-2) - candidate_sum)
                        .square()
                        .sum(dim=-1),
                        alpha=weight,
                    )
                choice = local_cost.flatten(-2).argmin(dim=-1)
                first_choice = torch.div(
                    choice, shortlist_size, rounding_mode="floor"
                )
                second_choice = choice.remainder(shortlist_size)
                new_indices = [
                    torch.gather(shortlists[0], -1, first_choice.unsqueeze(-1)).squeeze(-1),
                    torch.gather(shortlists[1], -1, second_choice.unsqueeze(-1)).squeeze(-1),
                ]
                changed_first += int(new_indices[0].ne(old_indices[0]).sum())
                changed_second += int(new_indices[1].ne(old_indices[1]).sum())
                new_words = [
                    _selected_head_codewords(local_tables[stage], index)
                    for stage, index in zip(block_stages, new_indices)
                ]
                deltas = [new - old for new, old in zip(new_words, old_words)]
                for prefix, _weight in affected:
                    for stage, delta in zip(block_stages, deltas):
                        if prefix > stage:
                            reconstructions[prefix][:, start:stop].add_(delta)
                for stage, index in zip(block_stages, new_indices):
                    local_indices[:, start:stop, stage] = index
            block_history.append(
                {
                    "stages": [first_stage + 1, second_stage + 1],
                    "changed_fraction": [
                        changed_first / (heads * total_tokens),
                        changed_second / (heads * total_tokens),
                    ],
                }
            )
        local_objective, local_per_exit = objective()
        sweep_history.append(
            {
                "sweep": sweep + 1,
                "weighted_objective": local_objective,
                "per_exit_mse": local_per_exit,
                "blocks": block_history,
                "mean_block_changed_fraction": sum(
                    sum(block["changed_fraction"]) / 2 for block in block_history
                )
                / len(block_history),
            }
        )
    final_objective, final_per_exit = objective()
    outputs = {
        prefix: reconstruction.reshape(heads, batch, tokens, dimension)
        .permute(1, 0, 2, 3)
        .to(dtype=states.dtype)
        .contiguous()
        for prefix, reconstruction in reconstructions.items()
    }
    refined_indices = (
        local_indices.reshape(heads, batch, tokens, maximum)
        .permute(1, 0, 2, 3)
        .to(torch.uint8)
        .contiguous()
    )
    diagnostics: dict[str, object] = {
        "mode": "multi_exit_additive_block",
        "block_schedule": "reverse_sliding_pairs",
        "sweeps": sweeps,
        "shortlist_size": shortlist_size,
        "prefixes": list(requested),
        "exit_weights": list(weights),
        "initial_weighted_objective": initial_objective,
        "final_weighted_objective": final_objective,
        "relative_objective_change": final_objective / max(initial_objective, 1e-30)
        - 1.0,
        "initial_per_exit_mse": initial_per_exit,
        "final_per_exit_mse": final_per_exit,
        "sweep_history": sweep_history,
    }
    return outputs, refined_indices, diagnostics


@torch.no_grad()
def causal_error_feedback_vq_reconstructions(
    states: torch.Tensor,
    shared_codebooks: Sequence[torch.Tensor],
    entry_factors: Sequence[torch.Tensor],
    dimension_factors: Sequence[torch.Tensor],
    *,
    stage_bits: Sequence[int],
    prefixes: Iterable[int],
    encoder: Mapping[str, object],
    value_exit_biases: Mapping[int | str, torch.Tensor] | None = None,
    token_chunk_size: int = 4096,
    return_indices: bool = False,
) -> tuple[dict[int, torch.Tensor], torch.Tensor | None, dict[str, object]]:
    """Causal first-order error feedback around a positive-depth VQ encoder.

    Tokens are split into fixed reset groups.  Within every group, token ``t``
    is encoded from ``state_t + gain * feedback_{t-1}``, and the feedback state
    is the selected exit's corrected-input reconstruction error.  Groups are
    processed in parallel, but no assignment observes a later token in its
    group.  The decoder and stored indices are unchanged; an incremental
    deployment retains one small ``[H,D]`` encoder state per layer/request.
    """

    requested = tuple(sorted({int(prefix) for prefix in prefixes}))
    if states.ndim != 4 or not torch.isfinite(states).all():
        raise ValueError("states must have finite [B,H,T,D] layout")
    if not requested or requested[0] <= 0:
        raise ValueError("at least one positive exit is required")
    reset_tokens = int(encoder.get("feedback_reset_tokens", 8))
    gain = float(encoder.get("feedback_gain", 1.0))
    raw_gain_per_head = encoder.get("feedback_gain_per_head")
    feedback_exit = int(encoder.get("feedback_state_exit", requested[len(requested) // 2]))
    state_dtype_name = str(encoder.get("feedback_state_dtype", "float16"))
    bias_in_feedback = bool(encoder.get("affine_bias_in_feedback", False))
    if reset_tokens <= 0 or not math.isfinite(gain) or not 0.0 < gain <= 1.0:
        raise ValueError("error-feedback reset/gain geometry is invalid")
    if feedback_exit not in requested:
        raise ValueError("feedback state exit must be one of the requested exits")
    if state_dtype_name not in {"float16", "float32"}:
        raise ValueError("feedback state dtype must be float16 or float32")
    state_dtype = torch.float16 if state_dtype_name == "float16" else torch.float32
    raw_inner = encoder.get("inner_encoder")
    if not isinstance(raw_inner, Mapping):
        raise ValueError("causal error feedback requires an inner encoder mapping")
    inner = dict(raw_inner)
    inner_mode = str(inner.get("mode", "greedy_residual"))
    if inner_mode not in {"greedy_residual", "multi_exit_additive_block"}:
        raise ValueError("error-feedback inner encoder must be greedy or additive block")

    inner_weights: list[float] | None = None
    tables: list[torch.Tensor] | None = None
    if inner_mode == "multi_exit_additive_block":
        raw_weights = inner.get("exit_weights")
        if not isinstance(raw_weights, Mapping):
            raise ValueError("additive block inner encoder requires exit weights")
        inner_weights = []
        for prefix in requested:
            if str(prefix) in raw_weights:
                inner_weights.append(float(raw_weights[str(prefix)]))
            elif prefix in raw_weights:
                inner_weights.append(float(raw_weights[prefix]))
            else:
                raise ValueError(f"inner encoder lacks exit {prefix} weight")
        tables = materialize_factorized_codebooks(
            shared_codebooks, entry_factors, dimension_factors, stage_bits
        )

    batch, heads, tokens, dimension = states.shape
    gain_per_head: torch.Tensor | None = None
    if raw_gain_per_head is not None:
        gain_per_head = torch.as_tensor(
            raw_gain_per_head, device=states.device, dtype=torch.float32
        )
        if gain_per_head.shape != (heads,) or not torch.isfinite(gain_per_head).all():
            raise ValueError("per-head feedback gain must have finite [H] layout")
        if bool((gain_per_head <= 0).any()) or bool((gain_per_head > 1).any()):
            raise ValueError("per-head feedback gains must lie in (0,1]")
    feedback_bias: torch.Tensor | None = None
    if bias_in_feedback:
        if value_exit_biases is None:
            raise ValueError("bias-aware feedback requires value exit biases")
        raw_bias = value_exit_biases.get(
            feedback_exit, value_exit_biases.get(str(feedback_exit))
        )
        if raw_bias is None:
            raise ValueError("feedback-state exit has no affine value bias")
        feedback_bias = raw_bias.to(device=states.device, dtype=torch.float32)
        if feedback_bias.shape != (heads, dimension) or not torch.isfinite(
            feedback_bias
        ).all():
            raise ValueError("feedback affine bias must have finite [H,D] layout")
    maximum = requested[-1]
    groups = (tokens + reset_tokens - 1) // reset_tokens
    feedback = torch.zeros(
        (batch, heads, groups, dimension),
        dtype=state_dtype,
        device=states.device,
    )
    outputs = {prefix: torch.empty_like(states) for prefix in requested}
    saved_indices = (
        torch.empty(
            (batch, heads, tokens, maximum),
            dtype=torch.uint8,
            device=states.device,
        )
        if return_indices
        else None
    )
    offset_history: list[dict[str, object]] = []
    for offset in range(min(reset_tokens, tokens)):
        current = states[:, :, offset::reset_tokens, :].contiguous()
        active_groups = current.shape[2]
        previous = feedback[:, :, :active_groups].float()
        if gain_per_head is None:
            corrected = current.float() + gain * previous
        else:
            corrected = (
                current.float()
                + gain_per_head[None, :, None, None] * previous
            )
        local_outputs, local_indices = factorized_residual_vq_reconstructions(
            corrected,
            shared_codebooks,
            entry_factors,
            dimension_factors,
            stage_bits=stage_bits,
            prefixes=requested,
            token_chunk_size=token_chunk_size,
            return_indices=True,
        )
        inner_diagnostics: dict[str, object] = {"mode": inner_mode}
        if local_indices is None:
            raise RuntimeError("error-feedback inner encoder returned no indices")
        if inner_mode == "multi_exit_additive_block":
            assert tables is not None and inner_weights is not None
            local_outputs, local_indices, inner_diagnostics = (
                multi_exit_additive_block_refinement(
                    corrected,
                    tables,
                    local_indices,
                    prefixes=requested,
                    exit_weights=inner_weights,
                    sweeps=int(inner.get("coordinate_sweeps", 1)),
                    shortlist_size=int(inner.get("block_shortlist_size", 4)),
                    token_chunk_size=int(inner.get("block_token_chunk_size", 512)),
                )
            )
        for prefix in requested:
            outputs[prefix][:, :, offset::reset_tokens, :].copy_(
                local_outputs[prefix].to(dtype=states.dtype)
            )
        if saved_indices is not None:
            saved_indices[:, :, offset::reset_tokens, :].copy_(local_indices)
        decoded_feedback_exit = local_outputs[feedback_exit].float()
        if feedback_bias is not None:
            decoded_feedback_exit = (
                decoded_feedback_exit + feedback_bias[None, :, None, :]
            )
        next_feedback = corrected - decoded_feedback_exit
        feedback[:, :, :active_groups].copy_(next_feedback.to(dtype=state_dtype))
        offset_history.append(
            {
                "offset": offset,
                "active_groups": active_groups,
                "feedback_rms": float(next_feedback.square().mean().sqrt().item()),
                "feedback_max_norm": float(next_feedback.norm(dim=-1).max().item()),
                "inner_relative_objective_change": inner_diagnostics.get(
                    "relative_objective_change"
                ),
            }
        )

    per_exit_mse = {
        str(prefix): float((states.float() - output.float()).square().mean().item())
        for prefix, output in outputs.items()
    }
    diagnostics: dict[str, object] = {
        "mode": "causal_error_feedback_block",
        "feedback_reset_tokens": reset_tokens,
        "feedback_gain": gain,
        "feedback_gain_per_head": (
            None if gain_per_head is None else gain_per_head.tolist()
        ),
        "feedback_state_exit": feedback_exit,
        "feedback_state_dtype": state_dtype_name,
        "affine_bias_in_feedback": bias_in_feedback,
        "causal_within_reset_group": True,
        "decoder_changed": False,
        "inner_encoder": inner,
        "per_exit_mse": per_exit_mse,
        "maximum_feedback_rms": max(row["feedback_rms"] for row in offset_history),
        "maximum_feedback_norm": max(
            row["feedback_max_norm"] for row in offset_history
        ),
        "offset_history": offset_history,
    }
    return outputs, saved_indices, diagnostics


@torch.no_grad()
def factorized_configured_vq_reconstructions(
    states: torch.Tensor,
    shared_codebooks: Sequence[torch.Tensor],
    entry_factors: Sequence[torch.Tensor],
    dimension_factors: Sequence[torch.Tensor],
    *,
    stage_bits: Sequence[int],
    prefixes: Iterable[int],
    encoder: Mapping[str, object] | None = None,
    value_exit_biases: Mapping[int | str, torch.Tensor] | None = None,
    token_chunk_size: int = 4096,
    return_indices: bool = False,
) -> tuple[dict[int, torch.Tensor], torch.Tensor | None, dict[str, object]]:
    """Dispatch between the legacy greedy and declared additive encoder."""

    config = {} if encoder is None else dict(encoder)
    mode = str(config.get("mode", "greedy_residual"))
    requested = tuple(sorted({int(prefix) for prefix in prefixes}))
    if mode == "causal_error_feedback_block":
        return causal_error_feedback_vq_reconstructions(
            states,
            shared_codebooks,
            entry_factors,
            dimension_factors,
            stage_bits=stage_bits,
            prefixes=requested,
            encoder=config,
            value_exit_biases=value_exit_biases,
            token_chunk_size=token_chunk_size,
            return_indices=return_indices,
        )
    if mode == "greedy_residual":
        outputs, indices = factorized_residual_vq_reconstructions(
            states,
            shared_codebooks,
            entry_factors,
            dimension_factors,
            stage_bits=stage_bits,
            prefixes=requested,
            token_chunk_size=token_chunk_size,
            return_indices=return_indices,
        )
        return outputs, indices, {"mode": mode}
    if mode not in {
        "multi_exit_additive_coordinate",
        "multi_exit_additive_block",
    }:
        raise ValueError(f"unknown nested encoder mode: {mode}")
    if str(config.get("coordinate_order", "reverse")) != "reverse":
        raise ValueError("only reverse coordinate order is currently supported")
    raw_weights = config.get("exit_weights")
    if not isinstance(raw_weights, Mapping):
        raise ValueError("coordinate encoder requires an exit-weight mapping")
    weights = []
    for prefix in requested:
        if str(prefix) in raw_weights:
            weights.append(float(raw_weights[str(prefix)]))
        elif prefix in raw_weights:
            weights.append(float(raw_weights[prefix]))
        else:
            raise ValueError(f"coordinate encoder lacks exit {prefix} weight")
    greedy_outputs, greedy_indices = factorized_residual_vq_reconstructions(
        states,
        shared_codebooks,
        entry_factors,
        dimension_factors,
        stage_bits=stage_bits,
        prefixes=requested,
        token_chunk_size=token_chunk_size,
        return_indices=True,
    )
    del greedy_outputs
    if greedy_indices is None:
        raise RuntimeError("greedy initializer did not return additive indices")
    tables = materialize_factorized_codebooks(
        shared_codebooks, entry_factors, dimension_factors, stage_bits
    )
    if mode == "multi_exit_additive_coordinate":
        outputs, indices, diagnostics = multi_exit_additive_coordinate_refinement(
            states,
            tables,
            greedy_indices,
            prefixes=requested,
            exit_weights=weights,
            sweeps=int(config.get("coordinate_sweeps", 1)),
            token_chunk_size=token_chunk_size,
        )
    else:
        outputs, indices, diagnostics = multi_exit_additive_block_refinement(
            states,
            tables,
            greedy_indices,
            prefixes=requested,
            exit_weights=weights,
            sweeps=int(config.get("coordinate_sweeps", 1)),
            shortlist_size=int(config.get("block_shortlist_size", 4)),
            token_chunk_size=int(config.get("block_token_chunk_size", 512)),
        )
    return outputs, indices if return_indices else None, diagnostics


def decode_factorized_multi_exit(
    indices: torch.Tensor,
    shared_codebooks: Sequence[torch.Tensor],
    entry_factors: Sequence[torch.Tensor],
    dimension_factors: Sequence[torch.Tensor],
    stage_depths: torch.Tensor,
    *,
    stage_bits: Sequence[int],
    fixed_prefixes: Iterable[int] = (),
    prevalidated_direct_tables: bool = False,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Differentiably decode variable and fixed nested prefixes."""

    if prevalidated_direct_tables:
        schedule = validate_stage_bits(stage_bits)
        if not shared_codebooks or not (
            len(shared_codebooks)
            == len(entry_factors)
            == len(dimension_factors)
            == len(schedule)
        ):
            raise ValueError("prevalidated direct stage lists differ")
        first = shared_codebooks[0]
        if first.ndim != 3:
            raise ValueError("prevalidated tables must be direct [H,K,D] tensors")
        heads, dimension = int(first.shape[0]), int(first.shape[2])
        cardinalities = tuple(1 << bits for bits in schedule)
        if any(entry.numel() or projection.numel() for entry, projection in zip(entry_factors, dimension_factors)):
            raise ValueError("prevalidated direct tables cannot have factors")
    else:
        heads, dimension, cardinalities = _validate_factorized_stages(
            shared_codebooks, entry_factors, dimension_factors, stage_bits
        )
    if indices.ndim != 4 or indices.shape[1] != heads:
        raise ValueError("indices must have [B,H,T,S] layout")
    if stage_depths.shape != indices.shape[:-1]:
        raise ValueError("stage depths must have [B,H,T] layout")
    if indices.shape[-1] > len(stage_bits):
        raise ValueError("indices exceed the codec stage schedule")
    if stage_depths.numel() and (
        int(stage_depths.min()) <= 0 or int(stage_depths.max()) > indices.shape[-1]
    ):
        raise ValueError("stage depth is outside the encoded stream")
    requested = sorted({int(prefix) for prefix in fixed_prefixes})
    if requested and (requested[0] <= 0 or requested[-1] > indices.shape[-1]):
        raise ValueError("fixed prefix is outside the encoded stream")
    batch, _heads, tokens, stages = indices.shape
    local_indices = indices.to(dtype=torch.long, device=shared_codebooks[0].device)
    local_depths = stage_depths.to(device=shared_codebooks[0].device)
    mixed = torch.zeros(
        (batch, heads, tokens, dimension),
        dtype=shared_codebooks[0].dtype,
        device=shared_codebooks[0].device,
    )
    cumulative = torch.zeros_like(mixed)
    fixed: dict[int, torch.Tensor] = {}
    for stage in range(stages):
        if indices.numel() and int(indices[..., stage].max()) >= cardinalities[stage]:
            raise ValueError(f"stage {stage} index is outside its codebook")
        words = []
        for head in range(heads):
            stage_indices = local_indices[:, head, :, stage]
            base_table = (
                shared_codebooks[stage]
                if shared_codebooks[stage].ndim == 2
                else shared_codebooks[stage][head]
            )
            base = F.embedding(stage_indices, base_table)
            if entry_factors[stage].shape[-1] == 0:
                words.append(base)
            else:
                coefficients = F.embedding(stage_indices, entry_factors[stage][head])
                words.append(
                    base
                    + torch.matmul(coefficients, dimension_factors[stage][head])
                )
        stage_words = torch.stack(words, dim=1)
        cumulative = cumulative + stage_words
        mixed = mixed + stage_words * local_depths.gt(stage).unsqueeze(-1)
        prefix = stage + 1
        if prefix in requested:
            fixed[prefix] = cumulative.clone()
    return mixed, fixed


def factorized_parameter_bytes(
    shared_codebooks: Sequence[torch.Tensor],
    entry_factors: Sequence[torch.Tensor],
    dimension_factors: Sequence[torch.Tensor],
) -> int:
    """Unique tensor bytes of the factorized decoder parameters."""

    return sum(
        tensor.numel() * tensor.element_size()
        for collection in (shared_codebooks, entry_factors, dimension_factors)
        for tensor in collection
    )
