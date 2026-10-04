"""Honest storage accounting for pruning plus KIVI-style quantization.

The paper shorthand ``retention_ratio * bit_width`` counts payload bits only.
This module keeps that useful number, but also reports affine quantizer state,
token indices, alignment padding, and an optional FP16 decode tail separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, log2
from typing import Dict, Mapping, Sequence


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class CacheGeometry:
    layers: int
    kv_heads: int
    head_dim: int
    prompt_tokens: int
    batch_size: int = 1
    reference_bits: int = 16

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    @property
    def prompt_kv_scalars(self) -> int:
        return (
            self.batch_size
            * self.layers
            * self.kv_heads
            * self.head_dim
            * self.prompt_tokens
            * 2
        )


@dataclass(frozen=True)
class StorageLedger:
    payload_bits: int
    affine_metadata_bits: int
    token_index_bits: int
    control_metadata_bits: int
    alignment_padding_bits: int
    fp16_tail_bits: int
    reference_bits: int
    nominal_effective_bits: float
    realized_effective_bits: float
    compression_ratio: float

    @property
    def total_bits(self) -> int:
        return (
            self.payload_bits
            + self.affine_metadata_bits
            + self.token_index_bits
            + self.control_metadata_bits
            + self.alignment_padding_bits
            + self.fp16_tail_bits
        )

    def to_dict(self) -> Dict[str, int | float]:
        result = asdict(self)
        result["total_bits"] = self.total_bits
        return result


def uniform_kivi_ledger(
    geometry: CacheGeometry,
    *,
    retained_prompt_tokens: int,
    quant_bits: int,
    group_size: int = 64,
    scale_bits: int = 16,
    zero_bits: int = 16,
    explicit_index_tokens: int | None = None,
    indices_per_head: bool = True,
    fp16_decode_tokens: int = 0,
    alignment: int = 1,
) -> StorageLedger:
    """Account for a uniform-bit, uniformly retained prompt cache.

    Keys are modeled as per-channel groups along the token axis; values are
    modeled as per-token groups along the feature axis, matching KIVI's core
    asymmetry.  The ledger assumes packed integer payloads and FP16 scale/zero
    tensors.  It deliberately does not pretend that metadata is free.
    """

    if not 0 < retained_prompt_tokens <= geometry.prompt_tokens:
        raise ValueError("retained_prompt_tokens must be in [1, prompt_tokens]")
    if quant_bits not in {1, 2, 3, 4, 8, 16}:
        raise ValueError(f"unsupported quant_bits={quant_bits}")
    if group_size <= 0 or scale_bits <= 0 or zero_bits < 0:
        raise ValueError("group/stat widths must be valid positive values")
    if fp16_decode_tokens < 0:
        raise ValueError("fp16_decode_tokens cannot be negative")
    if alignment <= 0:
        raise ValueError("alignment must be positive")

    b = geometry.batch_size
    l = geometry.layers
    h = geometry.kv_heads
    d = geometry.head_dim
    t = retained_prompt_tokens

    padded_d = _ceil_div(d, alignment) * alignment
    payload_bits = b * l * h * t * padded_d * 2 * quant_bits
    alignment_padding_bits = b * l * h * t * (padded_d - d) * 2 * quant_bits
    payload_bits -= alignment_padding_bits

    if quant_bits == 16:
        affine_metadata_bits = 0
    else:
        stats_width = scale_bits + zero_bits
        key_groups = _ceil_div(t, group_size)
        value_groups = _ceil_div(d, group_size)
        key_stats = b * l * h * d * key_groups
        value_stats = b * l * h * t * value_groups
        affine_metadata_bits = (key_stats + value_stats) * stats_width

    if explicit_index_tokens is None:
        explicit_index_tokens = t if t < geometry.prompt_tokens else 0
    if not 0 <= explicit_index_tokens <= t:
        raise ValueError("explicit_index_tokens must be in [0, retained_prompt_tokens]")
    position_width = max(1, ceil(log2(max(2, geometry.prompt_tokens))))
    index_heads = h if indices_per_head else 1
    token_index_bits = b * l * index_heads * explicit_index_tokens * position_width

    fp16_tail_bits = b * l * h * d * fp16_decode_tokens * 2 * geometry.reference_bits
    reference_scalars = geometry.prompt_kv_scalars + b * l * h * d * fp16_decode_tokens * 2
    reference_bits = reference_scalars * geometry.reference_bits
    total_bits = (
        payload_bits
        + affine_metadata_bits
        + token_index_bits
        + alignment_padding_bits
        + fp16_tail_bits
    )
    nominal_effective_bits = (t / geometry.prompt_tokens) * quant_bits
    realized_effective_bits = total_bits / reference_scalars

    return StorageLedger(
        payload_bits=payload_bits,
        affine_metadata_bits=affine_metadata_bits,
        token_index_bits=token_index_bits,
        control_metadata_bits=0,
        alignment_padding_bits=alignment_padding_bits,
        fp16_tail_bits=fp16_tail_bits,
        reference_bits=reference_bits,
        nominal_effective_bits=nominal_effective_bits,
        realized_effective_bits=realized_effective_bits,
        compression_ratio=reference_bits / total_bits,
    )


def rdkv_mixed_ledger(
    geometry: CacheGeometry,
    *,
    value_bit_counts: Sequence[Mapping[int, int]],
    key_bit_counts: Sequence[Mapping[int, int]],
    retained_tokens_per_head: Sequence[int],
    scale_bits: int = 16,
    zero_bits: int = 16,
    boundary_bits: int = 32,
    include_channel_permutation: bool = True,
    fp16_decode_tokens: int = 0,
) -> StorageLedger:
    """Conservative artifact ledger for an RDKV/TriZone-style cache.

    Each sequence entry represents one batch/head slice.  Values use one
    affine quantizer per retained token and keys one affine quantizer per
    retained channel, matching the layout described in RDKV Appendix B.
    Payload padding for 2-bit/4-bit K-channel segments, segment boundaries,
    and the persistent channel permutation are explicitly charged.  No token
    positions are charged because post-RoPE old-cache rows may be jointly
    reordered for decode; callers using a layout that stores positions should
    add those bytes separately rather than claiming this lower-bound ledger.

    This accounts logical persistent storage, not allocator fragmentation,
    tensor headers, workspace, or a packed kernel's peak HBM allocation.
    """

    slices = geometry.batch_size * geometry.kv_heads
    if not (
        len(value_bit_counts)
        == len(key_bit_counts)
        == len(retained_tokens_per_head)
        == slices
    ):
        raise ValueError("mixed-bit summaries must have one entry per batch/head slice")
    if scale_bits <= 0 or zero_bits < 0 or boundary_bits <= 0:
        raise ValueError("metadata widths must be positive")
    if fp16_decode_tokens < 0:
        raise ValueError("fp16_decode_tokens cannot be negative")

    valid_bits = {0, 2, 4, 8, 16}
    payload_per_layer = 0
    affine_per_layer = 0
    padding_per_layer = 0
    control_per_layer = 0
    stats_width = scale_bits + zero_bits

    for value_counts, key_counts, retained in zip(
        value_bit_counts, key_bit_counts, retained_tokens_per_head
    ):
        if retained < 0 or retained > geometry.prompt_tokens:
            raise ValueError("retained token count is outside the prompt")
        if set(value_counts) - valid_bits or set(key_counts) - valid_bits:
            raise ValueError("RDKV counts use unsupported bitwidths")
        if sum(int(count) for count in value_counts.values()) != geometry.prompt_tokens:
            raise ValueError("value bit counts must cover every prompt token")
        if sum(int(count) for count in key_counts.values()) != geometry.head_dim:
            raise ValueError("key bit counts must cover every head channel")
        if sum(int(value_counts.get(bit, 0)) for bit in (2, 4, 8, 16)) != retained:
            raise ValueError("retained count must match nonzero V assignments")
        if any(int(count) < 0 for count in value_counts.values()) or any(
            int(count) < 0 for count in key_counts.values()
        ):
            raise ValueError("bit counts cannot be negative")

        # V stores d scalars for each retained token. K stores every retained
        # row but may remove individual channels.
        payload_per_layer += geometry.head_dim * sum(
            bit * int(value_counts.get(bit, 0)) for bit in (2, 4, 8, 16)
        )
        payload_per_layer += retained * sum(
            bit * int(key_counts.get(bit, 0)) for bit in (2, 4, 8, 16)
        )

        quantized_value_rows = sum(int(value_counts.get(bit, 0)) for bit in (2, 4, 8))
        quantized_key_channels = sum(int(key_counts.get(bit, 0)) for bit in (2, 4, 8))
        affine_per_layer += stats_width * (quantized_value_rows + quantized_key_channels)

        # K segment sizes must align to the byte packer's 4/2 channel lanes.
        key_2 = int(key_counts.get(2, 0))
        key_4 = int(key_counts.get(4, 0))
        padding_per_layer += retained * ((_ceil_div(key_2, 4) * 4 - key_2) * 2)
        padding_per_layer += retained * ((_ceil_div(key_4, 2) * 2 - key_4) * 4)

        # Five cumulative offsets (0/2/4/8/16) for each K and V stream.
        control_per_layer += 10 * boundary_bits
        if include_channel_permutation:
            permutation_width = max(1, ceil(log2(max(2, geometry.head_dim))))
            control_per_layer += geometry.head_dim * permutation_width

    payload_bits = geometry.layers * payload_per_layer
    affine_metadata_bits = geometry.layers * affine_per_layer
    alignment_padding_bits = geometry.layers * padding_per_layer
    token_index_bits = 0
    control_metadata_bits = geometry.layers * control_per_layer
    fp16_tail_bits = (
        geometry.batch_size
        * geometry.layers
        * geometry.kv_heads
        * geometry.head_dim
        * fp16_decode_tokens
        * 2
        * geometry.reference_bits
    )
    reference_scalars = geometry.prompt_kv_scalars + (
        geometry.batch_size
        * geometry.layers
        * geometry.kv_heads
        * geometry.head_dim
        * fp16_decode_tokens
        * 2
    )
    reference_bits = reference_scalars * geometry.reference_bits
    total_bits = (
        payload_bits
        + affine_metadata_bits
        + token_index_bits
        + control_metadata_bits
        + alignment_padding_bits
        + fp16_tail_bits
    )
    nominal_effective_bits = payload_bits / geometry.prompt_kv_scalars
    realized_effective_bits = total_bits / reference_scalars
    return StorageLedger(
        payload_bits=payload_bits,
        affine_metadata_bits=affine_metadata_bits,
        token_index_bits=token_index_bits,
        control_metadata_bits=control_metadata_bits,
        alignment_padding_bits=alignment_padding_bits,
        fp16_tail_bits=fp16_tail_bits,
        reference_bits=reference_bits,
        nominal_effective_bits=nominal_effective_bits,
        realized_effective_bits=realized_effective_bits,
        compression_ratio=reference_bits / total_bits,
    )


def max_uniform_capacity_for_artifact_rate(
    geometry: CacheGeometry,
    *,
    target_effective_bits: float,
    quant_bits: int,
    group_size: int = 64,
    recent_tokens: int = 0,
    **ledger_kwargs: int | bool,
) -> tuple[int, StorageLedger]:
    """Find the largest uniform retained-token count within a true rate cap."""

    if target_effective_bits <= 0:
        raise ValueError("target_effective_bits must be positive")
    if not 0 <= recent_tokens <= geometry.prompt_tokens:
        raise ValueError("recent_tokens must be within the prompt")

    def ledger(capacity: int) -> StorageLedger:
        return uniform_kivi_ledger(
            geometry,
            retained_prompt_tokens=capacity,
            quant_bits=quant_bits,
            group_size=group_size,
            explicit_index_tokens=max(0, capacity - recent_tokens),
            **ledger_kwargs,
        )

    low, high = 1, geometry.prompt_tokens
    first = ledger(low)
    if first.realized_effective_bits > target_effective_bits:
        raise ValueError("target rate cannot hold even one retained token")
    best_capacity, best_ledger = low, first
    while low <= high:
        middle = (low + high) // 2
        candidate = ledger(middle)
        if candidate.realized_effective_bits <= target_effective_bits:
            best_capacity, best_ledger = middle, candidate
            low = middle + 1
        else:
            high = middle - 1
    return best_capacity, best_ledger


def residual_vq_ledger(
    geometry: CacheGeometry,
    *,
    retained_prompt_tokens: int,
    stages: int,
    codebook_bits: int = 8,
    explicit_position_tokens: int | None = None,
    positions_per_head: bool = True,
    fp16_decode_tokens: int = 0,
) -> StorageLedger:
    """Dynamic-cache ledger for a per-token progressive residual VQ.

    One index is stored per stage for each retained K and V head vector.
    Codebooks are model-static parameters and intentionally excluded from the
    per-request KV rate; callers should report their bytes separately.  When
    eviction makes retained positions non-contiguous, one shared position is
    charged per retained K/V pair so pre-RoPE key codewords can be rotated at
    decode time.
    """

    if not 0 < retained_prompt_tokens <= geometry.prompt_tokens:
        raise ValueError("retained_prompt_tokens must be within the prompt")
    if stages <= 0 or codebook_bits <= 0:
        raise ValueError("VQ stage and code widths must be positive")
    if fp16_decode_tokens < 0:
        raise ValueError("fp16_decode_tokens cannot be negative")
    if explicit_position_tokens is None:
        explicit_position_tokens = (
            retained_prompt_tokens
            if retained_prompt_tokens < geometry.prompt_tokens
            else 0
        )
    if not 0 <= explicit_position_tokens <= retained_prompt_tokens:
        raise ValueError("explicit_position_tokens is outside retained tokens")

    vectors = (
        geometry.batch_size
        * geometry.layers
        * geometry.kv_heads
        * retained_prompt_tokens
        * 2
    )
    payload_bits = vectors * stages * codebook_bits
    position_width = max(1, ceil(log2(max(2, geometry.prompt_tokens))))
    token_index_bits = (
        geometry.batch_size
        * geometry.layers
        * (geometry.kv_heads if positions_per_head else 1)
        * explicit_position_tokens
        * position_width
    )
    fp16_tail_bits = (
        geometry.batch_size
        * geometry.layers
        * geometry.kv_heads
        * geometry.head_dim
        * fp16_decode_tokens
        * 2
        * geometry.reference_bits
    )
    reference_scalars = geometry.prompt_kv_scalars + (
        geometry.batch_size
        * geometry.layers
        * geometry.kv_heads
        * geometry.head_dim
        * fp16_decode_tokens
        * 2
    )
    reference_bits = reference_scalars * geometry.reference_bits
    total_bits = payload_bits + token_index_bits + fp16_tail_bits
    nominal_effective_bits = (
        retained_prompt_tokens
        / geometry.prompt_tokens
        * stages
        * codebook_bits
        / geometry.head_dim
    )
    return StorageLedger(
        payload_bits=payload_bits,
        affine_metadata_bits=0,
        token_index_bits=token_index_bits,
        control_metadata_bits=0,
        alignment_padding_bits=0,
        fp16_tail_bits=fp16_tail_bits,
        reference_bits=reference_bits,
        nominal_effective_bits=nominal_effective_bits,
        realized_effective_bits=total_bits / reference_scalars,
        compression_ratio=reference_bits / total_bits,
    )


def max_vq_capacity_for_artifact_rate(
    geometry: CacheGeometry,
    *,
    target_effective_bits: float,
    stages: int,
    codebook_bits: int = 8,
    positions_per_head: bool = True,
) -> tuple[int, StorageLedger]:
    """Largest retained-token count for a progressive VQ cache rate."""

    if target_effective_bits <= 0:
        raise ValueError("target_effective_bits must be positive")

    def ledger(capacity: int) -> StorageLedger:
        return residual_vq_ledger(
            geometry,
            retained_prompt_tokens=capacity,
            stages=stages,
            codebook_bits=codebook_bits,
            positions_per_head=positions_per_head,
        )

    low, high = 1, geometry.prompt_tokens
    first = ledger(low)
    if first.realized_effective_bits > target_effective_bits:
        raise ValueError("target rate cannot hold one VQ token")
    best_capacity, best_ledger = low, first
    while low <= high:
        middle = (low + high) // 2
        candidate = ledger(middle)
        if candidate.realized_effective_bits <= target_effective_bits:
            best_capacity, best_ledger = middle, candidate
            low = middle + 1
        else:
            high = middle - 1
    return best_capacity, best_ledger


def progressive_vq_action_ledger(
    geometry: CacheGeometry,
    *,
    paired_stage_counts: Mapping[int, int],
    codebook_bits: int = 8,
    retention_bitmap: bool = True,
    bitmap_per_head: bool = True,
    stage_depth_bits: int = 0,
    fp16_decode_tokens: int = 0,
) -> StorageLedger:
    """Ledger for joint eviction and paired progressive K/V VQ actions.

    ``paired_stage_counts`` counts token/head units assigned to each residual
    prefix length. Stage zero is eviction. A retained unit stores one index per
    stage for K and one for V.  A dense bitmap is charged when eviction is
    enabled, avoiding the optimistic assumption that a jagged token layout is
    free. ``stage_depth_bits`` charges the per-unit action/depth label needed
    when different units use different prefix lengths. Model-static codebooks
    remain excluded from the per-request cache rate and must be disclosed
    separately.
    """

    if codebook_bits <= 0 or stage_depth_bits < 0 or fp16_decode_tokens < 0:
        raise ValueError("code width and decode tail must be valid")
    if any(int(stage) < 0 or int(count) < 0 for stage, count in paired_stage_counts.items()):
        raise ValueError("stage ids and counts must be non-negative")
    units = (
        geometry.batch_size
        * geometry.layers
        * geometry.kv_heads
        * geometry.prompt_tokens
    )
    if sum(int(count) for count in paired_stage_counts.values()) != units:
        raise ValueError("stage histogram must cover every prompt token/head unit")

    payload_bits = sum(
        int(stage) * int(count) * codebook_bits * 2
        for stage, count in paired_stage_counts.items()
    )
    if retention_bitmap:
        bitmap_heads = geometry.kv_heads if bitmap_per_head else 1
        control_metadata_bits = (
            geometry.batch_size
            * geometry.layers
            * bitmap_heads
            * geometry.prompt_tokens
        )
    else:
        control_metadata_bits = 0
    control_metadata_bits += units * stage_depth_bits
    fp16_tail_bits = (
        geometry.batch_size
        * geometry.layers
        * geometry.kv_heads
        * geometry.head_dim
        * fp16_decode_tokens
        * 2
        * geometry.reference_bits
    )
    reference_scalars = geometry.prompt_kv_scalars + (
        geometry.batch_size
        * geometry.layers
        * geometry.kv_heads
        * geometry.head_dim
        * fp16_decode_tokens
        * 2
    )
    reference_bits = reference_scalars * geometry.reference_bits
    total_bits = payload_bits + control_metadata_bits + fp16_tail_bits
    return StorageLedger(
        payload_bits=payload_bits,
        affine_metadata_bits=0,
        token_index_bits=0,
        control_metadata_bits=control_metadata_bits,
        alignment_padding_bits=0,
        fp16_tail_bits=fp16_tail_bits,
        reference_bits=reference_bits,
        nominal_effective_bits=payload_bits / geometry.prompt_kv_scalars,
        realized_effective_bits=total_bits / reference_scalars,
        compression_ratio=reference_bits / total_bits,
    )


def variable_cardinality_vq_action_ledger(
    geometry: CacheGeometry,
    *,
    paired_action_counts: Mapping[int, int],
    paired_action_payload_bits: Mapping[int, int],
    action_label_bits: int,
    fp16_decode_tokens: int = 0,
) -> StorageLedger:
    """Ledger for stage-only nested VQ with arbitrary fixed-width stage costs.

    Keys of both mappings are prefix depths.  Payload rates already include the
    K and V index streams of one token/head pair.  No retention bitmap or
    position list is charged because this stage-only codec retains every token.
    """

    if action_label_bits < 0 or fp16_decode_tokens < 0:
        raise ValueError("label width and decode tail must be non-negative")
    if set(paired_action_counts) != set(paired_action_payload_bits):
        raise ValueError("action counts and rates must define the same depths")
    if any(
        int(depth) <= 0 or int(count) < 0 or int(paired_action_payload_bits[depth]) <= 0
        for depth, count in paired_action_counts.items()
    ):
        raise ValueError("nested VQ depths, counts, and rates must be valid")
    units = (
        geometry.batch_size
        * geometry.layers
        * geometry.kv_heads
        * geometry.prompt_tokens
    )
    if sum(int(count) for count in paired_action_counts.values()) != units:
        raise ValueError("action histogram must cover every token/head unit")
    payload_bits = sum(
        int(count) * int(paired_action_payload_bits[depth])
        for depth, count in paired_action_counts.items()
    )
    control_metadata_bits = units * action_label_bits
    fp16_tail_bits = (
        geometry.batch_size
        * geometry.layers
        * geometry.kv_heads
        * geometry.head_dim
        * fp16_decode_tokens
        * 2
        * geometry.reference_bits
    )
    reference_scalars = geometry.prompt_kv_scalars + (
        geometry.batch_size
        * geometry.layers
        * geometry.kv_heads
        * geometry.head_dim
        * fp16_decode_tokens
        * 2
    )
    reference_bits = reference_scalars * geometry.reference_bits
    total_bits = payload_bits + control_metadata_bits + fp16_tail_bits
    return StorageLedger(
        payload_bits=payload_bits,
        affine_metadata_bits=0,
        token_index_bits=0,
        control_metadata_bits=control_metadata_bits,
        alignment_padding_bits=0,
        fp16_tail_bits=fp16_tail_bits,
        reference_bits=reference_bits,
        nominal_effective_bits=payload_bits / geometry.prompt_kv_scalars,
        realized_effective_bits=total_bits / reference_scalars,
        compression_ratio=reference_bits / total_bits,
    )
