"""CPU reference layout for physically packed progressive nested-VQ indices.

The canonical codec has three positive exits.  Every K/V unit stores a base
prefix, while medium and high refinements are compacted per token block:

    base: stages [0, exit_0)
    mid:  stages [exit_0, exit_1) for units with depth >= exit_1
    high: stages [exit_1, exit_2) for units with depth == exit_2

Stage indices are bit-packed at their exact widths and one shared two-bit
action label selects the K/V-pair depth.  This module is intentionally a CPU
correctness and byte-accounting reference, not an accelerated attention path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Sequence

import torch

from .nested_vq import prefix_index_bits, validate_stage_bits


def _ceil_div(left: int, right: int) -> int:
    return (left + right - 1) // right


def pack_fixed_width(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack integer values into an LSB-first contiguous CPU byte stream."""

    width = int(bits)
    if width <= 0 or width > 8:
        raise ValueError("bits must lie in [1, 8]")
    flat = values.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    if flat.numel() == 0:
        return torch.empty(0, dtype=torch.uint8)
    if bool((flat < 0).any()) or bool((flat >= 1 << width).any()):
        raise ValueError(f"a value is outside the unsigned {width}-bit range")
    bit_positions = torch.arange(width, dtype=torch.int64)
    bitstream = ((flat[:, None] >> bit_positions) & 1).to(torch.uint8).reshape(-1)
    padding = (-int(bitstream.numel())) % 8
    if padding:
        bitstream = torch.cat((bitstream, torch.zeros(padding, dtype=torch.uint8)))
    byte_weights = (1 << torch.arange(8, dtype=torch.int64)).to(torch.uint8)
    return (bitstream.reshape(-1, 8) * byte_weights).sum(dim=1).to(torch.uint8)


def unpack_fixed_width(packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    """Invert :func:`pack_fixed_width` and return a flat ``uint8`` tensor."""

    width = int(bits)
    number = int(count)
    if width <= 0 or width > 8:
        raise ValueError("bits must lie in [1, 8]")
    if number < 0:
        raise ValueError("count must be non-negative")
    expected_bytes = _ceil_div(number * width, 8)
    flat_bytes = packed.detach().to(device="cpu", dtype=torch.uint8).reshape(-1)
    if flat_bytes.numel() != expected_bytes:
        raise ValueError(
            f"packed stream has {flat_bytes.numel()} bytes, expected {expected_bytes}"
        )
    if number == 0:
        return torch.empty(0, dtype=torch.uint8)
    bit_positions = torch.arange(8, dtype=torch.int64)
    bits_flat = (
        (flat_bytes.to(torch.int64)[:, None] >> bit_positions) & 1
    ).to(torch.uint8).reshape(-1)[: number * width]
    value_weights = (1 << torch.arange(width, dtype=torch.int64)).to(torch.uint8)
    return (bits_flat.reshape(number, width) * value_weights).sum(dim=1).to(torch.uint8)


@dataclass(frozen=True)
class PackedProgressiveStream:
    """Three byte streams for either K or V indices."""

    base: torch.Tensor
    mid: torch.Tensor
    high: torch.Tensor

    @property
    def allocated_bytes(self) -> int:
        return sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in (self.base, self.mid, self.high)
        )


@dataclass(frozen=True)
class PackedLayoutLedger:
    """Physical tensor-byte inventory for one layer's packed K/V cache."""

    payload_bits: int
    action_label_bits: int
    block_header_bits: int
    alignment_padding_bits: int
    reference_bits: int
    logical_effective_bits: float
    physical_effective_bits: float
    physical_compression_ratio: float

    @property
    def total_bits(self) -> int:
        return (
            self.payload_bits
            + self.action_label_bits
            + self.block_header_bits
            + self.alignment_padding_bits
        )

    def to_dict(self) -> dict[str, int | float]:
        result = asdict(self)
        result["total_bits"] = self.total_bits
        result["total_bytes"] = self.total_bits // 8
        return result


@dataclass(frozen=True)
class PackedNestedKVCache:
    """Packed K/V indices and the metadata needed for block-local decoding."""

    key: PackedProgressiveStream
    value: PackedProgressiveStream
    action_labels: torch.Tensor
    mid_offsets: torch.Tensor
    high_offsets: torch.Tensor
    shape: tuple[int, int, int]
    head_dim: int
    stage_bits: tuple[int, ...]
    exits: tuple[int, int, int]
    block_size: int

    @property
    def units(self) -> int:
        batch, heads, tokens = self.shape
        return batch * heads * tokens

    @property
    def blocks_per_row(self) -> int:
        return _ceil_div(self.shape[-1], self.block_size)

    @property
    def block_count(self) -> int:
        return self.shape[0] * self.shape[1] * self.blocks_per_row

    @property
    def allocated_tensor_bytes(self) -> int:
        tensors = (
            self.action_labels,
            self.mid_offsets,
            self.high_offsets,
        )
        return (
            self.key.allocated_bytes
            + self.value.allocated_bytes
            + sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors)
        )

    def ledger(self, *, reference_bits: int = 16) -> PackedLayoutLedger:
        if reference_bits <= 0:
            raise ValueError("reference_bits must be positive")
        prefix_rates = prefix_index_bits(self.stage_bits, self.exits)
        depths = unpack_action_depths(self)
        payload_bits = 2 * sum(
            int(depths.eq(depth).sum()) * prefix_rates[depth]
            for depth in self.exits
        )
        action_label_bits = self.units * 2
        block_header_bits = sum(
            int(offsets.numel() * offsets.element_size() * 8)
            for offsets in (self.mid_offsets, self.high_offsets)
        )
        stream_allocated_bits = (
            self.key.allocated_bytes + self.value.allocated_bytes
        ) * 8
        label_allocated_bits = int(
            self.action_labels.numel() * self.action_labels.element_size() * 8
        )
        alignment_padding_bits = (
            stream_allocated_bits - payload_bits
            + label_allocated_bits - action_label_bits
        )
        if alignment_padding_bits < 0:
            raise RuntimeError("packed allocation is smaller than its logical fields")
        reference_scalars = 2 * self.units * self.head_dim
        reference_total = reference_scalars * int(reference_bits)
        total = (
            payload_bits
            + action_label_bits
            + block_header_bits
            + alignment_padding_bits
        )
        if total != self.allocated_tensor_bytes * 8:
            raise RuntimeError("physical ledger differs from allocated tensor bytes")
        return PackedLayoutLedger(
            payload_bits=payload_bits,
            action_label_bits=action_label_bits,
            block_header_bits=block_header_bits,
            alignment_padding_bits=alignment_padding_bits,
            reference_bits=reference_total,
            logical_effective_bits=(payload_bits + action_label_bits)
            / reference_scalars,
            physical_effective_bits=total / reference_scalars,
            physical_compression_ratio=reference_total / total,
        )


def estimate_paired_progressive_layout(
    depths: torch.Tensor,
    *,
    stage_bits: Sequence[int],
    exits: Sequence[int] = (14, 22, 30),
    head_dim: int = 128,
    block_size: int = 128,
    reference_bits: int = 16,
) -> PackedLayoutLedger:
    """Estimate exact packed tensor bytes without allocating index payloads."""

    schedule = validate_stage_bits(stage_bits)
    actions = tuple(int(depth) for depth in exits)
    if len(actions) != 3 or tuple(sorted(set(actions))) != actions:
        raise ValueError("exactly three sorted unique exits are required")
    if actions[0] <= 0 or actions[-1] > len(schedule):
        raise ValueError("exits must lie inside the stage schedule")
    if depths.ndim != 3 or any(int(value) <= 0 for value in depths.shape):
        raise ValueError("depths must have positive [B,H,T] layout")
    if head_dim <= 0 or block_size <= 0 or reference_bits <= 0:
        raise ValueError("dimensions, block size, and reference width must be positive")
    local_depths = depths.detach().to(device="cpu", dtype=torch.int64)
    valid = torch.zeros_like(local_depths, dtype=torch.bool)
    for depth in actions:
        valid |= local_depths.eq(depth)
    if not bool(valid.all()):
        raise ValueError("depth tensor contains an action outside the three exits")

    shape = tuple(int(value) for value in local_depths.shape)
    batch, heads, tokens = shape
    units = batch * heads * tokens
    prefix_rates = prefix_index_bits(schedule, actions)
    payload_bits = 2 * sum(
        int(local_depths.eq(depth).sum()) * prefix_rates[depth]
        for depth in actions
    )
    action_label_bits = units * 2
    label_allocated_bits = _ceil_div(action_label_bits, 8) * 8
    rows = batch * heads
    flat_depths = local_depths.reshape(rows, tokens)
    blocks_per_row = _ceil_div(tokens, block_size)
    padded_tokens = blocks_per_row * block_size
    if padded_tokens != tokens:
        flat_depths = torch.cat(
            (
                flat_depths,
                torch.full(
                    (rows, padded_tokens - tokens),
                    actions[0],
                    dtype=flat_depths.dtype,
                ),
            ),
            dim=1,
        )
    depth_blocks = flat_depths.reshape(rows, blocks_per_row, block_size)
    base_counts = torch.full(
        (rows, blocks_per_row), block_size, dtype=torch.int64
    )
    base_counts[:, -1] = tokens - (blocks_per_row - 1) * block_size
    tier_counts = (
        base_counts,
        depth_blocks.ge(actions[1]).sum(dim=-1),
        depth_blocks.ge(actions[2]).sum(dim=-1),
    )
    ranges = (
        (0, actions[0]),
        (actions[0], actions[1]),
        (actions[1], actions[2]),
    )
    one_kind_stream_bytes = 0
    for counts, (first, last) in zip(tier_counts, ranges):
        for stage in range(first, last):
            one_kind_stream_bytes += int(
                ((counts * schedule[stage] + 7) // 8).sum()
            )
    stream_allocated_bits = one_kind_stream_bytes * 2 * 8
    block_count = batch * heads * blocks_per_row
    block_header_bits = 2 * (block_count + 1) * 32
    alignment_padding_bits = (
        stream_allocated_bits - payload_bits
        + label_allocated_bits - action_label_bits
    )
    reference_scalars = 2 * units * int(head_dim)
    reference_total = reference_scalars * int(reference_bits)
    total = (
        payload_bits
        + action_label_bits
        + block_header_bits
        + alignment_padding_bits
    )
    return PackedLayoutLedger(
        payload_bits=payload_bits,
        action_label_bits=action_label_bits,
        block_header_bits=block_header_bits,
        alignment_padding_bits=alignment_padding_bits,
        reference_bits=reference_total,
        logical_effective_bits=(payload_bits + action_label_bits) / reference_scalars,
        physical_effective_bits=total / reference_scalars,
        physical_compression_ratio=reference_total / total,
    )


def estimate_layered_paired_progressive_layout(
    depths: torch.Tensor,
    *,
    stage_bits: Sequence[int],
    exits: Sequence[int] = (14, 22, 30),
    head_dim: int = 128,
    block_size: int = 128,
    reference_bits: int = 16,
) -> PackedLayoutLedger:
    """Sum independent per-layer packed streams for ``[L,B,H,T]`` depths.

    Each layer owns its own action stream and terminal mid/high offsets.  This
    distinction matters because concatenating layers before applying byte
    alignment would undercount the real persistent tensor layout.
    """

    if depths.ndim != 4 or any(int(value) <= 0 for value in depths.shape):
        raise ValueError("depths must have positive [L,B,H,T] layout")
    ledgers = [
        estimate_paired_progressive_layout(
            depths[layer],
            stage_bits=stage_bits,
            exits=exits,
            head_dim=head_dim,
            block_size=block_size,
            reference_bits=reference_bits,
        )
        for layer in range(depths.shape[0])
    ]
    payload_bits = sum(ledger.payload_bits for ledger in ledgers)
    action_label_bits = sum(ledger.action_label_bits for ledger in ledgers)
    block_header_bits = sum(ledger.block_header_bits for ledger in ledgers)
    alignment_padding_bits = sum(
        ledger.alignment_padding_bits for ledger in ledgers
    )
    reference_total = sum(ledger.reference_bits for ledger in ledgers)
    reference_scalars = 2 * depths.numel() * int(head_dim)
    total = (
        payload_bits
        + action_label_bits
        + block_header_bits
        + alignment_padding_bits
    )
    return PackedLayoutLedger(
        payload_bits=payload_bits,
        action_label_bits=action_label_bits,
        block_header_bits=block_header_bits,
        alignment_padding_bits=alignment_padding_bits,
        reference_bits=reference_total,
        logical_effective_bits=(payload_bits + action_label_bits)
        / reference_scalars,
        physical_effective_bits=total / reference_scalars,
        physical_compression_ratio=reference_total / total,
    )


def estimate_block_shared_paired_progressive_layout(
    block_depths: torch.Tensor,
    token_counts: torch.Tensor,
    *,
    stage_bits: Sequence[int],
    exits: Sequence[int] = (14, 22, 30),
    head_dim: int = 128,
    reference_bits: int = 16,
) -> PackedLayoutLedger:
    """Exact tensor-byte ledger when one depth is shared by a token block.

    ``block_depths`` uses ``[B,H,N_blocks]`` layout. ``token_counts`` contains
    the number of real tokens in each block position of one row. The persistent
    representation has one packed two-bit label per block and one packed index
    tensor per K/V stage. Fixed block boundaries plus the labels determine all
    stream lengths, so no persistent per-block offsets are necessary.
    """

    schedule = validate_stage_bits(stage_bits)
    actions = tuple(int(depth) for depth in exits)
    if len(actions) != 3 or tuple(sorted(set(actions))) != actions:
        raise ValueError("exactly three sorted unique exits are required")
    if actions[0] <= 0 or actions[-1] > len(schedule):
        raise ValueError("exits must lie inside the stage schedule")
    if block_depths.ndim != 3 or any(
        int(value) <= 0 for value in block_depths.shape
    ):
        raise ValueError("block depths must have positive [B,H,N_blocks] layout")
    if token_counts.ndim != 1 or token_counts.shape[0] != block_depths.shape[-1]:
        raise ValueError("token counts must have one entry per block position")
    if head_dim <= 0 or reference_bits <= 0:
        raise ValueError("head dimension and reference width must be positive")

    local_depths = block_depths.detach().to(device="cpu", dtype=torch.int64)
    local_counts = token_counts.detach().to(device="cpu", dtype=torch.int64)
    if bool((local_counts <= 0).any()):
        raise ValueError("every represented block must contain a real token")
    valid = torch.zeros_like(local_depths, dtype=torch.bool)
    for depth in actions:
        valid |= local_depths.eq(depth)
    if not bool(valid.all()):
        raise ValueError("block depth tensor contains an unknown action")

    batch, heads, blocks = map(int, local_depths.shape)
    expanded_counts = local_counts.reshape(1, 1, blocks).expand_as(local_depths)
    units = int(expanded_counts.sum())
    prefix_rates = prefix_index_bits(schedule, actions)
    payload_bits = 2 * sum(
        int(expanded_counts[local_depths.eq(depth)].sum()) * prefix_rates[depth]
        for depth in actions
    )

    # Each K/V stage is an independent byte tensor. Concatenating active blocks
    # within it removes block-internal padding but retains terminal-byte rounding.
    one_kind_allocated_bits = 0
    for stage, width in enumerate(schedule):
        active_tokens = int(expanded_counts[local_depths.gt(stage)].sum())
        one_kind_allocated_bits += _ceil_div(active_tokens * width, 8) * 8
    stream_allocated_bits = 2 * one_kind_allocated_bits

    block_count = batch * heads * blocks
    action_label_bits = block_count * 2
    label_allocated_bits = _ceil_div(action_label_bits, 8) * 8
    block_header_bits = 0
    alignment_padding_bits = (
        stream_allocated_bits - payload_bits
        + label_allocated_bits - action_label_bits
    )
    if alignment_padding_bits < 0:
        raise RuntimeError("block-shared allocation is smaller than its fields")
    reference_scalars = 2 * units * int(head_dim)
    reference_total = reference_scalars * int(reference_bits)
    total = payload_bits + action_label_bits + alignment_padding_bits
    return PackedLayoutLedger(
        payload_bits=payload_bits,
        action_label_bits=action_label_bits,
        block_header_bits=block_header_bits,
        alignment_padding_bits=alignment_padding_bits,
        reference_bits=reference_total,
        logical_effective_bits=(payload_bits + action_label_bits)
        / reference_scalars,
        physical_effective_bits=total / reference_scalars,
        physical_compression_ratio=reference_total / total,
    )


def _validate_layout_arguments(
    key_indices: torch.Tensor,
    value_indices: torch.Tensor,
    depths: torch.Tensor,
    *,
    stage_bits: Sequence[int],
    exits: Sequence[int],
    head_dim: int,
    block_size: int,
) -> tuple[tuple[int, ...], tuple[int, int, int]]:
    schedule = validate_stage_bits(stage_bits)
    actions = tuple(int(depth) for depth in exits)
    if len(actions) != 3 or tuple(sorted(set(actions))) != actions:
        raise ValueError("exactly three sorted unique exits are required")
    if actions[0] <= 0 or actions[-1] > len(schedule):
        raise ValueError("exits must lie inside the stage schedule")
    if key_indices.shape != value_indices.shape or key_indices.ndim != 4:
        raise ValueError("K/V indices must share [B,H,T,S] layout")
    if depths.shape != key_indices.shape[:-1]:
        raise ValueError("depths must have [B,H,T] layout")
    if key_indices.shape[-1] < actions[-1]:
        raise ValueError("indices do not contain the maximum exit")
    if head_dim <= 0 or block_size <= 0:
        raise ValueError("head_dim and block_size must be positive")
    local_depths = depths.detach().to(device="cpu", dtype=torch.int64)
    valid = torch.zeros_like(local_depths, dtype=torch.bool)
    for depth in actions:
        valid |= local_depths.eq(depth)
    if not bool(valid.all()):
        raise ValueError("depth tensor contains an action outside the three exits")
    return schedule, actions  # type: ignore[return-value]


def _block_slices(shape: tuple[int, int, int], block_size: int):
    batch, heads, tokens = shape
    for row in range(batch * heads):
        for start in range(0, tokens, block_size):
            yield row, start, min(tokens, start + block_size)


def _pack_stream(
    indices: torch.Tensor,
    depths: torch.Tensor,
    *,
    stage_bits: tuple[int, ...],
    exits: tuple[int, int, int],
    block_size: int,
) -> tuple[PackedProgressiveStream, tuple[list[int], list[int], list[int]]]:
    batch, heads, tokens, _stages = indices.shape
    rows = batch * heads
    local_indices = indices.detach().to(device="cpu", dtype=torch.int64).reshape(
        rows, tokens, -1
    )
    local_depths = depths.detach().to(device="cpu", dtype=torch.int64).reshape(
        rows, tokens
    )
    tier_ranges = ((0, exits[0]), (exits[0], exits[1]), (exits[1], exits[2]))
    pieces: list[list[torch.Tensor]] = [[], [], []]
    offsets: list[list[int]] = [[0], [0], [0]]
    for row, start, stop in _block_slices((batch, heads, tokens), block_size):
        block_depths = local_depths[row, start:stop]
        masks = (
            torch.ones_like(block_depths, dtype=torch.bool),
            block_depths.ge(exits[1]),
            block_depths.ge(exits[2]),
        )
        for tier, ((first, last), mask) in enumerate(zip(tier_ranges, masks)):
            for stage in range(first, last):
                selected = local_indices[row, start:stop, stage][mask]
                pieces[tier].append(pack_fixed_width(selected, stage_bits[stage]))
            offsets[tier].append(
                offsets[tier][-1]
                + sum(int(piece.numel()) for piece in pieces[tier][- (last - first) :])
            )
    streams = []
    for local_pieces in pieces:
        streams.append(
            torch.cat(local_pieces)
            if local_pieces
            else torch.empty(0, dtype=torch.uint8)
        )
    return (
        PackedProgressiveStream(*streams),
        (offsets[0], offsets[1], offsets[2]),
    )


def _offset_tensor(values: list[int]) -> torch.Tensor:
    if values[-1] > torch.iinfo(torch.int32).max:
        raise ValueError("a per-layer stream exceeds the int32 offset range")
    return torch.tensor(values, dtype=torch.int32)


def pack_paired_progressive_indices(
    key_indices: torch.Tensor,
    value_indices: torch.Tensor,
    depths: torch.Tensor,
    *,
    stage_bits: Sequence[int],
    exits: Sequence[int] = (14, 22, 30),
    head_dim: int = 128,
    block_size: int = 128,
) -> PackedNestedKVCache:
    """Pack one layer of paired K/V progressive indices on CPU."""

    schedule, actions = _validate_layout_arguments(
        key_indices,
        value_indices,
        depths,
        stage_bits=stage_bits,
        exits=exits,
        head_dim=head_dim,
        block_size=block_size,
    )
    key_stream, key_offsets = _pack_stream(
        key_indices,
        depths,
        stage_bits=schedule,
        exits=actions,
        block_size=block_size,
    )
    value_stream, value_offsets = _pack_stream(
        value_indices,
        depths,
        stage_bits=schedule,
        exits=actions,
        block_size=block_size,
    )
    if key_offsets != value_offsets:
        raise RuntimeError("paired K/V streams produced different block geometry")
    local_depths = depths.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    labels = torch.empty_like(local_depths, dtype=torch.uint8)
    for label, depth in enumerate(actions):
        labels[local_depths.eq(depth)] = label
    cache = PackedNestedKVCache(
        key=key_stream,
        value=value_stream,
        action_labels=pack_fixed_width(labels, 2),
        mid_offsets=_offset_tensor(key_offsets[1]),
        high_offsets=_offset_tensor(key_offsets[2]),
        shape=tuple(int(value) for value in depths.shape),
        head_dim=int(head_dim),
        stage_bits=schedule,
        exits=actions,
        block_size=int(block_size),
    )
    validate_packed_nested_cache(cache)
    return cache


def unpack_action_depths(cache: PackedNestedKVCache) -> torch.Tensor:
    labels = unpack_fixed_width(cache.action_labels, 2, cache.units).to(torch.int64)
    if labels.numel() and int(labels.max()) >= len(cache.exits):
        raise ValueError("action map contains its reserved two-bit code")
    actions = torch.tensor(cache.exits, dtype=torch.int16)
    return actions.index_select(0, labels).reshape(cache.shape)


def _derived_base_offsets(cache: PackedNestedKVCache) -> torch.Tensor:
    """Derive dense-base block offsets without persistent metadata."""

    offsets = [0]
    for _row, start, stop in _block_slices(cache.shape, cache.block_size):
        count = stop - start
        block_bytes = sum(
            _ceil_div(count * cache.stage_bits[stage], 8)
            for stage in range(cache.exits[0])
        )
        offsets.append(offsets[-1] + block_bytes)
    return torch.tensor(offsets, dtype=torch.int64)


def _unpack_stream(
    cache: PackedNestedKVCache,
    stream: PackedProgressiveStream,
    depths: torch.Tensor,
) -> torch.Tensor:
    batch, heads, tokens = cache.shape
    rows = batch * heads
    result = torch.zeros(
        (rows, tokens, len(cache.stage_bits)), dtype=torch.uint8
    )
    local_depths = depths.reshape(rows, tokens)
    tier_ranges = (
        (0, cache.exits[0]),
        (cache.exits[0], cache.exits[1]),
        (cache.exits[1], cache.exits[2]),
    )
    tier_data = (stream.base, stream.mid, stream.high)
    tier_offsets = (_derived_base_offsets(cache), cache.mid_offsets, cache.high_offsets)
    for block, (row, start, stop) in enumerate(
        _block_slices(cache.shape, cache.block_size)
    ):
        block_depths = local_depths[row, start:stop]
        masks = (
            torch.ones_like(block_depths, dtype=torch.bool),
            block_depths.ge(cache.exits[1]),
            block_depths.ge(cache.exits[2]),
        )
        for (first, last), data, offsets, mask in zip(
            tier_ranges, tier_data, tier_offsets, masks
        ):
            cursor = int(offsets[block])
            count = int(mask.sum())
            for stage in range(first, last):
                width = cache.stage_bits[stage]
                byte_count = _ceil_div(count * width, 8)
                selected = unpack_fixed_width(
                    data[cursor : cursor + byte_count], width, count
                )
                local = result[row, start:stop, stage]
                local[mask] = selected
                cursor += byte_count
            if cursor != int(offsets[block + 1]):
                raise ValueError("block offset does not match its packed stage segments")
    return result.reshape(batch, heads, tokens, len(cache.stage_bits))


def unpack_paired_progressive_indices(
    cache: PackedNestedKVCache,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode physically stored indices; inactive suffix entries are zero."""

    validate_packed_nested_cache(cache)
    depths = unpack_action_depths(cache)
    key = _unpack_stream(cache, cache.key, depths)
    value = _unpack_stream(cache, cache.value, depths)
    return key, value, depths


def validate_packed_nested_cache(cache: PackedNestedKVCache) -> None:
    """Reject malformed physical streams before an unpack or kernel handoff."""

    if len(cache.shape) != 3 or any(value <= 0 for value in cache.shape):
        raise ValueError("shape must contain positive [B,H,T] dimensions")
    if cache.head_dim <= 0 or cache.block_size <= 0:
        raise ValueError("head_dim and block_size must be positive")
    validate_stage_bits(cache.stage_bits)
    if (
        len(cache.exits) != 3
        or tuple(sorted(set(cache.exits))) != cache.exits
        or cache.exits[0] <= 0
        or cache.exits[-1] > len(cache.stage_bits)
    ):
        raise ValueError("cache exits are invalid")
    for tensor in (
        cache.key.base,
        cache.key.mid,
        cache.key.high,
        cache.value.base,
        cache.value.mid,
        cache.value.high,
        cache.action_labels,
    ):
        if tensor.device.type != "cpu" or tensor.dtype != torch.uint8 or tensor.ndim != 1:
            raise ValueError("packed byte streams must be flat CPU uint8 tensors")
    expected_labels = _ceil_div(cache.units * 2, 8)
    if cache.action_labels.numel() != expected_labels:
        raise ValueError("action-map byte length differs from the cache geometry")
    base_offsets = _derived_base_offsets(cache)
    if int(base_offsets[-1]) != cache.key.base.numel() or int(
        base_offsets[-1]
    ) != cache.value.base.numel():
        raise ValueError("derived base geometry differs from a paired stream size")
    for offsets, key_data, value_data in zip(
        (cache.mid_offsets, cache.high_offsets),
        (cache.key.mid, cache.key.high),
        (cache.value.mid, cache.value.high),
    ):
        if (
            offsets.device.type != "cpu"
            or offsets.dtype != torch.int32
            or offsets.ndim != 1
            or offsets.numel() != cache.block_count + 1
        ):
            raise ValueError("block offsets must be flat CPU int32 tensors")
        local = offsets.to(torch.int64)
        if int(local[0]) != 0 or bool((local[1:] < local[:-1]).any()):
            raise ValueError("block offsets must be monotone and start at zero")
        if int(local[-1]) != key_data.numel() or int(local[-1]) != value_data.numel():
            raise ValueError("terminal block offset differs from a paired stream size")
