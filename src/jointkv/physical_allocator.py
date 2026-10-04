"""Exact-cap repair for the physically packed nested-VQ layout.

The rich RD allocator first chooses one of three positive progressive exits.
Its logical payload can meet a nominal rate while block offsets and byte
alignment still overflow the physical cap.  This module applies the cheapest
adjacent depth downgrades, ordered by marginal distortion per logical payload
bit, until the *exact* packed tensor ledger is feasible.  A reverse add-back
pass reclaims padding slack without ever exceeding the byte cap.

This is a deterministic CPU reference and capacity guarantee, not an exact
MCKP optimum or a deployment-time selector.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Sequence

import torch

from .nested_vq import paired_prefix_rates, validate_stage_bits
from .packed_nested import (
    PackedLayoutLedger,
    estimate_layered_paired_progressive_layout,
)


@dataclass(frozen=True)
class PackedBudgetRepairResult:
    """Result of repairing one layered action map to an exact packed cap."""

    actions: torch.Tensor
    depths: torch.Tensor
    initial_ledger: PackedLayoutLedger
    final_ledger: PackedLayoutLedger
    budget_bits: int
    initial_objective: float
    final_objective: float
    attempted_downgrade_steps: int
    retained_downgrade_steps: int
    downgraded_units: int
    zero_saving_steps: int
    addback_steps: int

    @property
    def distortion_increase(self) -> float:
        return self.final_objective - self.initial_objective

    @property
    def physical_bits_saved(self) -> int:
        return self.initial_ledger.total_bits - self.final_ledger.total_bits

    @property
    def budget_slack_bits(self) -> int:
        return self.budget_bits - self.final_ledger.total_bits

    def summary_dict(self) -> dict[str, int | float | dict[str, int | float]]:
        return {
            "budget_bits": self.budget_bits,
            "budget_slack_bits": self.budget_slack_bits,
            "physical_bits_saved": self.physical_bits_saved,
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "distortion_increase": self.distortion_increase,
            "attempted_downgrade_steps": self.attempted_downgrade_steps,
            "retained_downgrade_steps": self.retained_downgrade_steps,
            "downgraded_units": self.downgraded_units,
            "zero_saving_steps": self.zero_saving_steps,
            "addback_steps": self.addback_steps,
            "initial_ledger": self.initial_ledger.to_dict(),
            "final_ledger": self.final_ledger.to_dict(),
        }


def _tier_bytes(count: int, widths: Sequence[int]) -> int:
    return sum((int(count) * int(width) + 7) // 8 for width in widths)


def _actions_to_depths(
    actions: torch.Tensor, exits: tuple[int, int, int]
) -> torch.Tensor:
    table = torch.tensor(exits, dtype=torch.int16)
    return table.index_select(0, actions.reshape(-1)).reshape(actions.shape)


def repair_depth_actions_to_packed_budget(
    distortions: torch.Tensor,
    actions: torch.Tensor,
    *,
    budget_bits: int,
    stage_bits: Sequence[int],
    exits: Sequence[int] = (14, 22, 30),
    head_dim: int = 128,
    block_size: int = 128,
    reference_bits: int = 16,
) -> PackedBudgetRepairResult:
    """Repair ``[L,B,H,T]`` action indices to an exact physical bit cap.

    ``distortions`` must have ``[L,B,H,T,3]`` layout and action indices map to
    the sorted ``exits``.  Only downgrades relative to the supplied rich-oracle
    allocation are considered.  The returned tensors live on CPU because this
    routine is an offline teacher/reference implementation.
    """

    schedule = validate_stage_bits(stage_bits)
    local_exits = tuple(int(depth) for depth in exits)
    if len(local_exits) != 3 or tuple(sorted(set(local_exits))) != local_exits:
        raise ValueError("exactly three sorted unique exits are required")
    if local_exits[0] <= 0 or local_exits[-1] > len(schedule):
        raise ValueError("exits must lie inside the stage schedule")
    if actions.ndim != 4 or any(int(value) <= 0 for value in actions.shape):
        raise ValueError("actions must have positive [L,B,H,T] layout")
    if distortions.shape != (*actions.shape, 3):
        raise ValueError("distortions must have [L,B,H,T,3] layout")
    if budget_bits < 0 or head_dim <= 0 or block_size <= 0:
        raise ValueError("budget and layout dimensions are invalid")
    if not bool(torch.isfinite(distortions).all()):
        raise ValueError("distortions must be finite")

    original = actions.detach().to(device="cpu", dtype=torch.int64).clone()
    if bool((original < 0).any()) or bool((original >= 3).any()):
        raise ValueError("an action index lies outside the three exits")
    local_d = distortions.detach().to(device="cpu", dtype=torch.float64)
    current = original.clone()
    initial_depths = _actions_to_depths(current, local_exits)
    initial_ledger = estimate_layered_paired_progressive_layout(
        initial_depths,
        stage_bits=schedule,
        exits=local_exits,
        head_dim=head_dim,
        block_size=block_size,
        reference_bits=reference_bits,
    )
    minimum_depths = torch.full_like(initial_depths, local_exits[0])
    minimum_ledger = estimate_layered_paired_progressive_layout(
        minimum_depths,
        stage_bits=schedule,
        exits=local_exits,
        head_dim=head_dim,
        block_size=block_size,
        reference_bits=reference_bits,
    )
    if minimum_ledger.total_bits > budget_bits:
        raise ValueError("budget is below the minimum packed positive exit")

    flat_d = local_d.reshape(-1, 3)
    flat_current = current.reshape(-1)
    rows = torch.arange(flat_current.numel())
    initial_objective = float(
        flat_d[rows, flat_current].sum(dtype=torch.float64).item()
    )
    if initial_ledger.total_bits <= budget_bits:
        return PackedBudgetRepairResult(
            actions=current,
            depths=initial_depths,
            initial_ledger=initial_ledger,
            final_ledger=initial_ledger,
            budget_bits=int(budget_bits),
            initial_objective=initial_objective,
            final_objective=initial_objective,
            attempted_downgrade_steps=0,
            retained_downgrade_steps=0,
            downgraded_units=0,
            zero_saving_steps=0,
            addback_steps=0,
        )

    rates_by_depth = paired_prefix_rates(schedule, local_exits)
    rates = tuple(rates_by_depth[depth] for depth in local_exits)
    layers, batch, heads, tokens = map(int, current.shape)
    rows_per_layer = batch * heads
    blocks_per_row = (tokens + block_size - 1) // block_size
    padded_tokens = blocks_per_row * block_size
    grid = current.reshape(layers, rows_per_layer, tokens)
    if padded_tokens != tokens:
        grid = torch.cat(
            (
                grid,
                torch.zeros(
                    (layers, rows_per_layer, padded_tokens - tokens),
                    dtype=grid.dtype,
                ),
            ),
            dim=-1,
        )
    blocks = grid.reshape(layers, rows_per_layer, blocks_per_row, block_size)
    mid_counts = blocks.ge(1).sum(dim=-1).to(torch.int64)
    high_counts = blocks.ge(2).sum(dim=-1).to(torch.int64)
    tier_widths = (
        tuple(schedule[local_exits[0] : local_exits[1]]),
        tuple(schedule[local_exits[1] : local_exits[2]]),
    )

    heap: list[tuple[float, float, int, int]] = []

    def push_candidate(unit: int) -> None:
        source = int(flat_current[unit])
        if source <= 0:
            return
        target = source - 1
        delta = float(flat_d[unit, target] - flat_d[unit, source])
        nominal_saved = rates[source] - rates[target]
        heapq.heappush(
            heap,
            (delta / nominal_saved, delta, int(unit), source),
        )

    for unit in flat_current.gt(0).nonzero(as_tuple=False).flatten().tolist():
        push_candidate(int(unit))

    total_bits = initial_ledger.total_bits
    history: list[tuple[int, int, int, float, int]] = []
    zero_saving_steps = 0
    units_per_layer = rows_per_layer * tokens
    while total_bits > budget_bits:
        if not heap:
            raise RuntimeError("downgrade heap exhausted before reaching the packed cap")
        _score, delta, unit, source = heapq.heappop(heap)
        if int(flat_current[unit]) != source:
            continue
        target = source - 1
        layer = unit // units_per_layer
        within_layer = unit % units_per_layer
        row = within_layer // tokens
        token = within_layer % tokens
        block = token // block_size
        counts = high_counts if source == 2 else mid_counts
        widths = tier_widths[1] if source == 2 else tier_widths[0]
        before_count = int(counts[layer, row, block])
        if before_count <= 0:
            raise RuntimeError("packed tier count underflow")
        physical_saved = 16 * (
            _tier_bytes(before_count, widths)
            - _tier_bytes(before_count - 1, widths)
        )
        counts[layer, row, block] -= 1
        flat_current[unit] = target
        total_bits -= physical_saved
        if physical_saved == 0:
            zero_saving_steps += 1
        history.append((unit, source, target, delta, physical_saved))
        push_candidate(unit)

    addback_steps = 0
    for unit, source, target, delta, _physical_saved in reversed(history):
        if delta <= 0.0 or int(flat_current[unit]) != target:
            continue
        layer = unit // units_per_layer
        within_layer = unit % units_per_layer
        row = within_layer // tokens
        token = within_layer % tokens
        block = token // block_size
        counts = high_counts if source == 2 else mid_counts
        widths = tier_widths[1] if source == 2 else tier_widths[0]
        before_count = int(counts[layer, row, block])
        physical_cost = 16 * (
            _tier_bytes(before_count + 1, widths)
            - _tier_bytes(before_count, widths)
        )
        if total_bits + physical_cost > budget_bits:
            continue
        counts[layer, row, block] += 1
        flat_current[unit] = source
        total_bits += physical_cost
        addback_steps += 1

    final_depths = _actions_to_depths(current, local_exits)
    final_ledger = estimate_layered_paired_progressive_layout(
        final_depths,
        stage_bits=schedule,
        exits=local_exits,
        head_dim=head_dim,
        block_size=block_size,
        reference_bits=reference_bits,
    )
    if final_ledger.total_bits != total_bits or total_bits > budget_bits:
        raise RuntimeError("incremental packed repair disagrees with the exact ledger")
    final_objective = float(
        flat_d[rows, flat_current].sum(dtype=torch.float64).item()
    )
    retained_steps = int((original - current).sum().item())
    return PackedBudgetRepairResult(
        actions=current,
        depths=final_depths,
        initial_ledger=initial_ledger,
        final_ledger=final_ledger,
        budget_bits=int(budget_bits),
        initial_objective=initial_objective,
        final_objective=final_objective,
        attempted_downgrade_steps=len(history),
        retained_downgrade_steps=retained_steps,
        downgraded_units=int(current.lt(original).sum().item()),
        zero_saving_steps=zero_saving_steps,
        addback_steps=addback_steps,
    )
