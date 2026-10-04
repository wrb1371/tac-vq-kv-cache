"""Discrete rate-distortion allocation primitives.

This is a transparent reference implementation of the Lagrangian allocation
described by RDKV (arXiv:2605.08317).  It is intended for diagnostics and unit
tests; it is not a packed deployment kernel and makes no speed claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class AllocationResult:
    bits: torch.Tensor
    mean_bits: float
    total_bits: int
    objective: float
    lagrange_multiplier: float
    target_mean_bits: float
    relative_budget_error: float


@dataclass(frozen=True)
class ActionAllocationResult:
    """Allocation over an arbitrary per-unit rate--distortion action table."""

    actions: torch.Tensor
    selected_rates: torch.Tensor
    total_rate: int
    budget: int
    objective: float
    lagrange_multiplier: float
    repair_upgrades: int


@dataclass(frozen=True)
class AnchorTrustRegionResult:
    """Projection of a three-exit allocation toward its middle-rate anchor."""

    actions: torch.Tensor
    max_non_anchor_fraction: float
    initial_non_anchor_units: int
    final_non_anchor_units: int
    reverted_low_high_pairs: int
    total_rate: int
    initial_objective: float
    final_objective: float

    def summary_dict(self) -> dict[str, int | float]:
        return {
            "max_non_anchor_fraction": self.max_non_anchor_fraction,
            "initial_non_anchor_units": self.initial_non_anchor_units,
            "final_non_anchor_units": self.final_non_anchor_units,
            "reverted_low_high_pairs": self.reverted_low_high_pairs,
            "total_rate": self.total_rate,
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "objective_increase": self.final_objective - self.initial_objective,
        }


def project_allocation_to_anchor_trust_region(
    distortions: torch.Tensor,
    actions: torch.Tensor,
    rates: torch.Tensor,
    *,
    max_non_anchor_fraction: float,
) -> AnchorTrustRegionResult:
    """Limit how many units leave the middle exit without changing total rate.

    The current nested-VQ schedule has three symmetric payload rates.  Moving
    one low-rate and one high-rate unit back to the middle exit preserves the
    exact payload budget.  For a requested cardinality cap this routine picks
    the low/high reversion pair with the smallest measured RD penalty; because
    the two sides are independent, sorting each side is optimal for a fixed
    number of reverted pairs.

    This is a trust region around uniform precision, not a new rate estimate.
    Fixed labels and packed metadata remain the responsibility of the caller.
    """

    if distortions.ndim < 2 or distortions.shape[-1] != 3:
        raise ValueError("trust-region projection requires exactly three actions")
    if actions.shape != distortions.shape[:-1]:
        raise ValueError("actions do not match the distortion units")
    if rates.ndim != 1 or rates.shape[0] != 3:
        raise ValueError("trust-region rates must contain three actions")
    if not 0.0 <= max_non_anchor_fraction <= 1.0:
        raise ValueError("trust-region fraction must lie in [0, 1]")
    if not torch.isfinite(distortions).all() or (distortions < 0).any():
        raise ValueError("distortions must be finite and non-negative")
    if not torch.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("rates must be finite and non-negative")
    local_rates = rates.to(device=distortions.device, dtype=torch.int64)
    if not (
        local_rates[1] - local_rates[0] == local_rates[2] - local_rates[1]
        and local_rates[0] < local_rates[1] < local_rates[2]
    ):
        raise ValueError("three trust-region rates must be strictly symmetric")

    flat_d = distortions.detach().to(torch.float64).reshape(-1, 3)
    flat_actions = actions.detach().to(device=flat_d.device, dtype=torch.int64).reshape(-1)
    if ((flat_actions < 0) | (flat_actions > 2)).any():
        raise ValueError("trust-region actions must lie in {0,1,2}")
    rows = torch.arange(flat_actions.numel(), device=flat_d.device)
    initial_non_anchor = int(flat_actions.ne(1).sum())
    maximum = int(max_non_anchor_fraction * flat_actions.numel())
    low_indices = flat_actions.eq(0).nonzero(as_tuple=False).flatten()
    high_indices = flat_actions.eq(2).nonzero(as_tuple=False).flatten()
    minimum_reachable = abs(int(low_indices.numel()) - int(high_indices.numel()))
    if maximum < minimum_reachable:
        raise ValueError(
            "trust-region cap is infeasible at the allocation's fixed total rate"
        )

    pairs = max(0, (initial_non_anchor - maximum + 1) // 2)
    pairs = min(pairs, int(low_indices.numel()), int(high_indices.numel()))
    projected = flat_actions.clone()
    if pairs:
        low_penalty = flat_d[low_indices, 1] - flat_d[low_indices, 0]
        high_penalty = flat_d[high_indices, 1] - flat_d[high_indices, 2]
        reverted_low = low_indices[
            torch.argsort(low_penalty, stable=True)[:pairs]
        ]
        reverted_high = high_indices[
            torch.argsort(high_penalty, stable=True)[:pairs]
        ]
        projected[reverted_low] = 1
        projected[reverted_high] = 1

    initial_rate = int(local_rates[flat_actions].sum())
    final_rate = int(local_rates[projected].sum())
    if initial_rate != final_rate:
        raise RuntimeError("anchor trust-region projection changed payload rate")
    initial_objective = float(flat_d[rows, flat_actions].sum())
    final_objective = float(flat_d[rows, projected].sum())
    return AnchorTrustRegionResult(
        actions=projected.reshape(actions.shape),
        max_non_anchor_fraction=float(max_non_anchor_fraction),
        initial_non_anchor_units=initial_non_anchor,
        final_non_anchor_units=int(projected.ne(1).sum()),
        reverted_low_high_pairs=pairs,
        total_rate=final_rate,
        initial_objective=initial_objective,
        final_objective=final_objective,
    )


def stabilize_grouped_eviction_distortions(
    distortions: torch.Tensor,
    *,
    worst_member_weight: float = 0.0,
    eviction_penalty: float = 1.0,
) -> torch.Tensor:
    """Make group eviction conservative without changing positive exits.

    ``distortions`` has ``[..., members, actions]`` layout.  The grouped
    allocator normally sums action-zero distortion across members, allowing a
    specialized head to be outvoted by many insensitive heads.  This helper
    interpolates every member's zero-action cost toward the worst member, so
    the group sum becomes ``(1-w) * sum + w * H * max``.  A separate penalty
    supports validation-time calibration of systematic eviction optimism.
    """

    if distortions.ndim < 3 or distortions.shape[-1] < 2:
        raise ValueError("distortions must have groups, members, and actions")
    if not 0.0 <= worst_member_weight <= 1.0:
        raise ValueError("worst_member_weight must lie in [0, 1]")
    if eviction_penalty <= 0.0:
        raise ValueError("eviction_penalty must be positive")
    if not torch.isfinite(distortions).all() or (distortions < 0).any():
        raise ValueError("distortions must be finite and non-negative")
    result = distortions.clone()
    zero = result[..., 0]
    worst = zero.max(dim=-1, keepdim=True).values
    result[..., 0] = (
        (1.0 - worst_member_weight) * zero
        + worst_member_weight * worst
    ) * eviction_penalty
    return result


def select_conservative_eviction_candidates(
    distortions: torch.Tensor,
    *,
    max_fraction: float = 1.0,
    sink_tokens: int = 0,
    recent_tokens: int = 0,
    protect_layer_zero: bool = True,
) -> torch.Tensor:
    """Return a layer/sample/token mask of groups allowed to use action zero.

    The expected layout is ``[L,B,T,H,A]``.  Sink and recent anchors are
    excluded first.  If ``max_fraction < 1``, each layer/sample may expose only
    the lowest zero-action-distortion groups to the allocator.  This is an
    eligibility cap, not a forced eviction fraction.
    """

    if distortions.ndim != 5 or distortions.shape[-1] < 2:
        raise ValueError("distortions must have [L,B,T,H,A] layout")
    if not 0.0 <= max_fraction <= 1.0:
        raise ValueError("max_fraction must lie in [0, 1]")
    if sink_tokens < 0 or recent_tokens < 0:
        raise ValueError("anchor token counts must be non-negative")
    layers, batch, tokens, _heads, _actions = distortions.shape
    allowed = torch.ones(
        (layers, batch, tokens), dtype=torch.bool, device=distortions.device
    )
    if protect_layer_zero:
        allowed[0] = False
    allowed[:, :, : min(tokens, sink_tokens)] = False
    if recent_tokens:
        allowed[:, :, max(0, tokens - recent_tokens) :] = False
    if max_fraction >= 1.0:
        return allowed

    zero_cost = distortions[..., 0].sum(dim=-1)
    capped = torch.zeros_like(allowed)
    maximum = int(tokens * max_fraction)
    if maximum == 0:
        return capped
    for layer in range(layers):
        for sample in range(batch):
            eligible = allowed[layer, sample].nonzero(as_tuple=False).flatten()
            count = min(maximum, int(eligible.numel()))
            if count:
                local_cost = zero_cost[layer, sample].index_select(0, eligible)
                selected = eligible[local_cost.topk(count, largest=False).indices]
                capped[layer, sample, selected] = True
    return capped


def _validate_table(bitwidths: Sequence[int], distortion: Mapping[int, float]) -> None:
    if tuple(sorted(bitwidths)) != tuple(bitwidths):
        raise ValueError("bitwidths must be sorted")
    if bitwidths[0] != 0:
        raise ValueError("bitwidths must include zero-rate eviction")
    if any(bit not in distortion for bit in bitwidths):
        raise ValueError("distortion table must define every bitwidth")
    errors = [float(distortion[bit]) for bit in bitwidths]
    if any(value < 0 for value in errors):
        raise ValueError("distortion values must be non-negative")
    if any(left < right for left, right in zip(errors, errors[1:])):
        raise ValueError("distortion must not increase with bitwidth")


def allocate_lagrangian(
    weights: torch.Tensor,
    *,
    target_mean_bits: float,
    distortion: Mapping[int, float],
    bitwidths: Sequence[int] = (0, 2, 4, 8, 16),
    tolerance: float = 1e-2,
    max_iterations: int = 64,
) -> AllocationResult:
    """Allocate one discrete bitwidth per unit using RDKV-style bisection.

    The result is always the best feasible bisection iterate observed (mean
    rate no larger than the target).  This is stricter than returning the last
    iterate, which can silently exceed the requested storage budget when the
    discrete rate has a jump around the water level.
    """

    _validate_table(bitwidths, distortion)
    if weights.numel() == 0:
        raise ValueError("weights cannot be empty")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and non-negative")
    if not 0 <= target_mean_bits <= max(bitwidths):
        raise ValueError("target_mean_bits is outside the action range")
    if tolerance <= 0 or max_iterations <= 0:
        raise ValueError("tolerance and max_iterations must be positive")

    original_shape = weights.shape
    flat_weights = weights.detach().to(dtype=torch.float64).reshape(-1)
    device = flat_weights.device
    bit_tensor = torch.tensor(bitwidths, dtype=torch.float64, device=device)
    error_tensor = torch.tensor(
        [float(distortion[bit]) for bit in bitwidths],
        dtype=torch.float64,
        device=device,
    )

    def choose(lam: float) -> tuple[torch.Tensor, float, float]:
        costs = flat_weights[:, None] * error_tensor[None, :] + lam * bit_tensor[None, :]
        indices = costs.argmin(dim=1)
        chosen = bit_tensor[indices]
        objective = float((flat_weights * error_tensor[indices]).sum().item())
        return chosen, float(chosen.mean().item()), objective

    if target_mean_bits == max(bitwidths):
        chosen, mean_bits, objective = choose(0.0)
        lam = 0.0
    elif target_mean_bits == 0:
        chosen = torch.zeros_like(flat_weights)
        mean_bits = 0.0
        objective = float((flat_weights * float(distortion[0])).sum().item())
        lam = float(flat_weights.max().item() + 1.0)
    else:
        low = 0.0
        high = max(1.0, float(flat_weights.max().item()))
        while choose(high)[1] > target_mean_bits:
            high *= 2.0

        best: tuple[torch.Tensor, float, float, float] | None = None
        for _ in range(max_iterations):
            lam = (low + high) / 2.0
            candidate, mean_bits, objective = choose(lam)
            if mean_bits <= target_mean_bits:
                if best is None or mean_bits > best[1] or (
                    mean_bits == best[1] and objective < best[2]
                ):
                    best = (candidate.clone(), mean_bits, objective, lam)
                high = lam
            else:
                low = lam
            if mean_bits > 0 and abs(mean_bits - target_mean_bits) / target_mean_bits <= tolerance:
                if mean_bits <= target_mean_bits:
                    break
        if best is None:
            chosen = torch.zeros_like(flat_weights)
            mean_bits = 0.0
            objective = float((flat_weights * float(distortion[0])).sum().item())
            lam = high
        else:
            chosen, mean_bits, objective, lam = best

    relative_error = (
        0.0
        if target_mean_bits == 0
        else abs(mean_bits - target_mean_bits) / target_mean_bits
    )
    return AllocationResult(
        bits=chosen.to(dtype=torch.int16).reshape(original_shape),
        mean_bits=mean_bits,
        total_bits=int(chosen.sum().item()),
        objective=objective,
        lagrange_multiplier=float(lam),
        target_mean_bits=float(target_mean_bits),
        relative_budget_error=float(relative_error),
    )


def allocate_action_table(
    distortions: torch.Tensor,
    rates: torch.Tensor,
    *,
    budget: int,
    max_iterations: int = 80,
    greedy_repair: bool = True,
) -> ActionAllocationResult:
    """Allocate arbitrary discrete actions under one integer storage budget.

    ``distortions`` has shape ``[..., actions]`` and may contain a different
    measured distortion curve for every KV unit. ``rates`` is either a shared
    one-dimensional action-rate vector or a tensor with the same shape.  This
    is the natural reference allocator for progressive VQ: action zero is
    eviction, while later actions retain the unit with progressively longer
    index prefixes.  Rates should include persistent metadata (for example a
    retained-token position) instead of only the code payload.

    Lagrangian bisection supplies a feasible solution.  A deterministic
    marginal-gain repair then spends otherwise stranded budget on individual
    upgrades.  The repair is useful at very low rates, where a global water
    level can jump across many equal-cost actions at once.  This remains a
    transparent MCKP approximation, not a claim of exact integer optimality.
    """

    if distortions.ndim < 2 or distortions.shape[-1] < 2:
        raise ValueError("distortions must have units plus at least two actions")
    if not torch.isfinite(distortions).all() or (distortions < 0).any():
        raise ValueError("distortions must be finite and non-negative")
    if budget < 0 or max_iterations <= 0:
        raise ValueError("budget must be non-negative and iterations positive")

    if rates.ndim == 1:
        if rates.shape[0] != distortions.shape[-1]:
            raise ValueError("shared rates must have one entry per action")
        expanded_rates = rates.reshape(
            *((1,) * (distortions.ndim - 1)), rates.shape[0]
        ).expand_as(distortions)
    elif rates.shape == distortions.shape:
        expanded_rates = rates
    else:
        raise ValueError("rates must be [actions] or match distortions")
    if not torch.isfinite(expanded_rates).all() or (expanded_rates < 0).any():
        raise ValueError("rates must be finite and non-negative")
    if not torch.all(expanded_rates[..., 1:] >= expanded_rates[..., :-1]):
        raise ValueError("action rates must be non-decreasing")

    original_shape = distortions.shape[:-1]
    action_count = distortions.shape[-1]
    flat_d = distortions.detach().to(torch.float64).reshape(-1, action_count)
    flat_r = expanded_rates.detach().to(torch.float64).reshape(-1, action_count)
    if (flat_r - flat_r.round()).abs().max().item() > 1e-8:
        raise ValueError("rates and budget must use the same integer unit")
    flat_r = flat_r.round()

    def choose(lam: float) -> tuple[torch.Tensor, int, float]:
        actions = (flat_d + lam * flat_r).argmin(dim=-1)
        rows = torch.arange(flat_d.shape[0], device=flat_d.device)
        total = int(flat_r[rows, actions].sum().item())
        objective = float(flat_d[rows, actions].sum().item())
        return actions, total, objective

    minimum_actions = flat_r.argmin(dim=-1)
    rows = torch.arange(flat_d.shape[0], device=flat_d.device)
    minimum_rate = int(flat_r[rows, minimum_actions].sum().item())
    if minimum_rate > budget:
        raise ValueError("budget is below the minimum representable action rate")

    unconstrained, unconstrained_rate, unconstrained_objective = choose(0.0)
    if unconstrained_rate <= budget:
        selected = unconstrained
        total_rate = unconstrained_rate
        objective = unconstrained_objective
        lam = 0.0
    else:
        low, high = 0.0, 1.0
        while choose(high)[1] > budget:
            high *= 2.0
        best: tuple[torch.Tensor, int, float, float] | None = None
        for _ in range(max_iterations):
            lam = (low + high) / 2.0
            candidate, candidate_rate, candidate_objective = choose(lam)
            if candidate_rate <= budget:
                if best is None or candidate_objective < best[2] or (
                    candidate_objective == best[2] and candidate_rate > best[1]
                ):
                    best = (
                        candidate.clone(),
                        candidate_rate,
                        candidate_objective,
                        lam,
                    )
                high = lam
            else:
                low = lam
        if best is None:
            selected = minimum_actions.clone()
            total_rate = minimum_rate
            objective = float(flat_d[rows, selected].sum().item())
            lam = high
        else:
            selected, total_rate, objective, lam = best

    upgrades = 0
    if greedy_repair and total_rate < budget:
        while True:
            current_d = flat_d[rows, selected]
            current_r = flat_r[rows, selected]
            delta_r = flat_r - current_r[:, None]
            gain = current_d[:, None] - flat_d
            feasible = (delta_r > 0) & (delta_r <= budget - total_rate) & (gain > 0)
            if not feasible.any():
                break
            efficiency = torch.full_like(gain, -torch.inf)
            efficiency[feasible] = gain[feasible] / delta_r[feasible]
            flat_choice = int(efficiency.reshape(-1).argmax().item())
            unit = flat_choice // action_count
            action = flat_choice % action_count
            rate_increment = int(delta_r[unit, action].item())
            selected[unit] = action
            total_rate += rate_increment
            upgrades += 1
        objective = float(flat_d[rows, selected].sum().item())

    selected_rates = flat_r[rows, selected].to(torch.int64).reshape(original_shape)
    return ActionAllocationResult(
        actions=selected.to(torch.int64).reshape(original_shape),
        selected_rates=selected_rates,
        total_rate=int(total_rate),
        budget=int(budget),
        objective=float(objective),
        lagrange_multiplier=float(lam),
        repair_upgrades=upgrades,
    )


def allocate_grouped_eviction_action_table(
    distortions: torch.Tensor,
    rates: torch.Tensor,
    *,
    budget: int,
    max_iterations: int = 80,
    eviction_allowed: torch.Tensor | None = None,
    action_allowed: torch.Tensor | None = None,
) -> ActionAllocationResult:
    """Allocate group eviction plus member-wise positive-rate precision exits.

    ``distortions`` uses ``[..., members, actions]`` layout. Action zero is a
    shared structural decision: either every member of a group is evicted, or
    every member independently chooses among actions 1..A-1. For a KV cache a
    group is one layer/token and its members are the KV heads. This produces
    rectangular per-layer tensors while retaining per-head precision choices.

    At a fixed Lagrange multiplier the problem is exactly separable: compare
    summed eviction distortion with the sum of each member's best positive
    action. Fixed depth-label costs are excluded from ``rates`` and must be
    removed from ``budget`` by the caller.
    """

    if distortions.ndim < 3 or distortions.shape[-1] < 2:
        raise ValueError("distortions must have groups, members, and actions")
    if rates.ndim != 1 or rates.shape[0] != distortions.shape[-1]:
        raise ValueError("rates must have one shared value per action")
    if not torch.isfinite(distortions).all() or (distortions < 0).any():
        raise ValueError("distortions must be finite and non-negative")
    if not torch.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("rates must be finite and non-negative")
    if rates[0].item() != 0 or not torch.all(rates[1:] >= rates[:-1]):
        raise ValueError("action zero must be zero-rate and rates non-decreasing")
    if (rates - rates.round()).abs().max().item() > 1e-8:
        raise ValueError("rates and budget must use the same integer unit")
    if budget < 0 or max_iterations <= 0:
        raise ValueError("budget and iteration count are invalid")

    original_shape = distortions.shape[:-1]
    members = distortions.shape[-2]
    action_count = distortions.shape[-1]
    flat_d = distortions.detach().to(torch.float64).reshape(
        -1, members, action_count
    )
    local_rates = rates.detach().to(
        device=flat_d.device, dtype=torch.float64
    ).round()
    group_rows = torch.arange(flat_d.shape[0], device=flat_d.device)[:, None]
    member_rows = torch.arange(members, device=flat_d.device)[None, :]
    group_shape = distortions.shape[:-2]
    if eviction_allowed is None:
        flat_eviction_allowed = torch.ones(
            flat_d.shape[0], dtype=torch.bool, device=flat_d.device
        )
    else:
        if eviction_allowed.shape != group_shape:
            raise ValueError("eviction_allowed must have one value per group")
        flat_eviction_allowed = eviction_allowed.to(
            device=flat_d.device, dtype=torch.bool
        ).reshape(-1)
    if action_allowed is None:
        flat_action_allowed = torch.ones_like(flat_d, dtype=torch.bool)
    else:
        if action_allowed.shape != distortions.shape:
            raise ValueError("action_allowed must match the distortion table")
        flat_action_allowed = action_allowed.to(
            device=flat_d.device, dtype=torch.bool
        ).reshape_as(flat_d)
    positive_allowed = flat_action_allowed[..., 1:]
    positive_rate_table = local_rates[None, None, 1:].expand_as(
        positive_allowed
    )
    minimum_member_rates = positive_rate_table.masked_fill(
        ~positive_allowed, torch.inf
    ).min(dim=-1).values
    if torch.isinf(minimum_member_rates[~flat_eviction_allowed]).any():
        raise ValueError("a non-evictable group member has no positive action")
    minimum_rate = int(
        minimum_member_rates[~flat_eviction_allowed].sum().item()
    )
    if minimum_rate > budget:
        raise ValueError("budget cannot retain every non-evictable group")

    def choose(lam: float) -> tuple[torch.Tensor, int, float]:
        positive_cost = flat_d[..., 1:] + lam * local_rates[None, None, 1:]
        positive_cost = positive_cost.masked_fill(~positive_allowed, torch.inf)
        positive_actions = positive_cost.argmin(dim=-1) + 1
        kept_cost = positive_cost.gather(
            -1, (positive_actions - 1).unsqueeze(-1)
        ).squeeze(-1).sum(dim=-1)
        evicted_cost = flat_d[..., 0].sum(dim=-1)
        evicted = (evicted_cost <= kept_cost) & flat_eviction_allowed
        selected = positive_actions.masked_fill(evicted[:, None], 0)
        total_rate = int(local_rates[selected].sum().item())
        objective = float(
            flat_d[group_rows, member_rows, selected].sum().item()
        )
        return selected, total_rate, objective

    unconstrained, unconstrained_rate, unconstrained_objective = choose(0.0)
    if unconstrained_rate <= budget:
        selected = unconstrained
        total_rate = unconstrained_rate
        objective = unconstrained_objective
        lam = 0.0
    else:
        low, high = 0.0, 1.0
        while choose(high)[1] > budget:
            high *= 2.0
        best: tuple[torch.Tensor, int, float, float] | None = None
        for _ in range(max_iterations):
            lam = (low + high) / 2.0
            candidate, candidate_rate, candidate_objective = choose(lam)
            if candidate_rate <= budget:
                if best is None or candidate_objective < best[2] or (
                    candidate_objective == best[2] and candidate_rate > best[1]
                ):
                    best = (
                        candidate.clone(),
                        candidate_rate,
                        candidate_objective,
                        lam,
                    )
                high = lam
            else:
                low = lam
        if best is None:
            selected = torch.zeros(
                (flat_d.shape[0], members),
                dtype=torch.long,
                device=flat_d.device,
            )
            total_rate = 0
            objective = float(flat_d[..., 0].sum().item())
            lam = high
        else:
            selected, total_rate, objective, lam = best

    selected_rates = local_rates[selected].to(torch.int64).reshape(original_shape)
    return ActionAllocationResult(
        actions=selected.to(torch.int64).reshape(original_shape),
        selected_rates=selected_rates,
        total_rate=int(total_rate),
        budget=int(budget),
        objective=float(objective),
        lagrange_multiplier=float(lam),
        repair_upgrades=0,
    )


RDKV_DISTORTION_TABLES = {
    "llama-3.1-8b": {
        "key": {0: 1.0, 2: 0.149, 4: 0.0062, 8: 2.2e-5, 16: 0.0},
        "value": {0: 1.0, 2: 0.313, 4: 0.0140, 8: 4.9e-5, 16: 0.0},
    },
    "mistral-7b": {
        "key": {0: 1.0, 2: 0.280, 4: 0.0116, 8: 6.9e-5, 16: 0.0},
        "value": {0: 1.0, 2: 0.296, 4: 0.0130, 8: 4.5e-5, 16: 0.0},
    },
}
