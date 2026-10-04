"""Reference tools for strict-rate joint KV-cache compression research."""

from .budget import (
    CacheGeometry,
    StorageLedger,
    progressive_vq_action_ledger,
    uniform_kivi_ledger,
    variable_cardinality_vq_action_ledger,
)
from .rd_allocator import (
    ActionAllocationResult,
    AllocationResult,
    allocate_action_table,
    allocate_lagrangian,
)
from .packed_nested import (
    PackedLayoutLedger,
    PackedNestedKVCache,
    PackedProgressiveStream,
    estimate_paired_progressive_layout,
    estimate_layered_paired_progressive_layout,
    pack_fixed_width,
    pack_paired_progressive_indices,
    unpack_fixed_width,
    unpack_paired_progressive_indices,
)
from .physical_allocator import (
    PackedBudgetRepairResult,
    repair_depth_actions_to_packed_budget,
)

__all__ = [
    "ActionAllocationResult",
    "AllocationResult",
    "CacheGeometry",
    "PackedLayoutLedger",
    "PackedNestedKVCache",
    "PackedBudgetRepairResult",
    "PackedProgressiveStream",
    "StorageLedger",
    "allocate_action_table",
    "allocate_lagrangian",
    "estimate_paired_progressive_layout",
    "estimate_layered_paired_progressive_layout",
    "pack_fixed_width",
    "pack_paired_progressive_indices",
    "progressive_vq_action_ledger",
    "repair_depth_actions_to_packed_budget",
    "uniform_kivi_ledger",
    "unpack_fixed_width",
    "unpack_paired_progressive_indices",
    "variable_cardinality_vq_action_ledger",
]
