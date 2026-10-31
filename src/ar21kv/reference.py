"""Original Torch execution under exactly the same independently packed rates."""
from ar19kv.packed_sparse import HybridPackedSparse
from .allocated import PhysicalAllocation
from .benefit import GainSelection
class ReferenceAllocated(PhysicalAllocation,HybridPackedSparse):pass
class ReferenceGain(GainSelection,ReferenceAllocated):pass
