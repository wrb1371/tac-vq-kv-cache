"""Original Torch numeric operators under the exact saturation planner/layout."""
from .saturating import SaturatingInnovations
from ar19kv.packed_sparse import HybridPackedSparse
class SaturatingReference(SaturatingInnovations):
    append=HybridPackedSparse.append
    decode=HybridPackedSparse.decode
    _read_slots=HybridPackedSparse._read_slots
    _innovation=HybridPackedSparse._innovation
