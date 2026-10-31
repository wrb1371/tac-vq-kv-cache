"""Allocated quantizer with GEMM/search/update RVQ stages fused in one kernel."""
from ar20kv.codec import FusedRVQ
from .allocated import AllocatedPacked
from .benefit import GainPacked
class FusedAllocatedPacked(FusedRVQ,AllocatedPacked):pass
class FusedGainPacked(FusedRVQ,GainPacked):pass
