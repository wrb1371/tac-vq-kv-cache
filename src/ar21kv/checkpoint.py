"""Evaluation-only prefix key: representation retained, future reserve costs omitted."""
from ar19kv.checkpoint import PrefixCheckpoint as Original
class PrefixCheckpoint(Original):
    def key(self,cache,inputs,positions,split,imeta):
        if not hasattr(cache,'codec'):return None
        plan=cache.codec.plan
        ignored={'allocation_stage_unit_bytes','allocation_stage_unit_cap','parent_resident_estimate_bytes','reclaimed_unallocated_bytes','unspent_physical_budget_bytes'}
        # Actual layer_R_K/V, correspondence, counts, bank and all media input
        # identities remain in the key. CPU checkpoint is never a cold result.
        cache.codec.plan={k:v for k,v in plan.items() if k not in ignored}
        try:return super().key(cache,inputs,positions,split,imeta)
        finally:cache.codec.plan=plan
