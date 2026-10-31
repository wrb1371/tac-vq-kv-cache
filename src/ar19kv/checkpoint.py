"""Evaluation-only compressed checkpoints, invariant to unused query reserve."""
from ar18kv.evaluation_checkpoint import PrefixCheckpoint as Original
class PrefixCheckpoint(Original):
    def key(self,cache,inputs,positions,split,imeta):
        if not hasattr(cache,'codec'):return None
        plan=cache.codec.plan
        # A fully retained BF16 pruning prefix is admitted but dense. Do not
        # checkpoint it; this preserves the precise compressed-only contract.
        if plan.get('media_kept_per_head')==plan.get('media_original') and 'media_kept_per_head' in plan:return None
        # These only reserve future query slots; media rate, selected count,
        # anchors, correspondence policy and all semantic fields stay in key.
        cache.codec.plan={k:v for k,v in plan.items() if k not in ['query_slots','encoded_slots_per_head','reserved_query_slots','temporal_metadata_bytes']}
        try:return super().key(cache,inputs,positions,split,imeta)
        finally:cache.codec.plan=plan
