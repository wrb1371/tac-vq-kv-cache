"""Transfer the unchanged paid prefix-text intervention to the older full VQ."""
from ar17kv.text_protection import PaidPrefixText
from ar17kv.crvq_v2 import CRVQFlexible
class FullVectorTextProtected(PaidPrefixText,CRVQFlexible):
    method='CRVQ-existing-paid-prefix-text-transfer'
    needs_pre_rope_K=True
