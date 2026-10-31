"""Raw NSN with exactly the same paid-prefix module as every matched control."""
from ar17kv.text_protection import PaidPrefixText
from r16kv.nsn64 import NSN64Reference
class NSNProtected(PaidPrefixText,NSN64Reference):
    needs_pre_rope_K=True
