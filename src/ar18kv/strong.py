"""Matched paid-prefix strong controls; not new self-developed quantizers."""
from ar18kv.hybrid import DenseFirst
from ar17kv.text_protection import PaidPrefixText,PrivateTextProtected
from r16kv.nsn64 import NSN64Reference
class HybridPrivateProtected(DenseFirst,PrivateTextProtected):
    needs_pre_rope_K=True
class HybridNSNProtected(DenseFirst,PaidPrefixText,NSN64Reference):
    needs_pre_rope_K=True
