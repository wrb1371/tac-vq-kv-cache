"""Common paid-prefix policy on the existing NOVA family; no new codec claim."""
from ar18kv.hybrid import DenseFirst
from ar17kv.text_protection import PaidPrefixText
from r15kv.codecs import NOVABudget
from r16kv.nova import NOVAAllocated
class StrongNOVA(DenseFirst,PaidPrefixText,NOVABudget):pass
class StrongNplus(DenseFirst,PaidPrefixText,NOVAAllocated):pass
