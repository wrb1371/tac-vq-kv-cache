"""Locally fitted additive centroids; two16-entry tables, one uint8 joint index."""
from ar17kv.factorized import CompactPrivate
class PrivateAdditive(CompactPrivate):
    method='PrivateAQ-independent-local-additive-centroids'
    needs_pre_rope_K=True
    def book(self,l,kind,stage):
        t=self.tables[l];a=t[kind+'_A'][:,stage].float();b=t[kind+'_B'][:,stage].float()
        # Effective256-centroid table is per-operation scratch, never persistent.
        return (a[:,:,None,:]+b[:,None,:,:]).reshape(2,256,128)
