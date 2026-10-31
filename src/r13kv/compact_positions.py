"""Lossless per-modality K/V position sharing with bounded unsigned16 storage.

No token positions are approximated. Both arrays temporarily return to the
author's int32 writer contract before appending. After both K and V agree,
they own one table. Beyond65535 use one shared int32 table. Encoder copies
and assertions are real runtime work, not a free-throughput claim.
"""
import torch

def position_dtype(last_position):
    assert last_position>=0
    return torch.uint16 if last_position<=65535 else torch.int32

class CompactPositionMixin:
    def __init__(self,cfg,*,compact_positions=False,**kwargs):
        super().__init__(cfg,**kwargs);self.compact_positions=compact_positions
    def _pack_append(self,x,offset,layer_idx,which):
        if not self.compact_positions:return super()._pack_append(x,offset,layer_idx,which)
        old=self._packed[which].get(layer_idx)
        if old is not None:
            for g in old.groups.values():
                if g.positions.dtype!=torch.int32:g.positions=g.positions.to(torch.int32)
        super()._pack_append(x,offset,layer_idx,which)
        dtype=position_dtype(offset+x.shape[2]-1)
        for mod,g in self._packed[which][layer_idx].groups.items():
            if which=='v':
                key=self._packed['k'][layer_idx].groups[mod]
                if key.length==g.length and key.capacity==g.capacity:
                    assert torch.equal(key.positions[:g.length].long(),g.positions[:g.length].long()),'never share different original positions'
                    g.positions=key.positions
                    continue
            if g.positions.dtype!=dtype:g.positions=g.positions.to(dtype)
    def summary(self):
        result=super().summary();result.update(compact_positions=self.compact_positions,
            position_layout='one shared original-position table per K/V modality group; uint16<=65535 elseint32' if self.compact_positions else 'author',
            byte_claim_source='independent CUDA storage ledger, not the author live estimator which assumes int32 twice')
        return result


# Independent convenience class; the mixin also wraps PreRCache before its K writer.
from .codec import EvolutionCache
class CompactPositionCache(CompactPositionMixin,EvolutionCache):
    pass
