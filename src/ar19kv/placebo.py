"""Same nine-candidate search cost, spatially wrong reference neighborhood."""
from .temporal import Temporal
from .packed_sparse import PackedSparse
from ar18kv.hybrid import DenseFirst

class ShiftedNeighborhood:
    def _reference_indices(self,ordinals,codes):
        p=self._h*self._w;gop=int(self.policy.get('gop',2));frame=ordinals//p;sp=ordinals%p
        yy=(sp//self._w+codes.long()//3-1).clamp(0,self._h-1);xx=(sp%self._w+codes.long()%3-1).clamp(0,self._w-1)
        xx=(xx+self._w//2+1)%self._w;reford=(frame//gop*gop)*p+yy*self._w+xx
        return self._vid_gpu.long()[reford]
class ShiftedTemporal(ShiftedNeighborhood,Temporal):pass
class HybridShiftedTemporal(DenseFirst,ShiftedTemporal):pass
class ShiftedPacked(ShiftedNeighborhood,PackedSparse):pass
class HybridShiftedPacked(DenseFirst,ShiftedPacked):pass
