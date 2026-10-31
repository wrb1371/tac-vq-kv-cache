"""Matched structural controls; not new deployable algorithms."""
import torch
from .stable import StablePacked
from .placebo import ShiftedNeighborhood
from ar18kv.hybrid import DenseFirst
class SamePosition:
    def _reference_indices(self,ordinals,codes):
        return super()._reference_indices(ordinals,torch.full_like(codes,4))
class SameStable(SamePosition,StablePacked):pass
class ShiftedStable(ShiftedNeighborhood,StablePacked):pass
class HybridSameStable(DenseFirst,SameStable):pass
class HybridShiftedStable(DenseFirst,ShiftedStable):pass
