"""Reference-only interventions for a configuration frozen before confirmation."""
from .controls import SamePosition
from .placebo import ShiftedNeighborhood
from .stable import HybridStablePacked
from .packed_sparse import HybridPackedSparse
from .sparse import HybridSparse
from .rate_gate import HybridRateGated
from .refinement import HybridRateMatched,HybridInnovationRefined
# Every class is a normal auditable MRO of the selected representation; no QA routing.
for _base in [HybridStablePacked,HybridPackedSparse,HybridSparse,HybridRateGated,HybridRateMatched,HybridInnovationRefined]:
    for _label,_mixin in [('Same',SamePosition),('Shift',ShiftedNeighborhood)]:
        _name=_label+'_'+_base.__name__
        globals()[_name]=type(_name,(_mixin,_base),{'__module__':__name__})
