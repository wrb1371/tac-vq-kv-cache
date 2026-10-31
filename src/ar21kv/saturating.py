"""Geometry-only innovation saturation after all parent depths reach maximum."""
import math
from .dynamic import DynamicExactAllocated

class SaturatingInnovations(DynamicExactAllocated):
    def make_plan(self,meta):
        base=super().make_plan(meta)
        if self._hybrid_dense:return base
        maximum=self.package['Rmax_K'];assert maximum==self.package['Rmax_V']==6
        if any(r!=maximum for k in ['layer_R_K','layer_R_V'] for r in base[k]):
            base['innovation_saturation_applied']=False;return base
        targets=base['media_original']-base['anchor_tokens']
        if targets<=base['innovation_slots']:
            base['innovation_saturation_applied']=False;return base
        # Same arena schema, N/P/maps/anchors and all depths held fixed.
        # Each extra innovation slot per head costs 36*(2*g*(RK+RV)+8).
        unit=36*(2*base['groups']*(maximum+maximum)+8)
        left=self.budget_bytes-base['estimate']['resident_capacity_bytes']
        extra=min(targets-base['innovation_slots'],left//unit)
        if extra<=0:
            base['innovation_saturation_applied']=False;return base
        wanted=base['innovation_slots']+extra;old=self.policy['keep_fraction']
        # Midpoint implements the desired ceil count without boundary ambiguity.
        self.policy['keep_fraction']=(wanted-.5)/targets
        try:plan=super().make_plan(meta)
        finally:self.policy['keep_fraction']=old
        assert plan['innovation_slots']==wanted
        assert all(r==maximum for k in ['layer_R_K','layer_R_V'] for r in plan[k])
        assert plan['estimate']['resident_capacity_bytes']-base['estimate']['resident_capacity_bytes']==unit*extra
        assert plan['estimate']['resident_capacity_bytes']<=self.budget_bytes
        plan.update(innovation_saturation_applied=True,parent_innovation_slots=base['innovation_slots'],extra_innovation_slots=extra,innovation_slot_cost_bytes=unit,effective_innovation_fraction=wanted/targets,parent_paid_capacity_bytes=base['estimate']['resident_capacity_bytes'],saturation_rule='increase innovations only after every K/V depth reaches Rmax; all parent depths and references preserved')
        return plan
