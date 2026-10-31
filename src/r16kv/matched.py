"""Matched-plan diagnostic only: use parent's planning fee, pay actual compact fee."""
from .nova import NOVAAllocated
class NOVAForcedPlan(NOVAAllocated):
    def make_plan(self,meta):
        actual=self.static_bytes;planner=int(self.policy['matched_plan_codec_bytes']);self.static_bytes=planner
        try:plan=super().make_plan(meta)
        finally:self.static_bytes=actual
        plan['estimate']['codec_static_bytes']=actual;plan['estimate']['resident_capacity_bytes']-=planner-actual
        plan['matched_layout_reserve_not_GPU_padding']=planner-actual
        return plan
