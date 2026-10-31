"""Budget-gated precision envelopes; frozen original planner at normal pressure."""
import torch
from ar22kv.joint import JointTemporal
from .risk_selective_v1 import SelectiveJoint


class PressureSwitch(SelectiveJoint):
    def make_plan(self,meta):
        if self._hybrid_dense:
            return JointTemporal.make_plan(self,meta)
        m=len(self._video_list);end=self._video_list[-1]+1
        q=meta.capacity-end;text=len(self._text_list)
        dtype=torch.int16 if max(meta.position_max)<=32767 else torch.int32
        global_bytes=3*meta.capacity*torch.empty((),dtype=dtype).element_size()+256+4*m+4*text
        fixed=self.static_bytes+global_bytes+36*(2*meta.capacity+1024*text)
        pressure=36864*(m+q)/max(1,self.budget_bytes-fixed)
        if pressure<=32.:
            plan=JointTemporal.make_plan(self,meta)
            branch='original_allcell_parent_guard'
        else:
            plan=super().make_plan(meta)
            branch='pressure_two_stage_envelope'
        plan.update(ar24_pressure_switch_ratio=pressure,
                    ar24_pressure_switch_branch=branch,
                    ar24_pressure_switch_threshold=32.)
        return plan
