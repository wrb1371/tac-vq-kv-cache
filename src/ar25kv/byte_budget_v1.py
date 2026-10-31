"""Exact integer-byte caps for fractional-MiB operating points; old formats intact."""
from ar22kv.joint import JointTemporal
from ar24kv.attribution import CheckedDecode
from ar24kv.risk_selective_v1 import SelectiveJoint
from ar24kv.pressure_switch_v1 import PressureSwitch

class ExactByteBudget:
 def __init__(self,package,budget_mib,**policy):
  scaled=float(budget_mib)*2**20;requested=int(round(scaled))
  assert requested>0 and abs(scaled-requested)<1e-6,'BUDGET_MUST_BE_INTEGER_BYTES'
  super().__init__(package,budget_mib,**policy)
  # The legacy loader's initial integer-MiB cap already admits this small fixed
  # codec. All subsequent planner/arena/registry admission uses the exact cap.
  assert self.static_bytes<=requested
  self.budget_bytes=requested;self.load_registry.budget=requested
  self.metadata.update(exact_requested_budget_bytes=requested,budget_input_unit='MiB representation of exact integer bytes',legacy_integer_MiB_truncation_bypassed=True)

class ByteGuard(ExactByteBudget,CheckedDecode,JointTemporal):pass
class ByteMin2(ExactByteBudget,SelectiveJoint):pass
class ByteSwitch(ExactByteBudget,PressureSwitch):pass
