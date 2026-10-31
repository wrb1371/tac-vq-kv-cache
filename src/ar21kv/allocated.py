"""Reclaim physical map-free capacity with independently packed layer depths."""
import hashlib,json,pathlib
import torch
from r14kv.common import ROOT
from r15kv.persistent_registry import Arena
from ar20kv.exact_gemm import ExactMatchStable,ExactMatchPacked
from .planner import allocate

class PhysicalAllocation:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        p=ROOT/self.policy['curve_file'];self._curve_SHA256=hashlib.sha256(p.read_bytes()).hexdigest();self._curves=json.loads(p.read_text());assert self._curves['completed']
        assert self._curves['fixed_bank_completion_SHA256']==self.package_sha256
    def make_plan(self,meta):
        parent=super().make_plan(meta)
        if self._hybrid_dense:return parent
        desc=parent['description'];e=parent['encoded_slots_per_head'];unit=2*e*parent['groups'];assert unit>0
        fixed=self.static_bytes+parent['estimate']['global_request_bytes']
        def description(k,v):return [(name,(2,e,k if name=='K_index' else v,parent['groups']) if name in ['K_index','V_index'] else shape,dtype,kind) for name,shape,dtype,kind in desc]
        constant=fixed+36*Arena.bytes(description(1,1))-72*unit
        capacity=(self.budget_bytes-constant)//unit
        if capacity<72:raise RuntimeError('INFEASIBLE_BY_INTEGER_ORACLE_HETEROGENEOUS_RVQ')
        curves=self._curves[self.policy.get('curve_kind','plain')];objective=[];lower=[]
        for l in range(36):
            for kind in ['K','V']:
                objective.append(curves[kind][l]);lower.append(parent['R_'+kind] if self.policy.get('refine_only',False) else 1)
        rates,value=allocate(objective,capacity,lower,mode=self.policy.get('allocator_mode','dp'))
        descriptions=[description(rates[2*l],rates[2*l+1]) for l in range(36)];total=fixed+sum(Arena.bytes(d) for d in descriptions);assert total==constant+unit*sum(rates)<=self.budget_bytes
        old_total=parent['estimate']['resident_capacity_bytes'];parent['description']=descriptions[0];parent['descriptions']=descriptions
        parent['estimate']={**parent['estimate'],'resident_capacity_bytes':total,'layer_arena_bytes':Arena.bytes(descriptions[0]),'all_layer_arena_bytes':sum(Arena.bytes(d) for d in descriptions)}
        parent.update(layer_R_K=rates[::2],layer_R_V=rates[1::2],allocator='exact discrete DP over calibrated depth costs; measured arenas certify admission',allocation_mode=self.policy.get('allocator_mode','dp'),allocation_curve_kind=self.policy.get('curve_kind','plain'),allocation_curve_SHA256=self._curve_SHA256,allocation_predicted_objective=value,allocation_stage_unit_bytes=unit,allocation_stage_units=sum(rates),allocation_stage_unit_cap=min(432,capacity),parent_resident_estimate_bytes=old_total,reclaimed_unallocated_bytes=total-old_total,unspent_physical_budget_bytes=self.budget_bytes-total,legacy_uniform_R_K=parent['R_K'],legacy_uniform_R_V=parent['R_V'])
        return parent
    def allocate_state(self,l,registry):
        if self._hybrid_dense:return super().allocate_state(l,registry)
        arena=Arena(self.plan['descriptions'][l],'cuda',registry,f'cache/layer{l:02}/arena');state={'arena':arena,'n':0,'text_filled':0,**arena.views};state['T_code'].fill_(255);return state

class AllocatedStable(PhysicalAllocation,ExactMatchStable):pass
class AllocatedPacked(PhysicalAllocation,ExactMatchPacked):pass
