"""Conservative reference rate; spend available bytes on innovations only."""
import torch
from .packed_sparse import PackedSparse
from ar18kv.hybrid import DenseFirst
from r15kv.persistent_registry import Arena
from r14kv.fitting import nearest

class RateMatchedPacked(PackedSparse):
    method='AR19-packed-conservative-mapped-rate'
    def make_plan(self,meta):
        # Difference between the old and packed arenas, including int32
        # alignment. This is an unallocated planning prior, never fake storage.
        reserve=36*(8*meta.capacity+(2 if meta.capacity%2 else 0));original=self.budget_bytes;self.budget_bytes-=reserve
        try:
            try:plan=super().make_plan(meta);fallback=False
            except RuntimeError as e:
                if 'INFEASIBLE_BY_INTEGER_ORACLE' not in str(e):raise
                self.budget_bytes=original;plan=super().make_plan(meta);fallback=True
        finally:self.budget_bytes=original
        plan['estimate']['unallocated_rate_prior_bytes']=reserve
        plan.update(rate_selection='mapped-parent rate prior; actual map-free storage',packed_admission_fallback=fallback)
        return plan

class InnovationRefined(RateMatchedPacked):
    method='AR19-coarse-reference-fine-innovation-RVQ'
    def make_plan(self,meta):
        plan=super().make_plan(meta);desc=plan['description'];nr=plan['innovation_slots'];g=plan['groups'];options=[]
        for dk in range(self.package['Rmax_K']-plan['R_K']+1):
            for dv in range(self.package['Rmax_V']-plan['R_V']+1):
                if not nr and (dk or dv):continue
                extra=[('K_extra_index',(2,nr,dk,g),torch.uint8,'payload'),('V_extra_index',(2,nr,dv,g),torch.uint8,'payload')]
                total=plan['estimate']['resident_capacity_bytes']+36*(Arena.bytes(desc+extra)-Arena.bytes(desc))
                if total<=self.budget_bytes:options.append((dk,dv,extra,total))
        dk,dv,extra,total=max(options,key=lambda x:(plan['R_K']+plan['R_V']+x[0]+x[1],-abs(plan['R_K']+x[0]-plan['R_V']-x[1]),plan['R_K']+x[0]))
        plan['description']=desc+extra;plan['estimate']['resident_capacity_bytes']=total;plan['estimate']['layer_arena_bytes']=Arena.bytes(plan['description'])
        plan.update(innovation_R_K=plan['R_K']+dk,innovation_R_V=plan['R_V']+dv,extra_R_K=dk,extra_R_V=dv,reference_R_K=plan['R_K'],reference_R_V=plan['R_V'],refinement_scope='selected innovations only; reference and correspondence unchanged')
        return plan
    def _innovation(self,l,state,raw,slots,kind):
        result=super()._innovation(l,state,raw,slots,kind);nr=self.plan['innovation_slots'];a=self.plan['anchor_tokens'];extra=self.plan['extra_R_'+kind]
        if not nr or not extra or len(slots)!=nr or int(slots[0])!=a:return result
        t=self.tables[0];d=self.package['group_dim'];g=128//d;base=self.plan['R_'+kind];before=0.;after=0.
        for b in range(0,nr,512):
            x=raw[:,b:b+512].float();nn=x.shape[1];where=slots[b:b+nn];rho=state[kind+'_norm'][:,where].float();z=((x/t[kind+'_scale'][l].float()[:,None])/rho)@t['H'];z=z.reshape(2,nn,g,d)
            for rr in range(base):z-=t[kind+'_books'][rr].float()[state[kind+'_index'][:,where,rr].long()]
            z=z.reshape(1,2*nn*g,d);before+=float(z.square().sum())
            for rr in range(extra):
                cb=t[kind+'_books'][base+rr][None].float();ix=nearest(z,cb);state[kind+'_extra_index'][:,b:b+nn,rr]=ix.reshape(2,nn,g).to(torch.uint8);z-=torch.gather(cb,1,ix[...,None].expand(-1,-1,d))
            after+=float(z.square().sum())
        state['distortion_refinement_'+kind+'_rotated_before']=before;state['distortion_refinement_'+kind+'_rotated_after']=after
        return result  # append uses stored indices; this scratch return is unused.
    def _read_slots(self,l,s,kind,slots,independent):
        z=super()._read_slots(l,s,kind,slots,independent);nr=self.plan['innovation_slots'];extra=self.plan['extra_R_'+kind]
        if independent or not nr or not extra:return z
        assert slots.shape[1]==nr and int(slots[0,0])==self.plan['anchor_tokens']
        t=self.tables[0];d=self.package['group_dim'];g=128//d;base=self.plan['R_'+kind];delta=torch.zeros(2,nr,g,d,device='cuda')
        for rr in range(extra):delta+=t[kind+'_books'][base+rr].float()[s[kind+'_extra_index'][:,:,rr].long()]
        rho=s[kind+'_norm'][:,self.plan['anchor_tokens']:self.plan['anchor_tokens']+nr].float();delta=(delta.reshape(2,nr,128)@t['H'].T)*rho*t[kind+'_scale'][l].float()[:,None]
        return (z.float()+delta).bfloat16()

class HybridRateMatched(DenseFirst,RateMatchedPacked):pass
class HybridInnovationRefined(DenseFirst,InnovationRefined):pass
