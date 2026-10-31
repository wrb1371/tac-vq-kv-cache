"""Fixed private-first, shared-tail FP8 books; no duplicated resident expansion."""
import torch
from r15kv.codecs import GSRQBudget
from r15kv.budget import gsrq_description,estimate_resident_bytes
from r14kv.fitting import nearest

class GSRQA0(GSRQBudget):
    method='GSRQ-R16-CAL-prefix-A0'
    def book(self,l,kind,stage):return self.tables[l][kind][:,stage].float()
    def make_plan(self,meta):
        dtype=torch.int16 if max(meta.position_max)+max(0,meta.generation_cap-1)<=32767 else torch.int32
        posbytes=3*meta.capacity*torch.empty((),dtype=dtype).element_size()+256
        maximum=min(self.package['Rmax_K'],self.package['Rmax_V']);options=[]
        scores=self.policy.get('prefix_scores',{})
        for r in sorted(set([3,6,9]+([12] if maximum>=12 else []))):
            if r>maximum:continue
            desc=gsrq_description(meta.capacity,r,r);est=estimate_resident_bytes(self.static_bytes,posbytes,desc)
            if est['resident_capacity_bytes']<=self.budget_bytes:options.append((float(scores.get(str(r),1./r)),r,desc,est))
        if not options:raise RuntimeError('INFEASIBLE_BY_INTEGER_ORACLE_GSRQ_R16')
        loss,r,desc,est=min(options,key=lambda x:(x[0],x[3]['resident_capacity_bytes']))
        return {'R_K':r,'R_V':r,'loaded_Rmax_K':self.package['Rmax_K'],'loaded_Rmax_V':self.package['Rmax_V'],'description':desc,'estimate':est,'position_dtype':str(dtype),'capacity':meta.capacity,'allocator':'CAL-validated finite whole-prefix A0','CAL_proxy_score':loss,'remaining_bytes':self.budget_bytes-est['resident_capacity_bytes'],'stop_reason':'best CAL-validated feasible prefix; more stages are not assumed to improve task quality'}
    @torch.no_grad()
    def encode_kind(self,l,kind,x,dest):
        for b in range(0,x.shape[1],4096):
            residual=x[:,b:b+4096].float().clone()
            for stage in range(dest.shape[-1]):
                cb=self.book(l,kind,stage);ix=nearest(residual,cb);dest[:,b:b+residual.shape[1],stage].copy_(ix.to(torch.uint8));residual-=torch.gather(cb,1,ix[...,None].expand(-1,-1,128))
    def append(self,l,state,kpost,v,pre=None):
        assert pre is not None and pre.shape==kpost.shape;start=state['n'];n=pre.shape[-2];assert start+n<=self.meta.capacity
        self.encode_kind(l,'K',pre[0],state['K_index'][:,start:start+n]);self.encode_kind(l,'V',v[0],state['V_index'][:,start:start+n]);state['n']+=n;return state
    def raw_decode_kind(self,l,kind,index):
        out=torch.zeros(2,index.shape[1],128,device=index.device)
        for stage in range(index.shape[-1]):out+=torch.gather(self.book(l,kind,stage),1,index[...,stage].long()[...,None].expand(-1,-1,128))
        return out
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        from r14kv.common import apply_key_rope
        n=state['n'];k=self.raw_decode_kind(l,'K',state['K_index'][:,:n])[None];v=self.raw_decode_kind(l,'V',state['V_index'][:,:n])[None]
        phase=positions.float()[:,:,None]*inv_freq[None,None,:];cos=(phase.cos()*attention_scaling).bfloat16();sin=(phase.sin()*attention_scaling).bfloat16();widths=[16,24,24]
        sc=torch.cat([cos[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);ss=torch.cat([sin[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1)
        return apply_key_rope(k.bfloat16(),torch.cat([sc,sc],-1)[None,None],torch.cat([ss,ss],-1)[None,None]),v.bfloat16()

class GSRQShared(GSRQA0):
    method='GSRQ-R16-two-private-then-shared-head-tail'
    def book(self,l,kind,stage):
        private=int(self.package['private_stages'])
        if stage<private:return self.tables[l][kind+'_private'][:,stage].float()
        return self.tables[l][kind+'_shared'][:,stage-private].float().expand(2,-1,-1)
