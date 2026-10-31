"""Independent compact normalized shared RVQ reference implementation."""
import torch
from r16kv.gsrq import GSRQA0
from r15kv.budget import gsrq_description,estimate_resident_bytes
from r14kv.fitting import nearest
class CRVQReference(GSRQA0):
    method='CRVQ-independent-normalized-shared-RVQ';needs_pre_rope_K=True
    def book(self,l,kind,stage):
        g=l//(36//self.package['groups']);return self.tables[0][kind+'_books'][g,stage][None].float().expand(2,-1,-1)
    def make_plan(self,meta):
        dtype=torch.int16 if max(meta.position_max)+max(0,meta.generation_cap-1)<=32767 else torch.int32;posbytes=3*meta.capacity*torch.empty((),dtype=dtype).element_size()+256;options=[]
        for k in range(1,13):
            for v in range(1,13):
                desc=gsrq_description(meta.capacity,k,v)+[(side+'_norm',(2,meta.capacity,1),torch.bfloat16,'metadata') for side in ['K','V']];est=estimate_resident_bytes(self.static_bytes,posbytes,desc)
                if est['resident_capacity_bytes']<=self.budget_bytes:options.append((k,v,desc,est))
        if not options:raise RuntimeError('INFEASIBLE_BY_INTEGER_ORACLE_CRVQ')
        k,v,desc,est=max(options,key=lambda x:(x[0]+x[1],-abs(x[0]-x[1]),x[0]));return {'R_K':k,'R_V':v,'description':desc,'estimate':est,'capacity':meta.capacity,'position_dtype':str(dtype),'allocator':'max feasible total prefix with balanced K/V; norm bytes paid','loaded_Rmax_K':12,'loaded_Rmax_V':12}
    def append(self,l,state,kpost,v,pre=None):
        assert pre is not None and pre.shape==kpost.shape;start=state['n'];n=v.shape[-2];assert start+n<=self.meta.capacity;t=self.tables[0]
        for kind,x in [('K',pre[0]),('V',v[0])]:
            z=(x.float()-t[kind+'_mean'][l].float()[:,None])/t[kind+'_scale'][l].float()[:,None];norm=z.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6).bfloat16();state[kind+'_norm'][:,start:start+n].copy_(norm);self.encode_kind(l,kind,z/norm.float(),state[kind+'_index'][:,start:start+n])
        state['n']+=n;return state
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        from r14kv.common import apply_key_rope
        t=self.tables[0];n=state['n'];result=[]
        for kind in ['K','V']:
            z=self.raw_decode_kind(l,kind,state[kind+'_index'][:,:n]);z=z*state[kind+'_norm'][:,:n].float()*t[kind+'_scale'][l].float()[:,None]+t[kind+'_mean'][l].float()[:,None];result.append(z.bfloat16()[None])
        phase=positions.float()[:,:,None]*inv_freq[None,None,:];out=[];widths=[16,24,24]
        for trig in [torch.cos,torch.sin]:
            a=(trig(phase)*attention_scaling).bfloat16();b=torch.cat([a[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);out.append(torch.cat([b,b],-1)[None,None])
        return apply_key_rope(result[0],*out),result[1]
