"""Fresh 16-centroid product RVQ with physically packed 4-bit addresses."""
import torch
from ar17kv.psrvq import ProductRVQ
from ar17kv.text_protection import PaidPrefixText
from r15kv.budget import estimate_resident_bytes,pack_indices,unpack_indices
class PackedProduct(ProductRVQ):
    method='PSRVQ-16centroid-packed4bit'
    def make_plan(self,meta):
        d=self.package['group_dim'];g=128//d;maximum=self.package['Rmax_K'];dtype=torch.int16 if max(meta.position_max)+max(0,meta.generation_cap-1)<=32767 else torch.int32;posbytes=3*meta.capacity*torch.empty((),dtype=dtype).element_size()+256;options=[]
        for k in range(1,maximum+1):
            for v in range(1,maximum+1):
                desc=[('K_index',(2,meta.capacity,k,g//2),torch.uint8,'payload'),('V_index',(2,meta.capacity,v,g//2),torch.uint8,'payload')]+[(s+'_norm',(2,meta.capacity,1),torch.bfloat16,'metadata') for s in ['K','V']];est=estimate_resident_bytes(self.static_bytes,posbytes,desc)
                if est['resident_capacity_bytes']<=self.budget_bytes:options.append((k,v,desc,est))
        if not options:raise RuntimeError('INFEASIBLE_BY_INTEGER_ORACLE_PACKED_PSRVQ')
        k,v,desc,est=max(options,key=lambda x:(x[0]+x[1],-abs(x[0]-x[1]),x[0]));return {'R_K':k,'R_V':v,'group_dim':d,'groups':g,'index_bits':4,'K_bits_per_component':4*k/d,'V_bits_per_component':4*v/d,'description':desc,'estimate':est,'capacity':meta.capacity,'position_dtype':str(dtype),'loaded_Rmax_K':maximum,'loaded_Rmax_V':maximum,'allocator':'balanced maximum feasible paid packed4prefix'}
    def append(self,l,state,kpost,v,pre=None):
        assert pre is not None and pre.shape==kpost.shape;t=self.tables[0];start=state['n'];n=v.shape[-2];d=self.package['group_dim'];g=128//d;assert start+n<=self.meta.capacity
        for side,x in [('K',pre[0]),('V',v[0])]:
            for b in range(0,n,512):
                raw=x[:,b:b+512].float();nn=raw.shape[1];u=(raw-t[side+'_mean'][l].float()[:,None])/t[side+'_scale'][l].float()[:,None];rho=u.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6).bfloat16();state[side+'_norm'][:,start+b:start+b+nn].copy_(rho);z=torch.matmul(u/rho.float(),t['H']).reshape(1,2*nn*g,d)
                from r14kv.fitting import nearest
                for r in range(state[side+'_index'].shape[2]):
                    cb=t[side+'_books'][r][None].float();assert cb.shape[1]==16;ix=nearest(z,cb);state[side+'_index'][:,start+b:start+b+nn,r].copy_(pack_indices(ix.reshape(2,nn,g),4));z-=torch.gather(cb,1,ix[...,None].expand(-1,-1,d))
        state['n']+=n;return state
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        from r14kv.common import apply_key_rope
        t=self.tables[0];n=state['n'];d=self.package['group_dim'];g=128//d;result=[]
        for side in ['K','V']:
            z=torch.zeros(2,n,g,d,device='cuda')
            for r in range(state[side+'_index'].shape[2]):z+=t[side+'_books'][r].float()[unpack_indices(state[side+'_index'][:,:n,r],4,g).long()]
            z=torch.matmul(z.reshape(2,n,128),t['H'].T);z=z*state[side+'_norm'][:,:n].float()*t[side+'_scale'][l].float()[:,None]+t[side+'_mean'][l].float()[:,None];result.append(z.bfloat16()[None])
        phase=positions.float()[:,:,None]*inv_freq[None,None,:];out=[];widths=[16,24,24]
        for trig in [torch.cos,torch.sin]:
            a=(trig(phase)*attention_scaling).bfloat16();b=torch.cat([a[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);out.append(torch.cat([b,b],-1)[None,None])
        return apply_key_rope(result[0],*out),result[1]
class ProtectedPacked(PaidPrefixText,PackedProduct):
    method='PSRVQ-paid-text-packed4bit';needs_pre_rope_K=True
