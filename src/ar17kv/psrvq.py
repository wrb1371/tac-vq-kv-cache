"""Shared small-dimensional product RVQ; paid global tables, explicit packed indices."""
import torch
from r15kv.codecs import PackageCodec
from r15kv.budget import estimate_resident_bytes
from r14kv.fitting import nearest
class ProductRVQ(PackageCodec):
    method='PSRVQ-independent-product-residual';needs_positions=True;needs_pre_rope_K=True
    def make_plan(self,meta):
        d=self.package['group_dim'];g=128//d;maximum=self.package['Rmax_K'];dtype=torch.int16 if max(meta.position_max)+max(0,meta.generation_cap-1)<=32767 else torch.int32;posbytes=3*meta.capacity*torch.empty((),dtype=dtype).element_size()+256;options=[]
        for k in range(1,maximum+1):
            for v in range(1,maximum+1):
                desc=[('K_index',(2,meta.capacity,k,g),torch.uint8,'payload'),('V_index',(2,meta.capacity,v,g),torch.uint8,'payload')]+[(side+'_norm',(2,meta.capacity,1),torch.bfloat16,'metadata') for side in ['K','V']];est=estimate_resident_bytes(self.static_bytes,posbytes,desc)
                if est['resident_capacity_bytes']<=self.budget_bytes:options.append((k,v,desc,est))
        if not options:raise RuntimeError('INFEASIBLE_BY_INTEGER_ORACLE_PSRVQ')
        k,v,desc,est=max(options,key=lambda x:(x[0]+x[1],-abs(x[0]-x[1]),x[0]));return {'R_K':k,'R_V':v,'group_dim':d,'groups':g,'K_bits_per_component':8*k/d,'V_bits_per_component':8*v/d,'description':desc,'estimate':est,'capacity':meta.capacity,'position_dtype':str(dtype),'loaded_Rmax_K':maximum,'loaded_Rmax_V':maximum,'allocator':'balanced maximum feasible paid prefix'}
    def append(self,l,state,kpost,v,pre=None):
        assert pre is not None and pre.shape==kpost.shape;assert not torch.backends.cuda.matmul.allow_tf32;t=self.tables[0];start=state['n'];n=v.shape[-2];assert start+n<=self.meta.capacity;d=self.package['group_dim'];g=128//d
        for kind,x in [('K',pre[0]),('V',v[0])]:
            for b in range(0,n,512):
                raw=x[:,b:b+512].float();nn=raw.shape[1];u=(raw-t[kind+'_mean'][l].float()[:,None])/t[kind+'_scale'][l].float()[:,None];rho=u.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6).bfloat16();state[kind+'_norm'][:,start+b:start+b+nn].copy_(rho);z=torch.matmul(u/rho.float(),t['H']).reshape(1,2*nn*g,d)
                for r in range(state[kind+'_index'].shape[2]):
                    cb=t[kind+'_books'][r][None].float();ix=nearest(z,cb);state[kind+'_index'][:,start+b:start+b+nn,r].copy_(ix.reshape(2,nn,g).to(torch.uint8));z-=torch.gather(cb,1,ix[...,None].expand(-1,-1,d))
        state['n']+=n;return state
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        from r14kv.common import apply_key_rope
        t=self.tables[0];n=state['n'];d=self.package['group_dim'];g=128//d;result=[]
        for kind in ['K','V']:
            z=torch.zeros(2,n,g,d,device='cuda')
            for r in range(state[kind+'_index'].shape[2]):z+=t[kind+'_books'][r].float()[state[kind+'_index'][:,:n,r].long()]
            z=torch.matmul(z.reshape(2,n,128),t['H'].T);z=z*state[kind+'_norm'][:,:n].float()*t[kind+'_scale'][l].float()[:,None]+t[kind+'_mean'][l].float()[:,None];result.append(z.bfloat16()[None])
        phase=positions.float()[:,:,None]*inv_freq[None,None,:];out=[];widths=[16,24,24]
        for trig in [torch.cos,torch.sin]:
            a=(trig(phase)*attention_scaling).bfloat16();b=torch.cat([a[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);out.append(torch.cat([b,b],-1)[None,None])
        return apply_key_rope(result[0],*out),result[1]
