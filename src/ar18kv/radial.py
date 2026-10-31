"""Physical one-byte radial metadata, shared ProductRVQ directions unchanged."""
import torch
from ar17kv.psrvq import ProductRVQ
from ar17kv.text_protection import PaidPrefixText
from ar18kv.hybrid import DenseFirst
from r15kv.budget import estimate_resident_bytes
from r14kv.fitting import nearest
class Radius8(ProductRVQ):
    method='PSRVQ-one-byte-radial-metadata';needs_pre_rope_K=True
    def make_plan(self,meta):
        g=128//self.package['group_dim'];maximum=self.package['Rmax_K'];dtype=torch.int16 if max(meta.position_max)+max(0,meta.generation_cap-1)<=32767 else torch.int32;extra=3*meta.capacity*torch.empty((),dtype=dtype).element_size()+256;normtype=torch.float8_e4m3fn if self.package['radial_codec']=='fp8' else torch.uint8;opts=[]
        for k in range(1,maximum+1):
            for v in range(1,maximum+1):
                desc=[('K_index',(2,meta.capacity,k,g),torch.uint8,'payload'),('V_index',(2,meta.capacity,v,g),torch.uint8,'payload')]+[(s+'_norm8',(2,meta.capacity,1),normtype,'metadata') for s in ['K','V']];est=estimate_resident_bytes(self.static_bytes,extra,desc)
                if est['resident_capacity_bytes']<=self.budget_bytes:opts.append((k,v,desc,est))
        if not opts:raise RuntimeError('INFEASIBLE_BY_INTEGER_ORACLE_RADIAL8')
        k,v,desc,est=max(opts,key=lambda x:(x[0]+x[1],-abs(x[0]-x[1]),x[0]));return {'R_K':k,'R_V':v,'group_dim':16,'groups':g,'radial_codec':self.package['radial_codec'],'norm_bytes_per_scalar':1,'description':desc,'estimate':est,'capacity':meta.capacity,'position_dtype':str(dtype),'loaded_Rmax_K':maximum,'loaded_Rmax_V':maximum,'encoder_uses_reconstructed_radius':True}
    def _encode_radius(self,l,side,rho):
        if self.package['radial_codec']=='fp8':
            codes=rho.float().clamp(2**-9,448).to(torch.float8_e4m3fn);clipped=int(((rho<2**-9)|(rho>448)).sum());return codes,codes.float(),clipped
        levels=self.tables[0][side+'_radius_levels'][l];indices=[];restored=[];clipped=0
        for h in range(2):
            b=(levels[h,:-1].float()*levels[h,1:].float()).sqrt();ix=torch.searchsorted(b,rho[h,:,0].float().contiguous(),right=False);indices.append(ix.to(torch.uint8));restored.append(levels[h][ix].float());clipped+=int(((rho[h]<levels[h,0])|(rho[h]>levels[h,-1])).sum())
        return torch.stack(indices)[...,None],torch.stack(restored)[...,None],clipped
    def append(self,l,state,kpost,v,pre=None):
        assert pre is not None;t=self.tables[0];start=state['n'];n=v.shape[-2];d=16;g=8;clips={s:0 for s in ['K','V']}
        for side,x in [('K',pre[0]),('V',v[0])]:
            for b in range(0,n,512):
                raw=x[:,b:b+512].float();nn=raw.shape[1];u=(raw-t[side+'_mean'][l].float()[:,None])/t[side+'_scale'][l].float()[:,None];rho=u.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6).bfloat16();code,reconstructed,clipped=self._encode_radius(l,side,rho);clips[side]+=clipped;assert (reconstructed>0).all() and torch.isfinite(reconstructed).all();state[side+'_norm8'][:,start+b:start+b+nn].copy_(code);z=torch.matmul(u/reconstructed,t['H']).reshape(1,2*nn*g,d)
                for r in range(state[side+'_index'].shape[2]):
                    cb=t[side+'_books'][r][None].float();ix=nearest(z,cb);state[side+'_index'][:,start+b:start+b+nn,r].copy_(ix.reshape(2,nn,g).to(torch.uint8));z-=torch.gather(cb,1,ix[...,None].expand(-1,-1,d))
        state['n']+=n
        import json,sys
        from r14kv.common import ROOT
        rid=sys.argv[sys.argv.index('--run-id')+1]
        with (ROOT/'data/runs'/rid/'radial_range_counts.jsonl').open('a') as f:f.write(json.dumps({'layer':l,'start':start,'tokens':n,'codec':self.package['radial_codec'],'range_clips_by_side':clips})+'\n')
        return state
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        from r14kv.common import apply_key_rope
        t=self.tables[0];n=state['n'];result=[]
        for side in ['K','V']:
            z=torch.zeros(2,n,8,16,device='cuda')
            for r in range(state[side+'_index'].shape[2]):z+=t[side+'_books'][r].float()[state[side+'_index'][:,:n,r].long()]
            if self.package['radial_codec']=='fp8':rho=state[side+'_norm8'][:,:n].float()
            else:rho=torch.stack([t[side+'_radius_levels'][l,h][state[side+'_norm8'][h,:n,0].long()].float() for h in range(2)])[...,None]
            raw=torch.matmul(z.reshape(2,n,128),t['H'].T)*rho*t[side+'_scale'][l].float()[:,None]+t[side+'_mean'][l].float()[:,None];result.append(raw.bfloat16()[None])
        phase=positions.float()[:,:,None]*inv_freq[None,None,:];out=[];widths=[16,24,24]
        for trig in [torch.cos,torch.sin]:
            a=(trig(phase)*attention_scaling).bfloat16();b=torch.cat([a[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);out.append(torch.cat([b,b],-1)[None,None])
        return apply_key_rope(result[0],*out),result[1]
class ProtectedRadius8(PaidPrefixText,Radius8):pass
class HybridRadius8(DenseFirst,ProtectedRadius8):pass
