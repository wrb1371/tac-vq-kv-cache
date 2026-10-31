"""Media-only per-video affine preconditioner, all request parameters GPU-owned."""
import torch
from ar18kv.fused import FusedProduct
from ar18kv.hybrid import DenseFirst
from ar17kv.text_protection import PaidPrefixText
from r15kv.persistent_registry import Arena
from r14kv.fitting import nearest
class VideoStatsFused(FusedProduct):
    def make_plan(self,meta):
        desc=[(s+'_'+k,(2,128),torch.bfloat16,'metadata') for s in ['K','V'] for k in ['video_mean','video_scale']];extra=36*Arena.bytes(desc);original=self.budget_bytes;self.budget_bytes-=extra
        try:p=super().make_plan(meta)
        finally:self.budget_bytes=original
        p['description']+=desc;p['estimate']['resident_capacity_bytes']+=extra;p['estimate']['layer_arena_bytes']+=Arena.bytes(desc);p.update(video_affine_bytes=extra,video_statistic_alpha=self.policy.get('statistic_alpha',1.),statistic_source='64 uniformly-spaced native media tokens; no query/gold');return p
    def _stats(self,l,side):return self._current_state[side+'_video_mean'].float()[:,None],self._current_state[side+'_video_scale'].float()[:,None]
    def _normalized(self,l,side,raw):
        mean,scale=self._stats(l,side);u=(raw.float()-mean)/scale;rho=u.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6).bfloat16();return torch.matmul(u/rho.float(),self.tables[0]['H']),rho
    def _choose(self,l,side,raw,width):
        t=self.tables[0];sample=raw.index_select(1,self._sample.long());space,rho=self._normalized(l,side,sample);mean,scale=self._stats(l,side);possible=[]
        for bank,spec in enumerate(self.package['bank_specs']):
            d=spec['dim'];g=128//d;step=g*spec['bits']//8;r=min(spec['Rmax'],width//step)
            if r<1:continue
            z=space.reshape(1,-1,d).clone();decoded=torch.zeros_like(z)
            for j in range(r):
                cb=t[f'{side}_books_b{bank}'][j].float()[None];ix=nearest(z,cb);chosen=torch.gather(cb,1,ix[...,None].expand(-1,-1,d));z-=chosen;decoded+=chosen
            pred=(torch.matmul(decoded.reshape_as(space),t['H'].T)*rho.float()*scale+mean).bfloat16().float();error=float((pred-sample.float()).square().sum()/sample.float().square().sum().clamp_min(1e-20));possible.append((error,r*spec['clusters'],bank,r))
        assert possible;_,_,bank,r=min(possible);return bank,r
    def append(self,l,state,kpost,v,pre=None):
        assert pre is not None;self._current_state=state
        try:
            if state['n']==0:
                a=float(self.policy.get('statistic_alpha',1.));assert 0<a<=1;t=self.tables[0]
                for side,x in [('K',pre[0]),('V',v[0])]:
                    sample=x.index_select(1,self._sample.long()).float();mean=sample.mean(1);scale=sample.var(1,unbiased=False).sqrt();base=t[side+'_scale'][l].float();scale=scale.clamp_min(base*.1).clamp_min(1e-6);state[side+'_video_mean'].copy_((1-a)*t[side+'_mean'][l].float()+a*mean);state[side+'_video_scale'].copy_((1-a)*base+a*scale)
            return super().append(l,state,kpost,v,pre)
        finally:self._current_state=None
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        from r14kv.common import apply_key_rope
        t=self.tables[0];n=state['n'];result=[];self._current_state=state
        try:
            for side in ['K','V']:
                bank,r=map(int,state[side+'_choice'].tolist());spec=self.package['bank_specs'][bank];d=spec['dim'];g=128//d;step=g*spec['bits']//8;z=torch.zeros(2,n,g,d,device='cuda')
                from r15kv.budget import unpack_indices
                for j in range(r):
                    ix=unpack_indices(state[side+'_blob'][:,:n,j*step:(j+1)*step],spec['bits'],g).long();z+=t[f'{side}_books_b{bank}'][j].float()[ix]
                mean,scale=self._stats(l,side);raw=torch.matmul(z.reshape(2,n,128),t['H'].T)*state[side+'_norm'][:,:n].float()*scale+mean;result.append(raw.bfloat16()[None])
        finally:self._current_state=None
        phase=positions.float()[:,:,None]*inv_freq[None,None,:];out=[];widths=[16,24,24]
        for trig in [torch.cos,torch.sin]:
            a=(trig(phase)*attention_scaling).bfloat16();z=torch.cat([a[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);out.append(torch.cat([z,z],-1)[None,None])
        return apply_key_rope(result[0],*out),result[1]
class ProtectedVideoStats(PaidPrefixText,VideoStatsFused):pass
class HybridVideoStats(DenseFirst,ProtectedVideoStats):pass
