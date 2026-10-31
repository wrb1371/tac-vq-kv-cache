"""Closed-loop anchor innovations with paid local correspondence codes.

All native positions survive. A one-hop GOP anchor prevents recursive drift.
Correspondence is a feature-space proxy, not supervised object tracking.
"""
import bisect,torch
from ar17kv.psrvq import ProductRVQ
from ar17kv.text_protection import PaidPrefixText
from ar18kv.hybrid import DenseFirst
from r14kv.fitting import nearest
from r14kv.common import apply_key_rope
from r15kv.persistent_registry import Arena

class Temporal(PaidPrefixText,ProductRVQ):
    method='AR19-closed-loop-temporal-PSRVQ'
    def prepare_request(self,meta,registry):
        tags=self.context['tags'];self._video_list=(tags==2).nonzero().flatten().tolist()
        grid=self.context['video_grid_thw'];assert grid.shape==(1,3)
        self._nt,self._h,self._w=map(int,grid[0].tolist());self._h//=2;self._w//=2
        assert len(self._video_list)==self._nt*self._h*self._w
        self._vid_gpu=None
        plan=super().prepare_request(meta,registry)
        self._vid_gpu=torch.tensor(self._video_list,dtype=torch.int32,device='cuda');registry.own('cache/native_video_sequence_positions',self._vid_gpu)
        return plan
    def make_plan(self,meta):
        desc=[('T_code',(2,meta.capacity),torch.uint8,'metadata')]
        extra=36*Arena.bytes(desc)+4*len(self._video_list)
        original=self.budget_bytes;self.budget_bytes-=extra
        try:plan=super().make_plan(meta)
        finally:self.budget_bytes=original
        before=Arena.bytes(plan['description']);plan['description']+=desc
        assert Arena.bytes(plan['description'])-before==Arena.bytes(desc)
        plan['estimate']['resident_capacity_bytes']+=extra
        plan['estimate']['global_request_bytes']+=4*len(self._video_list)
        plan['estimate']['layer_arena_bytes']=Arena.bytes(plan['description'])
        plan.update(temporal_policy=self.policy,merged_video_grid=[self._nt,self._h,self._w],temporal_metadata_bytes=extra,temporal_reference='one-hop reconstructed GOP anchor; 255 independent escape')
        return plan
    def allocate_state(self,l,registry):
        s=super().allocate_state(l,registry);s['T_code'].fill_(255);return s
    def _decode_raw(self,l,state,n,kind):
        t=self.tables[0];d=self.package['group_dim'];g=128//d
        z=torch.zeros(2,n,g,d,device='cuda')
        for rr in range(state[kind+'_index'].shape[2]):z+=t[kind+'_books'][rr].float()[state[kind+'_index'][:,:n,rr].long()]
        z=(z.reshape(2,n,128)@t['H'].T)*state[kind+'_norm'][:,:n].float()*t[kind+'_scale'][l].float()[:,None]
        independent=(state['T_code'][:,:n]==255)
        return (z+independent[...,None]*t[kind+'_mean'][l].float()[:,None]).bfloat16()
    def _innovation(self,l,state,raw,slots,kind):
        t=self.tables[0];d=self.package['group_dim'];g=128//d;answers=[]
        for b in range(0,len(slots),512):
            x=raw[:,b:b+512].float();nn=x.shape[1];u=x/t[kind+'_scale'][l].float()[:,None]
            rho=u.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6).bfloat16();where=slots[b:b+nn];state[kind+'_norm'][:,where]=rho
            z=(u/rho.float())@t['H'];z=z.reshape(1,2*nn*g,d);rec=torch.zeros_like(z)
            for rr in range(state[kind+'_index'].shape[2]):
                cb=t[kind+'_books'][rr][None].float();ix=nearest(z,cb);state[kind+'_index'][:,where,rr]=ix.reshape(2,nn,g).to(torch.uint8)
                q=torch.gather(cb,1,ix[...,None].expand(-1,-1,d));z-=q;rec+=q
            answers.append(((rec.reshape(2,nn,128)@t['H'].T)*rho.float()*t[kind+'_scale'][l].float()[:,None]).bfloat16())
        return torch.cat(answers,1)
    def _reference_indices(self,ordinals,codes):
        p=self._h*self._w;gop=int(self.policy.get('gop',2));frame=ordinals//p;sp=ordinals%p
        yy=(sp//self._w+codes.long()//3-1).clamp(0,self._h-1);xx=(sp%self._w+codes.long()%3-1).clamp(0,self._w-1)
        if self.policy.get('mode')=='shuffled':xx=(xx+self._w//2+1)%self._w
        reford=(frame//gop*gop)*p+yy*self._w+xx
        return self._vid_gpu.long()[reford]
    @torch.no_grad()
    def append(self,l,state,kpost,v,pre=None):
        start=state['n'];n=kpost.shape[-2];state=super().append(l,state,kpost,v,pre)
        if start or self.policy.get('mode')=='independent':return state
        assert n>self._video_list[-1], 'full finite video prefix required for this port'
        p=self._h*self._w;gop=int(self.policy.get('gop',2));ordinals=torch.arange(len(self._video_list),device='cuda')
        targets=ordinals[(ordinals//p)%gop!=0]
        if targets.numel()==0:return state
        slots=self._vid_gpu.long()[targets];t=self.tables[0]
        original={'K':self._decode_raw(l,state,n,'K'),'V':self._decode_raw(l,state,n,'V')}
        raw={'K':pre[0].float(),'V':v[0].float()};best=torch.full((2,len(targets)),float('inf'),device='cuda');codes=torch.full_like(best,4,dtype=torch.uint8)
        modes=[4] if self.policy.get('mode') in ['same','shuffled'] else range(9)
        for c in modes:
            ref=self._reference_indices(targets,torch.full_like(codes,c));score=0
            for kind in ['K','V']:
                prediction=original[kind].gather(1,ref[...,None].expand(-1,-1,128)).float()
                score=score+((raw[kind][:,slots]-prediction)/t[kind+'_scale'][l].float()[:,None]).square().mean(-1)
            take=score<best;codes=torch.where(take,torch.full_like(codes,c),codes);best=torch.minimum(best,score)
        ref=self._reference_indices(targets,codes);keep={};err0=0;err1=0
        for kind in ['K','V']:
            oldix=state[kind+'_index'][:,slots].clone();oldnorm=state[kind+'_norm'][:,slots].clone()
            prediction=original[kind].gather(1,ref[...,None].expand(-1,-1,128));innovation=raw[kind][:,slots]-prediction.float()
            decoded=self._innovation(l,state,innovation,slots,kind);reconstructed=(prediction.float()+decoded.float()).bfloat16()
            scale=t[kind+'_scale'][l].float()[:,None]
            err0=err0+((raw[kind][:,slots]-original[kind][:,slots].float())/scale).square().mean(-1)
            err1=err1+((raw[kind][:,slots]-reconstructed.float())/scale).square().mean(-1)
            keep[kind]=(oldix,oldnorm)
        escape=err0<=err1 if self.policy.get('escape',False) else torch.zeros_like(codes,dtype=torch.bool)
        for kind,(oldix,oldnorm) in keep.items():
            state[kind+'_index'][:,slots]=torch.where(escape[...,None,None],oldix,state[kind+'_index'][:,slots])
            state[kind+'_norm'][:,slots]=torch.where(escape[...,None],oldnorm,state[kind+'_norm'][:,slots])
        state['T_code'][:,slots]=torch.where(escape,torch.full_like(codes,255),codes)
        state['temporal_targets']=int(targets.numel());state['temporal_escape_heads']=int(escape.sum());state['temporal_innovation_heads']=int((~escape).sum())
        state['distortion_independent_sum']=float(err0.sum());state['distortion_temporal_sum']=float(torch.where(escape,err0,err1).sum())
        return state
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        n=state['n'];out=[];p=self._h*self._w;gop=int(self.policy.get('gop',2))
        ordinals=torch.arange(len(self._video_list),device='cuda');targets=ordinals[(ordinals//p)%gop!=0];slots=self._vid_gpu.long()[targets]
        codes=state['T_code'][:,slots];active=codes!=255;ref=self._reference_indices(targets,codes.clamp_max(8))
        for kind in ['K','V']:
            z=self._decode_raw(l,state,n,kind)
            if len(targets):
                prediction=z.gather(1,ref[...,None].expand(-1,-1,128)).float();z[:,slots]=torch.where(active[...,None],(z[:,slots].float()+prediction).bfloat16(),z[:,slots])
            out.append(z[None])
        phase=positions.float()[:,:,None]*inv_freq[None,None,:];trigs=[];widths=[16,24,24]
        for trig in [torch.cos,torch.sin]:
            a=(trig(phase)*attention_scaling).bfloat16();b=torch.cat([a[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);trigs.append(torch.cat([b,b],-1)[None,None])
        k=apply_key_rope(out[0],*trigs);v=out[1];m=bisect.bisect_left(self._text_list,n);assert m==state['text_filled']
        if m:
            ix=self._text_gpu[:m].long();k[0].index_copy_(1,ix,state['text_FP_K'][:,:m]);v[0].index_copy_(1,ix,state['text_FP_V'][:,:m])
        return k,v
    def finish_request(self):
        self._vid_gpu=None;self._video_list=[];super().finish_request()

class HybridTemporal(DenseFirst,Temporal):pass
