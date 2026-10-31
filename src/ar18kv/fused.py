"""Paid multi-geometry banks, fixed-width arenas, native-content or CAL selector."""
import torch
from r15kv.codecs import PackageCodec
from r15kv.persistent_registry import Arena
from r15kv.budget import estimate_resident_bytes,pack_indices,unpack_indices
from r14kv.fitting import nearest
from ar17kv.text_protection import PaidPrefixText
from ar18kv.hybrid import DenseFirst
class FusedProduct(PackageCodec):
    method='PSRVQ-paid-multigeometry-content-selector';needs_positions=True;needs_pre_rope_K=True
    def prepare_request(self,meta,registry):
        self._trace_item=self.context.get('item_id');self._trace_layers=set()
        media=(self.context['tags']!=0).nonzero().flatten();n=min(64,len(media));ii=torch.linspace(0,len(media)-1,n,device=media.device).round().long();self._sample=media[ii].to(device='cuda',dtype=torch.int32);self._sample_n=n
        plan=super().prepare_request(meta,registry);registry.own('cache/selector_sample_positions',self._sample);return plan
    def make_plan(self,meta):
        dtype=torch.int16 if max(meta.position_max)+max(0,meta.generation_cap-1)<=32767 else torch.int32;extra=3*meta.capacity*torch.empty((),dtype=dtype).element_size()+256+4*self._sample_n;opts=[];menu=range(8,49,8) if self.policy.get('selector')=='parent_only' else range(4,49,4)
        for k in menu:
            for v in menu:
                desc=[]
                for side,w in [('K',k),('V',v)]:desc += [(side+'_blob',(2,meta.capacity,w),torch.uint8,'payload'),(side+'_norm',(2,meta.capacity,1),torch.bfloat16,'metadata'),(side+'_choice',(2,),torch.int16,'metadata')]
                est=estimate_resident_bytes(self.static_bytes,extra,desc)
                if est['resident_capacity_bytes']<=self.budget_bytes:opts.append((k,v,desc,est))
        if not opts:raise RuntimeError('INFEASIBLE_BY_INTEGER_ORACLE_FUSED_PSRVQ')
        k,v,desc,est=max(opts,key=lambda x:(x[0]+x[1],-abs(x[0]-x[1]),x[0]));return {'K_row_bytes':k,'V_row_bytes':v,'description':desc,'estimate':est,'capacity':meta.capacity,'position_dtype':str(dtype),'selector':self.policy.get('selector','content'),'sample_tokens':self._sample_n,'all_banks_paid':True,'selector_metadata_bytes':288+4*self._sample_n}
    def _normalized(self,l,side,raw):
        t=self.tables[0];u=(raw.float()-t[side+'_mean'][l].float()[:,None])/t[side+'_scale'][l].float()[:,None];rho=u.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6).bfloat16();return torch.matmul(u/rho.float(),t['H']),rho
    def _choose(self,l,side,raw,width):
        t=self.tables[0];mode=self.policy.get('selector','content');possible=[];sample=raw.index_select(1,self._sample.long());space,rho=self._normalized(l,side,sample)
        for bank,spec in enumerate(self.package['bank_specs']):
            if mode=='parent_only' and bank!=0:continue
            d=spec['dim'];g=128//d;step=g*spec['bits']//8;r=min(spec['Rmax'],width//step)
            if r<1:continue
            if mode=='calibrated':error=float(t['CAL_curves'][0 if side=='K' else 1,l,bank,r-1])
            elif mode=='parent_only':error=0.
            else:
                z=space.reshape(1,-1,d).clone();decoded=torch.zeros_like(z)
                for j in range(r):
                    cb=t[f'{side}_books_b{bank}'][j].float()[None];ix=nearest(z,cb);chosen=torch.gather(cb,1,ix[...,None].expand(-1,-1,d));z-=chosen;decoded+=chosen
                pred=torch.matmul(decoded.reshape_as(space),t['H'].T);pred=(pred*rho.float()*t[side+'_scale'][l].float()[:,None]+t[side+'_mean'][l].float()[:,None]).bfloat16().float();error=float((pred-sample.float()).square().sum()/sample.float().square().sum().clamp_min(1e-20))
            possible.append((error,r*spec['clusters'],bank,r))
        assert possible;_,_,bank,r=min(possible);return bank,r
    def append(self,l,state,kpost,v,pre=None):
        assert pre is not None;t=self.tables[0];start=state['n'];n=v.shape[-2];assert start+n<=self.meta.capacity
        for side,raw in [('K',pre[0]),('V',v[0])]:
            if start==0:
                assert int(self._sample.max())<n;bank,r=self._choose(l,side,raw,state[side+'_blob'].shape[-1]);state[side+'_choice'].copy_(torch.tensor([bank,r],dtype=torch.int16,device='cuda'))
            else:bank,r=map(int,state[side+'_choice'].tolist())
            spec=self.package['bank_specs'][bank];d=spec['dim'];g=128//d;step=g*spec['bits']//8
            for b in range(0,n,512):
                z,rho=self._normalized(l,side,raw[:,b:b+512]);nn=z.shape[1];state[side+'_norm'][:,start+b:start+b+nn].copy_(rho);z=z.reshape(1,-1,d)
                for j in range(r):
                    cb=t[f'{side}_books_b{bank}'][j].float()[None];ix=nearest(z,cb);encoded=pack_indices(ix.reshape(2,nn,g),spec['bits']);state[side+'_blob'][:,start+b:start+b+nn,j*step:(j+1)*step].copy_(encoded);z-=torch.gather(cb,1,ix[...,None].expand(-1,-1,d))
        state['n']+=n
        if l not in self._trace_layers:
            import hashlib,json,pathlib,sys
            from r14kv.common import ROOT
            rid=sys.argv[sys.argv.index('--run-id')+1] if '--run-id' in sys.argv else 'unassigned'
            row={'item_id':self._trace_item,'layer':l,'first_observed_append_start_tokens':start,'cached_tokens_at_observation':state['n'],'selector':self.policy.get('selector','content'),'sample_tokens':self._sample_n,'actual_GPU_choice_headers':{side:[int(x) for x in state[side+'_choice'].tolist()] for side in ['K','V']},'row_bytes':{side:state[side+'_blob'].shape[-1] for side in ['K','V']},'all_fixed_banks_paid':True}
            if 'K_video_mean' in state:
                row['video_affine_metadata_SHA256']={side+'_'+kind:hashlib.sha256(state[side+'_video_'+kind].view(torch.uint16).cpu().numpy().tobytes()).hexdigest() for side in ['K','V'] for kind in ['mean','scale']};row['statistic_alpha']=self.policy.get('statistic_alpha')
            path=ROOT/'data/runs'/rid/'codec_choice_headers.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a') as trace:trace.write(json.dumps(row)+'\n')
            self._trace_layers.add(l)
        return state
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        from r14kv.common import apply_key_rope
        t=self.tables[0];n=state['n'];result=[]
        for side in ['K','V']:
            bank,r=map(int,state[side+'_choice'].tolist());spec=self.package['bank_specs'][bank];d=spec['dim'];g=128//d;step=g*spec['bits']//8;z=torch.zeros(2,n,g,d,device='cuda')
            for j in range(r):
                ix=unpack_indices(state[side+'_blob'][:,:n,j*step:(j+1)*step],spec['bits'],g).long();z+=t[f'{side}_books_b{bank}'][j].float()[ix]
            raw=torch.matmul(z.reshape(2,n,128),t['H'].T);raw=raw*state[side+'_norm'][:,:n].float()*t[side+'_scale'][l].float()[:,None]+t[side+'_mean'][l].float()[:,None];result.append(raw.bfloat16()[None])
        phase=positions.float()[:,:,None]*inv_freq[None,None,:];out=[];widths=[16,24,24]
        for trig in [torch.cos,torch.sin]:
            a=(trig(phase)*attention_scaling).bfloat16();z=torch.cat([a[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);out.append(torch.cat([z,z],-1)[None,None])
        return apply_key_rope(result[0],*out),result[1]
    def finish_request(self):self._sample=None;super().finish_request()
class ProtectedFused(PaidPrefixText,FusedProduct):pass
class HybridFused(DenseFirst,ProtectedFused):pass
