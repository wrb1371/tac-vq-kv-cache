"""Video-local full-vector RVQ: every inferred dictionary byte is paid request S."""
import hashlib,json,pathlib,sys,torch
from r14kv.fitting import nearest
from r14kv.common import ROOT,apply_key_rope
from r15kv.codecs import PackageCodec
from r15kv.budget import estimate_resident_bytes
from ar17kv.text_protection import PaidPrefixText
from ar18kv.hybrid import DenseFirst
class LocalVideoDictionary(PackageCodec):
    method='video-local-full-vector-residual-dictionary';needs_positions=True;needs_pre_rope_K=True
    def prepare_request(self,meta,registry):
        self._trace_item=self.context.get('item_id');self._traced=set();media=(self.context['tags']==2).nonzero().flatten();self._media_n=len(media);assert self._media_n>=8
        n=min(int(self.policy.get('sample_max',2048)),self._media_n);ii=torch.linspace(0,self._media_n-1,n,device=media.device).round().long();self._sample=media[ii].to(device='cuda',dtype=torch.int32);self._sample_n=n
        plan=super().prepare_request(meta,registry);registry.own('cache/local_dictionary_sample_positions',self._sample);return plan
    def make_plan(self,meta):
        r=int(self.policy['stages']);assert r in [1,2];maxk=1024//r;dtype=torch.int16 if max(meta.position_max)+meta.generation_cap-1<=32767 else torch.int32
        extra=3*meta.capacity*torch.empty((),dtype=dtype).element_size()+256+4*self._sample_n
        for k in [1024,768,512,384,256,192,128,96,64,32,16,8,4]:
            if k>maxk or k>self._sample_n or r*k>self._media_n//2:continue
            index_dtype=torch.uint8 if k<=256 else torch.int16;desc=[]
            for side in ['K','V']:desc += [(side+'_index',(2,meta.capacity,r),index_dtype,'payload'),(side+'_local_books',(r,2,k,128),torch.float16,'metadata')]
            est=estimate_resident_bytes(self.static_bytes,extra,desc)
            if est['resident_capacity_bytes']<=self.budget_bytes:return {'local_R':r,'local_K':k,'description':desc,'estimate':est,'capacity':meta.capacity,'position_dtype':str(dtype),'sample_tokens':self._sample_n,'media_tokens':self._media_n,'Lloyd_iterations':8,'index_dtype':str(index_dtype),'norm_metadata':'none; effective full-vector centroids include magnitude','local_dictionary_payload_bytes':36*2*r*2*k*128*2,'local_dictionaries_paid_as_video_representation':True}
        raise RuntimeError('INFEASIBLE_BY_INTEGER_ORACLE_LOCAL_VIDEO_VQ')
    def _space(self,l,side,x):
        t=self.tables[0];return (x.float()-t[side+'_mean'][l].float()[:,None])/t[side+'_scale'][l].float()[:,None].clamp_min(1e-6)
    def _fit(self,l,side,state,raw):
        assert int(self._sample.max())<raw.shape[1];sample=self._space(l,side,raw.index_select(1,self._sample.long()));res=sample.clone();r=self.plan['local_R'];k=self.plan['local_K'];initial=torch.linspace(0,res.shape[1]-1,k,device=res.device).round().long()
        for stage in range(r):
            centers=res.index_select(1,initial).clone()
            for _ in range(8):
                ix=nearest(res,centers);one=torch.nn.functional.one_hot(ix,num_classes=k).float();counts=one.sum(1).unsqueeze(-1);means=torch.bmm(one.transpose(1,2),res)/counts.clamp_min(1);centers=torch.where(counts>0,means,centers)
            stored=centers.half();assert torch.isfinite(stored).all(),'local dictionary FP16 range invalid';state[side+'_local_books'][stage].copy_(stored)
            cb=state[side+'_local_books'][stage].float();ix=nearest(res,cb);res-=torch.gather(cb,1,ix[...,None].expand(-1,-1,128))
    def append(self,l,state,kpost,v,pre=None):
        assert pre is not None and pre.shape==kpost.shape;start=state['n'];n=v.shape[-2];assert start+n<=self.meta.capacity
        for side,raw in [('K',pre[0]),('V',v[0])]:
            if start==0:self._fit(l,side,state,raw)
            for b in range(0,n,2048):
                z=self._space(l,side,raw[:,b:b+2048]);nn=z.shape[1]
                for stage in range(self.plan['local_R']):
                    cb=state[side+'_local_books'][stage].float();ix=nearest(z,cb);state[side+'_index'][:,start+b:start+b+nn,stage].copy_(ix);z-=torch.gather(cb,1,ix[...,None].expand(-1,-1,128))
        state['n']+=n
        if l not in self._traced:
            rid=sys.argv[sys.argv.index('--run-id')+1];books={side:state[side+'_local_books'] for side in ['K','V']};record={'item_id':self._trace_item,'layer':l,'R':self.plan['local_R'],'K':self.plan['local_K'],'sample_tokens':self._sample_n,'actual_dictionary_tensor_bytes':sum(t.numel()*t.element_size() for t in books.values()),'actual_dictionary_SHA256':{s:hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for s,t in books.items()},'dictionary_placement':'paid cache/layer arena metadata, not fixed codec tables','first_append_start':start}
            with (ROOT/'data/runs'/rid/'local_dictionary_headers.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            self._traced.add(l)
        return state
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        n=state['n'];t=self.tables[0];out=[]
        for side in ['K','V']:
            z=torch.zeros(2,n,128,device='cuda')
            for stage in range(self.plan['local_R']):
                cb=state[side+'_local_books'][stage].float();ix=state[side+'_index'][:,:n,stage].long();z+=torch.gather(cb,1,ix[...,None].expand(-1,-1,128))
            z=z*t[side+'_scale'][l].float()[:,None]+t[side+'_mean'][l].float()[:,None];out.append(z.bfloat16()[None])
        phase=positions.float()[:,:,None]*inv_freq[None,None,:];cs=[];widths=[16,24,24]
        for trig in [torch.cos,torch.sin]:
            a=(trig(phase)*attention_scaling).bfloat16();z=torch.cat([a[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);cs.append(torch.cat([z,z],-1)[None,None])
        return apply_key_rope(out[0],*cs),out[1]
    def finish_request(self):self._sample=None;self._traced=set();super().finish_request()
class ProtectedLocalDictionary(PaidPrefixText,LocalVideoDictionary):pass
class HybridLocalDictionary(DenseFirst,ProtectedLocalDictionary):pass
