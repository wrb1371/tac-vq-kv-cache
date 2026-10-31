"""Paid BF16 selected-prefix arenas with full physical query capacity reserved."""
import time
import torch
from transformers.cache_utils import DynamicCache
from r15kv.persistent_registry import Arena,PersistentRegistry
from .selection import select

class ReadCache(DynamicCache):
    def __init__(self,c):super().__init__();self.codec=c;self._seen_tokens=c.end
    def get_seq_length(self,layer_idx=0):return self.codec.end
    def update(self,k,v,layer_idx,cache_kwargs=None):
        a,b=self.codec.decode(layer_idx);return torch.cat([a[None],k],2),torch.cat([b[None],v],2)

class Codec:
    @torch.no_grad()
    def __init__(self,method,item,vi,ti,teacher,gate=False):
        self.name=method;self.end=teacher.end;self.L,self.N,self.H,self.D=teacher.geo;L,N,H,D=teacher.geo
        self.q=int(item['query_reservation_tokens']);self.budget=int(item['budget_mib']*2**20);self.registry=PersistentRegistry(self.budget);self.states={};self.selected={};self.maps={'video':vi.to(torch.int32),'text':ti.to(torch.int32)};self.registry.own('cache/maps',self.maps)
        M,T=len(vi),len(ti);slots=min(self.end+self.q,(self.budget-4*(M+T))//(L*4*H*D));keep=min(M,slots-T-self.q)
        if keep<74:raise RuntimeError('INFEASIBLE_PUBLISHED_PROTECTED_ZONES')
        desc=[(s,(H,slots,D),torch.bfloat16,'payload')for s in ['K_BF16','V_BF16']];start=time.time()
        self.plan={'method':method,'geometry':list(teacher.geo),'physical_slots':slots,'media_kept_per_head':keep,'media_original':M,'protected_text':T,'query_slots':self.q,'codec_static_bytes':0,'maps_bytes':4*(M+T),'independent_byte_formula':4*(M+T)+L*4*H*D*slots,'question_used':False,'selection_seconds':0.}
        for l,s in teacher.states.items():
            a=Arena(desc,'cuda',self.registry,f'cache/layer{l}');a.tensor.zero_();k=s['K_post'].cuda();v=s['V'].cuda();q=teacher.queries[l].cuda();tick=time.time()
            chosen=select(method,k,q,teacher.rows,vi,keep,N//H,self.end,teacher.anchor_rows)
            torch.cuda.synchronize();self.plan['selection_seconds']+=time.time()-tick
            order=torch.cat([chosen,ti[None].expand(H,-1)],1).sort(1).values;n=order.shape[1]
            assert all(len(torch.unique(z))==n for z in order)
            st={'arena':a,**a.views,'n':n}
            for label,vals in [('K_BF16',k),('V_BF16',v)]:st[label][:,:n].copy_(vals.gather(1,order[...,None].expand(-1,-1,D)))
            self.states[l]=st;self.selected[l]=order.cpu()
        torch.cuda.synchronize();self.plan['encoding_seconds_including_selection']=time.time()-start;self.receipt=self.registry.receipt()
        assert self.receipt['resident_bytes']==self.plan['independent_byte_formula']<=self.budget
    def decode(self,l):
        st=self.states[l];return st['K_BF16'][:,:st['n']],st['V_BF16'][:,:st['n']]
    def cache(self):return ReadCache(self)
