"""Allocate and charge every physical selected-KV slot, including query reserve."""
import time
import torch
from r15kv.persistent_registry import Arena,PersistentRegistry
from ar32kv.codec import ReadCache
from .selection import select,layer_quotas

class Codec:
    @torch.no_grad()
    def __init__(self,method,item,vi,ti,teacher,gate=False):
        self.name=method;self.end=teacher.end;self.L,self.N,self.H,self.D=teacher.geo;L,N,H,D=teacher.geo
        self.q=int(item['query_reservation_tokens']);self.budget=int(item['budget_mib']*2**20);self.registry=PersistentRegistry(self.budget);self.states={};self.selected={}
        self.maps={'video':vi.to(torch.int32),'text':ti.to(torch.int32)};self.registry.own('cache/maps',self.maps)
        M,T=len(vi),len(ti);slot_total=min(L*(self.end+self.q),(self.budget-4*(M+T))//(4*H*D));video_total=slot_total-L*(T+self.q)
        recent=int((vi>=max(0,self.end-64)).sum())
        if method=='PyramidKV':keeps=layer_quotas(video_total,L,M,recent)
        else:keeps=[min(M,video_total//L)]*L
        slots=[k+T+self.q for k in keeps];assert min(keeps)>=4
        start=time.time();self.plan={'method':method,'geometry':list(teacher.geo),'physical_slots_per_layer':slots,'media_kept_per_layer':keeps,'media_original':M,'protected_text':T,'query_slots':self.q,'codec_static_bytes':0,'maps_bytes':4*(M+T),'independent_byte_formula':4*(M+T)+4*H*D*sum(slots),'question_used':False,'selection_seconds':0.,'future_native_text_position_start':teacher.future_start,'statistics_rows':self.end-4,'future_positions':512}
        for l,s in teacher.states.items():
            a=Arena([(v,(H,slots[l],D),torch.bfloat16,'payload')for v in ['K_BF16','V_BF16']],'cuda',self.registry,f'cache/layer{l}');a.tensor.zero_()
            k=s['K_post'].cuda();v=s['V'].cuda();q=teacher.queries[l].cuda();tick=time.time()
            chosen=select(method,k,v,q,teacher.statistics[l],vi,keeps[l],N//H,self.end)
            torch.cuda.synchronize();self.plan['selection_seconds']+=time.time()-tick
            order=torch.cat([chosen,ti[None].expand(H,-1)],1).sort(1).values;n=order.shape[1];assert all(len(z.unique())==n for z in order)
            st={'arena':a,**a.views,'n':n}
            for name,z in [('K_BF16',k),('V_BF16',v)]:st[name][:,:n].copy_(z.gather(1,order[...,None].expand(-1,-1,D)))
            self.states[l]=st;self.selected[l]=order.cpu()
        torch.cuda.synchronize();self.plan['encoding_seconds_including_selection']=time.time()-start;self.receipt=self.registry.receipt()
        assert self.receipt['resident_bytes']==self.plan['independent_byte_formula']<=self.budget
    def decode(self,l):
        st=self.states[l];return st['K_BF16'][:,:st['n']],st['V_BF16'][:,:st['n']]
    def cache(self):return ReadCache(self)
