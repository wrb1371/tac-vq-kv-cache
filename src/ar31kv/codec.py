"""Frozen native7B GSRQ with a physically sliced long96 bank."""
import json,time,os,pathlib
import torch
from ar26kv.codec import Codec as Base
from ar26kv.runtime import ROOT
from r15kv.persistent_registry import Arena,PersistentRegistry
L,H,D=28,4,128
class BudgetGSRQ(Base):
    def __init__(self,name,model,item,pos,vi,ti,end,teacher):
        assert name in ['GSRQ-aware','GSRQ-export']
        self.name=name;self.item=item;self.end=end;self.q=int(item['query_reservation_tokens']);self.capacity=end+self.q
        self.budget=int(item['budget_mib']*2**20);self.registry=PersistentRegistry(self.budget);self.states={};self.table=None;self.model=model
        self._vi=vi;self._ti=ti;self._positions=pos[:,0,:end];self.M=len(vi);self.T=len(ti);self.grid=list(model._ar26_grid)
        start=time.time();admission=json.loads((ROOT/'data/ar31/GSRQ_ADMISSION.json').read_text());r=admission['Rmax_by_group'][item['duration_group']]
        family='GSRQAware'if name=='GSRQ-aware'else'GSRQExport';folder=pathlib.Path(os.environ.get('STIRVQ_BASELINE_ROOT',str(ROOT)))/f'artifacts/ar31/packages/{family}/R{r}'
        assert json.loads((folder/'completion.json').read_text())['completed']
        self.books=[{k:v.cuda()for k,v in torch.load(folder/f'layer{l:02}.pt',map_location='cpu',weights_only=True).items()}for l in range(L)]
        self.registry.own('codec/private',self.books,'codec');self.static=self.registry.end;self.max_private=r
        dense=self.static+L*4*H*D*self.capacity<=self.budget
        self.plan={'method':name,'budget_bytes':self.budget,'static_bytes':self.static,'hybrid_dense':dense,'geometry':[L,28,H,D],
            'query_slots':self.q,'media_original':self.M,'loaded_stages':r,'bank_folder':str(folder)}
        if dense:
            self.descs=[[('K_BF16',(H,self.capacity,D),torch.bfloat16,'payload'),('V_BF16',(H,self.capacity,D),torch.bfloat16,'payload')]]*L
        else:
            dtype=torch.int16 if int(pos.max())<=32767 else torch.int32
            coords=torch.empty((3,self.capacity),dtype=dtype,device='cuda');coords[:,:end].copy_(self._positions)
            rotary=model.model.language_model.rotary_emb
            self.global_maps={'positions':coords,'freq':rotary.inv_freq.detach().float().clone(),'video':vi.to(torch.int32),'text':ti.to(torch.int32)}
            self.scale=float(rotary.attention_scaling);self.registry.own('cache/maps',self.global_maps)
            base=self.registry.end+L*4*H*D*self.T;unit=L*H*self.capacity;maxsum=min(2*r,(self.budget-base)//unit)
            candidates=[(a,b)for a in range(1,r+1)for b in range(1,r+1)if a+b<=maxsum]
            if not candidates:raise RuntimeError('INFEASIBLE_PRIVATE_GSRQ')
            rk,rv=max(candidates,key=lambda ab:(sum(ab),-abs(ab[0]-ab[1]),ab[0]));self.plan.update(rK=rk,rV=rv)
            self.descs=[[('K_index',(H,self.capacity,rk),torch.uint8,'payload'),('V_index',(H,self.capacity,rv),torch.uint8,'payload'),
                ('text_FP_K',(H,self.T,D),torch.bfloat16,'payload'),('text_FP_V',(H,self.T,D),torch.bfloat16,'payload')]]*L
        self.plan['planning_seconds']=time.time()-start
        estimated=self.registry.end+sum(Arena.bytes(d)for d in self.descs);self.plan['estimated_resident_bytes']=estimated;assert estimated<=self.budget
        start=time.time()
        for l in range(L):
            arena=Arena(self.descs[l],'cuda',self.registry,f'cache/layer{l}');arena.tensor.zero_();self.states[l]={'arena':arena,**arena.views};self.encode(l,teacher.states[l])
        torch.cuda.synchronize();self.plan['encoding_seconds']=time.time()-start;self.receipt=self.registry.receipt();assert self.receipt['resident_bytes']==estimated
        self._vi=None;self._ti=None;self._positions=None

from functools import lru_cache
from ar30kv.codec import NativeControl,DT,sha
from ar30kv.planning import select,nbytes
@lru_cache(maxsize=1)
def nova_admission():return json.loads((ROOT/'data/ar31/NOVA_ADMISSION.json').read_text())
class BudgetNOVA(NativeControl):
    def __init__(self,name,model,item,pos,vi,ti,end,teacher):
        assert name in ['NOVA-KV-g8','NOVA-KV-g4']
        self.name=name;self.model=model;self.item=item;self.end=end;self.q=int(item['query_reservation_tokens']);self.capacity=end+self.q
        self.T=len(ti);self.M=len(vi);self.budget=int(item['budget_mib']*2**20);started=time.time()
        info=nova_admission()['videos'][item['video']];assert(info['media_end'],info['capacity'],info['protected_text'])==(end,self.capacity,self.T)
        chosen,options=select(name,self.capacity,self.T,info['position_bytes'],self.budget)
        if chosen is None:raise RuntimeError('INFEASIBLE_BY_INTEGER_ORACLE '+name+' '+str(min(z['resident_bytes']for z in options))+' > '+str(self.budget))
        assert info['methods'][name]['feasible']and chosen['resident_bytes']==info['methods'][name]['selected']['resident_bytes']
        self.plan={k:v for k,v in chosen.items()if k!='description'};self.plan.update(method=name,query_slots=self.q,media_original=self.M,geometry=[L,28,H,D])
        self.registry=PersistentRegistry(self.budget);self.states={};self.table=None
        directory=ROOT/f"artifacts/ar30/packages/NOVA_g{chosen['group_size']}_C{chosen['centroids']}";receipt=json.loads((directory/'completion.json').read_text());assert receipt['completed'];self.tables=[]
        for row in receipt['records']:
            p=directory/row['file'];assert sha(p)==row['SHA256'];self.tables.append({k:v.cuda()for k,v in torch.load(p,map_location='cpu',weights_only=True).items()})
        self.registry.own('codec/tables',self.tables,'codec');assert self.registry.end==chosen['codec_bytes']==receipt['static_tensor_bytes']
        self.plan.update(package=str(directory.relative_to(ROOT)),package_SHA256=sha(directory/'completion.json'),static_bytes=self.registry.end)
        self.global_maps={'text':ti.to(torch.int32)};self.migration=torch.zeros((2,H,min(chosen['recent'],max(0,self.capacity-chosen['sink'])),D),device='cuda',dtype=torch.bfloat16)
        self.registry.own('cache/migration',self.migration);self.registry.own('cache/maps',self.global_maps)
        desc=[(k,tuple(shape),DT[dtype],kind)for k,shape,dtype,kind in chosen['description']];assert Arena.bytes(desc)==nbytes(chosen['description'])
        self.plan['planning_seconds']=time.time()-started;started=time.time();ti_cpu=ti.cpu()
        for l in range(L):
            arena=Arena(desc,'cuda',self.registry,f'cache/layer{l}');arena.tensor.zero_();st={'arena':arena,**arena.views};self.states[l]=st
            self.encode(l,teacher.states[l]);st['text_FP_K'].copy_(teacher.states[l]['K_post'][:,ti_cpu].cuda());st['text_FP_V'].copy_(teacher.states[l]['V'][:,ti_cpu].cuda())
        torch.cuda.synchronize();self.plan['encoding_seconds']=time.time()-started;self.receipt=self.registry.receipt();assert self.receipt['resident_bytes']==chosen['resident_bytes'];self.plan['estimated_resident_bytes']=chosen['resident_bytes']
