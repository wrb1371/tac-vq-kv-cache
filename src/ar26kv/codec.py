"""Geometry-generic paid physical layouts; frozen STI policy and external transfers.

Read-only MC scoring reconstructs a layer before native attention. Workspace and
RAM teacher are evaluation infrastructure, never credited as resident capacity.
"""
import json,math,time,types,os,pathlib
import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from r15kv.persistent_registry import Arena,PersistentRegistry
from r14kv.fitting import nearest
from ar22kv.planner import allocate
from ar21kv.planner import allocate as depth_allocate
from ar26kv.runtime import ROOT
from ar26kv.matching import fixed_reference,references
from ar26kv.calibration import prefixes

L,H,D=28,4,128

def apply_rope(k,positions,freq,scale):
    phase=positions.float()[:,:,None]*freq[None,None,:];widths=[16,24,24];cs=[]
    for trig in [torch.cos,torch.sin]:
        v=(trig(phase)*scale).bfloat16();v=torch.cat([v[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);cs.append(torch.cat([v,v],-1)[None])
    rotated=torch.cat([-k[...,64:],k[...,:64]],-1)
    return (k*cs[0]+rotated*cs[1]).bfloat16()

class ReadCache(DynamicCache):
    def __init__(self,codec):super().__init__();self.codec=codec;self._seen_tokens=codec.end
    def get_seq_length(self,layer_idx=0):return self.codec.end
    def update(self,k,v,layer_idx,cache_kwargs=None):
        a,b=self.codec.decode(layer_idx)
        return torch.cat([a[None],k],2),torch.cat([b[None],v],2)

class Codec:
    def __init__(self,name,model,item,pos,vi,ti,end,teacher):
        self.name=name;self.item=item;self.end=end;self.q=int(item['query_reservation_tokens']);self.capacity=end+self.q;self.budget=int(item['budget_mib']*2**20);self.registry=PersistentRegistry(self.budget);self.states={};self.table=None;self.books=None;self.model=model
        self._vi=vi;self._ti=ti;self._positions=pos[ :,0,:end];self.M=len(vi);self.T=len(ti);self.grid=[int(v) for v in model._ar26_grid];self.static=0
        start=time.time()
        if name=='STI-RVQ':
            self.table={k:v.cuda() for k,v in torch.load(ROOT/'artifacts/ar26/packages/STI_RVQ/global.pt',map_location='cpu',weights_only=True).items()};self.registry.own('codec/shared',self.table,'codec');self.static=self.registry.end
        elif name.startswith('GSRQ'):
            r={16:1,32:3,64:5}[int(item['budget_mib'])];family='GSRQAware' if name=='GSRQ-aware' else 'GSRQExport';folder=pathlib.Path(os.environ.get('STIRVQ_BASELINE_ROOT',str(ROOT)))/'artifacts/ar26/packages'/family/f'R{r}';assert json.loads((folder/'completion.json').read_text())['completed']
            self.books=[{k:v.cuda() for k,v in torch.load(folder/f'layer{l:02}.pt',map_location='cpu',weights_only=True).items()} for l in range(L)];self.registry.own('codec/private',self.books,'codec');self.static=self.registry.end;self.max_private=r
        dense=self.static+L*4*H*D*self.capacity<=self.budget and name not in ['InfiniPot-V','CoRDS']
        self.plan={'method':name,'budget_bytes':self.budget,'static_bytes':self.static,'hybrid_dense':dense,'geometry':[L,28,H,D],'query_slots':self.q,'media_original':self.M}
        if dense:
            self.descs=[[('K_BF16',(H,self.capacity,D),torch.bfloat16,'payload'),('V_BF16',(H,self.capacity,D),torch.bfloat16,'payload')]]*L
        elif name in ['InfiniPot-V','CoRDS']:
            self.global_maps={'video':vi.to(torch.int32),'text':ti.to(torch.int32)};self.registry.own('cache/maps',self.global_maps)
            slots=min(self.capacity,(self.budget-4*(self.M+self.T))//(L*4*H*D));keep=min(self.M,slots-self.T-self.q)
            if keep<1:raise RuntimeError('INFEASIBLE_VIDEO_SELECTOR')
            self.plan.update(physical_slots=slots,media_kept_per_head=keep)
            self.descs=[[('K_BF16',(H,slots,D),torch.bfloat16,'payload'),('V_BF16',(H,slots,D),torch.bfloat16,'payload')]]*L
        else:
            dtype=torch.int16 if int(pos.max())<=32767 else torch.int32
            coords=torch.empty((3,self.capacity),dtype=dtype,device='cuda');coords[:,:end].copy_(self._positions)
            rotary=model.model.language_model.rotary_emb
            self.global_maps={'positions':coords,'freq':rotary.inv_freq.detach().float().clone(),'video':vi.to(torch.int32),'text':ti.to(torch.int32)};self.scale=float(rotary.attention_scaling);self.registry.own('cache/maps',self.global_maps)
            if name=='STI-RVQ':self.plan_sti(teacher)
            else:
                base=self.registry.end+L*4*H*D*self.T;unit=L*H*self.capacity;maxsum=min(2*self.max_private,(self.budget-base)//unit)
                candidates=[(a,b) for a in range(1,self.max_private+1) for b in range(1,self.max_private+1) if a+b<=maxsum]
                if not candidates:raise RuntimeError('INFEASIBLE_PRIVATE_GSRQ')
                rk,rv=max(candidates,key=lambda ab:(sum(ab),-abs(ab[0]-ab[1]),ab[0]));self.plan.update(rK=rk,rV=rv,loaded_stages=self.max_private)
                self.descs=[[('K_index',(H,self.capacity,rk),torch.uint8,'payload'),('V_index',(H,self.capacity,rv),torch.uint8,'payload'),('text_FP_K',(H,self.T,D),torch.bfloat16,'payload'),('text_FP_V',(H,self.T,D),torch.bfloat16,'payload')]]*L
        self.plan['planning_seconds']=time.time()-start
        estimated=self.registry.end+sum(Arena.bytes(d) for d in self.descs);self.plan['estimated_resident_bytes']=estimated;assert estimated<=self.budget
        started=time.time()
        for l in range(L):
            arena=Arena(self.descs[l],'cuda',self.registry,f'cache/layer{l}');arena.tensor.zero_();self.states[l]={'arena':arena,**arena.views};self.encode(l,teacher.states[l])
        torch.cuda.synchronize();self.plan['encoding_seconds']=time.time()-started;self.receipt=self.registry.receipt();assert self.receipt['resident_bytes']==estimated
        self._vi=None;self._ti=None;self._positions=None

    def plan_sti(self,teacher):
        self.curves=json.loads((ROOT/'data/ar26/VL_conditional_curves.json').read_text());rc=json.loads((ROOT/'data/ar26/VL_rate_curves.json').read_text())['plain'];c=self.curves['groups'][self.item['duration_group']]['norm'];nt,h,w=self.grid;p=h*w
        fixed=self.registry.end+L*(H*self.capacity+4*H*D*self.T);available=self.budget-fixed;pressure=L*4*H*D*(self.M+self.q)/max(1,available)
        floors=[(1,1)]*L
        a0=math.ceil(nt/8)*p;i0=math.ceil((self.M-a0)/8);e0=a0+i0+self.q;unit=H*e0*8;constant=fixed+L*H*e0*4;cap=(self.budget-constant)//unit
        if pressure>32:floors=[(2,2)]*L
        elif cap>=2*L:
            rates,_=depth_allocate([rc[s][l] for l in range(L) for s in ['K','V']],cap)
            floors=[(max(1,int(rates[2*l])-1),max(1,int(rates[2*l+1])-1)) for l in range(L)]
        # Before-arena layer0 content probe, exactly the old bounded ratio rule.
        raw={'K':teacher.states[0]['K_pre'].cuda().float(),'V':teacher.states[0]['V'].cuda().float()};ordinal=torch.arange(self.M,device='cuda');alpha={}
        for g in [2,4,8,16]:
            ao=ordinal[(ordinal//p)%g==0];to=ordinal[(ordinal//p)%g!=0];aa=ao[torch.linspace(0,len(ao)-1,min(48,len(ao)),device='cuda').long()];tt=to[torch.linspace(0,len(to)-1,min(48,len(to)),device='cuda').long()]
            codes,_=fixed_reference(raw,self._vi.to(torch.int32),tt,[self.table['K_scale'][0],self.table['V_scale'][0]],h,w,g);refs=self._vi[references(tt,codes,h,w,g)];alpha[str(g)]={}
            for s in ['K','V']:
                a=raw[s][:,self._vi[aa]];ref=raw[s].gather(1,refs[...,None].expand(-1,-1,D));da=prefixes(a,self.table,s,0)[-1].float();dr=prefixes(ref,self.table,s,0)[-1].float();metric=self.table[s+'_scale'][0].float().square().reciprocal()[:,None];al=float(((a-da).square()*metric).sum(-1).mean());tl=float(((raw[s][:,self._vi[tt]]-dr).square()*metric).sum(-1).mean());base=c[str(g)][s][0];relative=(tl/max(al,1e-12))/(base['target'][0][5]/max(base['anchor'][5],1e-12));alpha[str(g)][s]=max(.25,min(4.,relative))
        best=None
        for g in [2,4,8,16]:
            a=math.ceil(nt/g)*p;n=self.M-a;options=[]
            for l in range(L):
                oo=[]
                for fi,f in enumerate(self.curves['fractions']):
                    count=math.ceil(n*f);e=a+count+self.q
                    for rk in range(floors[l][0],7):
                        for rv in range(floors[l][1],7):
                            loss=0.
                            for s,r in [('K',rk),('V',rv)]:
                                z=c[str(g)][s][l];loss+=a*z['anchor'][r-1]+n*alpha[str(g)][s]*z['target'][fi][r-1]+self.q*z['query'][r-1]
                            oo.append({'bytes':H*e*(8*(rk+rv)+4),'loss':loss,'rK':rk,'rV':rv,'innovations':count,'slots':e,'fi':fi})
                options.append(oo)
            try:selected,report=allocate(options,available,4096)
            except RuntimeError:continue
            key=(report['objective'],report['actual_variable_bytes'],g)
            if best is None or key<best[0]:best=(key,g,a,selected,report)
        if best is None:raise RuntimeError('INFEASIBLE_JOINT_STI')
        _,g,a,selected,report=best;self.plan.update(gop=g,anchors=a,layer_choices=selected,allocation=report,pressure=pressure,floors=floors,content_alpha=alpha)
        self.descs=[]
        for z in selected:
            e=z['slots'];self.descs.append([(s+'_index',(H,e,z['r'+s],8),torch.uint8,'payload') for s in ['K','V']]+[(s+'_norm',(H,e,1),torch.bfloat16,'metadata') for s in ['K','V']]+[('T_code',(H,self.capacity),torch.uint8,'metadata')]+[(s,(H,self.T,D),torch.bfloat16,'payload') for s in ['text_FP_K','text_FP_V']])

    @torch.no_grad()
    def encode_product(self,l,s,x,state,slots,independent):
        t=self.table;index=state[s+'_index'];mean=t[s+'_mean'][l].float()[:,None] if independent else 0.;scale=t[s+'_scale'][l].float()[:,None]
        for b in range(0,x.shape[1],512):
            vals=x[:,b:b+512].float();u=(vals-mean)/scale;gain=u.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6).bfloat16();n=vals.shape[1];sl=slots[b:b+n];state[s+'_norm'][:,sl]=gain;z=((u/gain.float())@t['H']).reshape(1,-1,16)
            for r in range(index.shape[2]):
                book=t[s+'_books'][r][None].float();ids=nearest(z,book);index[:,sl,r]=ids.reshape(H,n,8).to(torch.uint8);z-=torch.gather(book,1,ids[...,None].expand(-1,-1,16))

    def read_product(self,l,s,state,n,independent,start=0):
        t=self.table;ix=state[s+'_index'][:,start:start+n];z=torch.zeros((H,n,8,16),device='cuda')
        for r in range(ix.shape[2]):z+=t[s+'_books'][r].float()[ix[:,:,r].long()]
        z=(z.reshape(H,n,D)@t['H'].T)*state[s+'_norm'][:,start:start+n].float()*t[s+'_scale'][l].float()[:,None]
        if independent:z+=t[s+'_mean'][l].float()[:,None]
        return z.bfloat16()

    def select(self,k,v):
        from ar19kv.video_prune import reference_module
        keep=self.plan['media_kept_per_head'];nt,h,w=self.grid;p=h*w
        if keep==self.M:return torch.arange(self.M,device='cuda')[None].expand(H,-1)
        recent=min(int(nt*.25),(keep-round(keep*.5))//p);qr=(recent+1e-8)/nt if recent else 0.;tar=.5 if recent else 0.;dummy=types.SimpleNamespace(config=types.SimpleNamespace(height_width=h),model=types.SimpleNamespace(language_model=types.SimpleNamespace(layers=[None])))
        if self.name=='InfiniPot-V':
            official=reference_module('infinipot');layer=types.SimpleNamespace(keys=k[None],values=v[None])
            class Cache:
                layers=[layer]
                def get_seq_length(self):return self.layers[0].keys.shape[-2]
            cache,_=official.process_kv_cache(Cache(),dummy,0,0,p,keep/p,method='infinipot-v',tar_ratio=tar,query_ratio=qr,adaptive_pooling=False);ix=cache.selection_indices
        else:ix=reference_module('cords')._cords_select_indices(k[None],v[None],H,p,nt,keep/p,tar,qr,-1,1,dummy,8,'cords',.5,1.0,'maxcos',.25,0.0,1.0,'none',False,1)
        assert ix.shape==(H,keep);return ix

    @torch.no_grad()
    def encode(self,l,native):
        st=self.states[l];vi=self._vi;ti=self._ti;kpost=native['K_post'].cuda();v=native['V'].cuda()
        if self.plan['hybrid_dense']:
            st['K_BF16'][:,:self.end].copy_(kpost);st['V_BF16'][:,:self.end].copy_(v);return
        if self.name in ['InfiniPot-V','CoRDS']:
            ix=self.select(kpost[:,vi],v[:,vi]);order=torch.cat([ti[None].expand(H,-1),vi[ix]],1).sort(1).values;st['n']=order.shape[1]
            for s,values in [('K',kpost),('V',v)]:st[s+'_BF16'][:,:st['n']].copy_(values.gather(1,order[...,None].expand(-1,-1,D)))
            return
        st['text_FP_K'].copy_(kpost[:,ti]);st['text_FP_V'].copy_(v[:,ti]);raw={'K':native['K_pre'].cuda().float(),'V':v.float()}
        if self.name.startswith('GSRQ'):
            for s in ['K','V']:
                idx=st[s+'_index'];book=self.books[l][s]
                for b in range(0,self.end,1024):
                    z=raw[s][:,b:b+1024].clone()
                    for r in range(idx.shape[2]):
                        cb=book[:,r].float();ids=nearest(z,cb);idx[:,b:b+z.shape[1],r]=ids.to(torch.uint8);z-=torch.gather(cb,1,ids[...,None].expand(-1,-1,D))
            return
        nt,h,w=self.grid;g=self.plan['gop'];p=h*w;o=torch.arange(self.M,device='cuda');ao=o[(o//p)%g==0];to=o[(o//p)%g!=0];a=len(ao);nr=self.plan['layer_choices'][l]['innovations'];decoded={}
        st['T_code'].zero_()
        for s in ['K','V']:
            self.encode_product(l,s,raw[s][:,vi[ao]],st,torch.arange(a,device='cuda'),True);decoded[s]=self.read_product(l,s,st,a,True)
        codes,scores=fixed_reference(raw,vi.to(torch.int32),to,[self.table['K_scale'][l],self.table['V_scale'][l]],h,w,g);st['T_code'][:,vi[to]]=codes
        if nr:
            picked=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:nr].sort(1).values;chosen=to[picked];reford=references(to,codes,h,w,g).gather(1,picked);refslot=(reford//p//g)*p+reford%p;heads=torch.arange(H,device='cuda')[:,None];st['T_code'][heads,vi[chosen]]=codes.gather(1,picked)|128
            for s in ['K','V']:
                current=raw[s].gather(1,vi[chosen][...,None].expand(-1,-1,D));pred=decoded[s].gather(1,refslot[...,None].expand(-1,-1,D)).float();self.encode_product(l,s,current-pred,st,torch.arange(a,a+nr,device='cuda'),False)

    @torch.no_grad()
    def decode(self,l):
        st=self.states[l]
        if self.plan['hybrid_dense']:return st['K_BF16'][:,:self.end],st['V_BF16'][:,:self.end]
        if self.name in ['InfiniPot-V','CoRDS']:return st['K_BF16'][:,:st['n']],st['V_BF16'][:,:st['n']]
        result={};ti=self.global_maps['text'].long()
        if self.name.startswith('GSRQ'):
            for s in ['K','V']:
                ix=st[s+'_index'][:,:self.end];out=torch.zeros((H,self.end,D),device='cuda')
                for r in range(ix.shape[2]):out+=torch.gather(self.books[l][s][:,r].float(),1,ix[:,:,r].long()[...,None].expand(-1,-1,D))
                result[s]=out.bfloat16()
        else:
            nt,h,w=self.grid;p=h*w;g=self.plan['gop'];o=torch.arange(self.M,device='cuda');ao=o[(o//p)%g==0];to=o[(o//p)%g!=0];vi=self.global_maps['video'].long();code=st['T_code'][:,vi[to]];ref=references(to,code&127,h,w,g);slot=(ref//p//g)*p+ref%p;a=self.plan['anchors'];nr=self.plan['layer_choices'][l]['innovations'];mask=(code&128)!=0
            for s in ['K','V']:
                anchors=self.read_product(l,s,st,a,True);out=torch.empty((H,self.end,D),dtype=torch.bfloat16,device='cuda');out[:,vi[ao]]=anchors;prediction=anchors.gather(1,slot[...,None].expand(-1,-1,D))
                if nr:
                    innovation=self.read_product(l,s,st,nr,False,start=a)
                    for head in range(H):
                        assert int(mask[head].sum())==nr;prediction[head,mask[head]]=(prediction[head,mask[head]].float()+innovation[head].float()).bfloat16()
                out[:,vi[to]]=prediction;result[s]=out
        result['K']=apply_rope(result['K'],self.global_maps['positions'][:,:self.end],self.global_maps['freq'],self.scale)
        result['K'][:,ti]=st['text_FP_K'];result['V'][:,ti]=st['text_FP_V'];return result['K'],result['V']

    def cache(self):return ReadCache(self)
