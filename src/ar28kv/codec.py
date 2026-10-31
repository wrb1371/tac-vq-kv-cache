"""Frozen-bank model-aware planning, including unpadded per-physical-head layouts."""
import hashlib,json,math
import torch
from ar26kv.codec import Codec as BaseCodec,apply_rope
from ar26kv.runtime import ROOT
from ar26kv.matching import fixed_reference,references
from ar22kv.planner import allocate
from r14kv.fitting import nearest

L,H,D=28,4,128

class ModelAwareCodec(BaseCodec):
    def __init__(self,variant,policy,model,item,pos,vi,ti,end,teacher):
        self.variant=variant;self.policy=policy;self.headwise=bool(policy.get('headwise',False))
        effective={**item,'budget_mib':float(item['budget_mib'])*policy.get('budget_multiplier',1.)}
        super().__init__('STI-RVQ',model,effective,pos,vi,ti,end,teacher)
        self.plan.update(variant=variant,eligible=policy.get('eligible',True),declared_original_budget_MiB=item['budget_mib'])

    def plan_sti(self,teacher):
        # Reuse the exact original content observation and precision floor.
        # The preliminary plan is CPU-only; no layer arena exists here.
        super().plan_sti(teacher)
        p=self.grid[1]*self.grid[2];nt=self.grid[0];floors=self.plan['floors']
        if self.policy.get('floor_mode')=='min2':floors=[(2,2)]*L
        floors=[(max(k,self.policy.get('min_k',1)),max(v,self.policy.get('min_v',1))) for k,v in floors]
        alpha=self.plan['content_alpha'] if self.policy.get('probe',True) else {str(g):{'K':1.,'V':1.} for g in [2,4,8,16]}
        if self.headwise:
            path=ROOT/'data/ar28/VL_head_curves.json';head=json.loads(path.read_text());assert head['completed']
            curves=head['groups'][self.item['duration_group']]['norm'];metadata_bytes=L*H*3*4+4
            self.plan['head_curves_SHA256']=hashlib.sha256(path.read_bytes()).hexdigest()
        else:curves=self.curves['groups'][self.item['duration_group']]['norm'];metadata_bytes=0
        fixed=self.registry.end+L*(H*self.capacity+4*H*D*self.T)+metadata_bytes
        available=self.budget-fixed;best=None
        for g in self.policy.get('gops',[2,4,8,16]):
            a=math.ceil(nt/g)*p;n=self.M-a;options=[]
            for l in range(L):
                for hh in range(H if self.headwise else 1):
                    choices=[]
                    for fi,fraction in enumerate(self.curves['fractions']):
                        count=math.ceil(n*fraction);e=a+count+self.q
                        for rk in range(floors[l][0],7):
                            for rv in range(floors[l][1],7):
                                loss=0.
                                for s,r,weight in [('K',rk,self.policy.get('key_weight',1.)),('V',rv,self.policy.get('value_weight',1.))]:
                                    z=curves[str(g)][s][l]
                                    anc=z['anchor'][hh] if self.headwise else z['anchor']
                                    tgt=z['target'][hh] if self.headwise else z['target']
                                    qry=z['query'][hh] if self.headwise else z['query']
                                    loss+=weight*(a*anc[r-1]+n*alpha[str(g)][s]*tgt[fi][r-1]+self.q*qry[r-1])
                                choices.append({'bytes':(1 if self.headwise else H)*e*(8*(rk+rv)+4),
                                    'loss':loss,'rK':rk,'rV':rv,'innovations':count,'slots':e,'fi':fi})
                    options.append(choices)
            try:selected,report=allocate(options,available,1024 if self.headwise else 4096)
            except RuntimeError:continue
            key=(report['objective'],report['actual_variable_bytes'],g)
            if best is None or key<best[0]:best=(key,g,a,selected,report)
        if best is None:raise RuntimeError('INFEASIBLE_MODEL_AWARE_PLANNER')
        _,g,a,selected,report=best
        self.plan.update(gop=g,anchors=a,allocation=report,floors=floors,content_alpha=alpha,
                         headwise=self.headwise,head_metadata_bytes=metadata_bytes,model_aware_policy=self.policy)
        self.descs=[]
        if not self.headwise:
            self.plan['layer_choices']=selected
            for z in selected:
                e=z['slots'];self.descs.append([(s+'_index',(H,e,z['r'+s],8),torch.uint8,'payload') for s in ['K','V']]+
                    [(s+'_norm',(H,e,1),torch.bfloat16,'metadata') for s in ['K','V']]+
                    [('T_code',(H,self.capacity),torch.uint8,'metadata')]+
                    [(s,(H,self.T,D),torch.bfloat16,'payload') for s in ['text_FP_K','text_FP_V']])
        else:
            layers=[selected[l*H:(l+1)*H] for l in range(L)];self.plan['layer_head_choices']=layers
            self.plan.pop('layer_choices',None)
            header=torch.tensor([[[z['rK'],z['rV'],z['innovations']] for z in ll] for ll in layers],dtype=torch.int32,device='cuda')
            self.head_header={'rates_and_counts':header,'gop':torch.tensor([g],dtype=torch.int32,device='cuda')}
            self.registry.own('cache/head_plan',self.head_header)
            for ll in layers:
                desc=[]
                for h,z in enumerate(ll):
                    e=z['slots']
                    desc += [(s+f'_index_h{h}',(1,e,z['r'+s],8),torch.uint8,'payload') for s in ['K','V']]
                    desc += [(s+f'_norm_h{h}',(1,e,1),torch.bfloat16,'metadata') for s in ['K','V']]
                desc += [('T_code',(H,self.capacity),torch.uint8,'metadata')]
                desc += [(s,(H,self.T,D),torch.bfloat16,'payload') for s in ['text_FP_K','text_FP_V']]
                self.descs.append(desc)

    @torch.no_grad()
    def encode_h(self,l,h,s,x,st,slots,independent):
        t=self.table;index=st[s+f'_index_h{h}'];mean=t[s+'_mean'][l,h:h+1].float()[:,None] if independent else 0.
        scale=t[s+'_scale'][l,h:h+1].float()[:,None]
        for b in range(0,x.shape[1],512):
            vals=x[:,b:b+512].float();u=(vals-mean)/scale
            gain=u.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6).bfloat16();n=vals.shape[1];sl=slots[b:b+n]
            st[s+f'_norm_h{h}'][:,sl]=gain;z=((u/gain.float())@t['H']).reshape(1,-1,16)
            for rr in range(index.shape[2]):
                book=t[s+'_books'][rr][None].float();ids=nearest(z,book)
                index[:,sl,rr]=ids.reshape(1,n,8).to(torch.uint8)
                z-=torch.gather(book,1,ids[...,None].expand(-1,-1,16))

    def read_h(self,l,h,s,st,n,independent,start=0):
        t=self.table;ix=st[s+f'_index_h{h}'][:,start:start+n]
        z=torch.zeros((1,n,8,16),device='cuda')
        for rr in range(ix.shape[2]):z+=t[s+'_books'][rr].float()[ix[:,:,rr].long()]
        z=(z.reshape(1,n,D)@t['H'].T)*st[s+f'_norm_h{h}'][:,start:start+n].float()*t[s+'_scale'][l,h:h+1].float()[:,None]
        if independent:z+=t[s+'_mean'][l,h:h+1].float()[:,None]
        return z.bfloat16()

    @torch.no_grad()
    def encode(self,l,native):
        if not self.headwise or self.plan['hybrid_dense']:return super().encode(l,native)
        st=self.states[l];vi=self._vi;ti=self._ti
        kp=native['K_post'].cuda();v=native['V'].cuda()
        st['text_FP_K'].copy_(kp[:,ti]);st['text_FP_V'].copy_(v[:,ti])
        raw={'K':native['K_pre'].cuda().float(),'V':v.float()}
        nt,gh,gw=self.grid;p=gh*gw;g=self.plan['gop'];o=torch.arange(self.M,device='cuda')
        ao=o[(o//p)%g==0];to=o[(o//p)%g!=0];a=len(ao)
        codes,scores=fixed_reference(raw,vi.to(torch.int32),to,[self.table['K_scale'][l],self.table['V_scale'][l]],gh,gw,g)
        ref=references(to,codes,gh,gw,g);st['T_code'].zero_()
        for h,z in enumerate(self.plan['layer_head_choices'][l]):
            anchors={}
            for s in ['K','V']:
                self.encode_h(l,h,s,raw[s][h:h+1,vi[ao]],st,torch.arange(a,device='cuda'),True)
                anchors[s]=self.read_h(l,h,s,st,a,True)
            st['T_code'][h,vi[to]]=codes[h];nr=z['innovations']
            if not nr:continue
            picked=torch.argsort(scores[h],descending=True,stable=True)[:nr].sort().values
            chosen=to[picked];reford=ref[h,picked];refslot=(reford//p//g)*p+reford%p
            st['T_code'][h,vi[chosen]]=codes[h,picked]|128
            for s in ['K','V']:
                current=raw[s][h:h+1,vi[chosen]];prediction=anchors[s][:,refslot].float()
                self.encode_h(l,h,s,current-prediction,st,torch.arange(a,a+nr,device='cuda'),False)

    @torch.no_grad()
    def decode(self,l):
        if not self.headwise or self.plan['hybrid_dense']:return super().decode(l)
        st=self.states[l];vi=self.global_maps['video'].long();ti=self.global_maps['text'].long()
        nt,gh,gw=self.grid;p=gh*gw;g=self.plan['gop'];o=torch.arange(self.M,device='cuda')
        ao=o[(o//p)%g==0];to=o[(o//p)%g!=0];codes=st['T_code'][:,vi[to]]
        ref=references(to,codes&127,gh,gw,g);refslots=(ref//p//g)*p+ref%p;a=self.plan['anchors']
        result={s:torch.empty((H,self.end,D),dtype=torch.bfloat16,device='cuda') for s in ['K','V']}
        for h,z in enumerate(self.plan['layer_head_choices'][l]):
            nr=z['innovations'];mask=(codes[h]&128)!=0;assert int(mask.sum())==nr
            for s in ['K','V']:
                anchor=self.read_h(l,h,s,st,a,True);pred=anchor[:,refslots[h]]
                if nr:
                    innovation=self.read_h(l,h,s,st,nr,False,start=a)
                    pred[:,mask]=(pred[:,mask].float()+innovation.float()).bfloat16()
                result[s][h,vi[ao]]=anchor[0];result[s][h,vi[to]]=pred[0]
        result['K']=apply_rope(result['K'],self.global_maps['positions'][:,:self.end],self.global_maps['freq'],self.scale)
        result['K'][:,ti]=st['text_FP_K'];result['V'][:,ti]=st['text_FP_V']
        return result['K'],result['V']
