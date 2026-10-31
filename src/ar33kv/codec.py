"""Frozen shared codec, unchanged physical format, explicit ablation policies."""
import copy,json,math
from functools import lru_cache
import torch
from ar26kv.codec import Codec as Base,ROOT,L,H,D
from ar26kv.matching import fixed_reference,references
from ar21kv.dynamic_kernels import _match9
from ar22kv.planner import allocate
from ar21kv.planner import allocate as depth_allocate

@lru_cache(maxsize=1)
def tables():return json.loads((ROOT/'data/ar26/VL_conditional_curves.json').read_text()),json.loads((ROOT/'data/ar26/VL_rate_curves.json').read_text())['plain']

class Ablation(Base):
    def __init__(self,profile,model,item,pos,vi,ti,end,teacher,parent=None):
        self.profile=profile;self.parent=parent
        super().__init__('STI-RVQ',model,item,pos,vi,ti,end,teacher)
        self.plan.update(profile=profile,ablation_source='AR33 native7B migration; no fit/QA tuning',quality_shared_parent_template_not_deployment_timing=True)

    def descriptions(self,choices):
        return [[(s+'_index',(H,z['slots'],z['r'+s],8),torch.uint8,'payload')for s in ['K','V']]+[(s+'_norm',(H,z['slots'],1),torch.bfloat16,'metadata')for s in ['K','V']]+[('T_code',(H,self.capacity),torch.uint8,'metadata')]+[(s,(H,self.T,D),torch.bfloat16,'payload')for s in ['text_FP_K','text_FP_V']]for z in choices]

    def plan_sti(self,teacher):
        if self.profile=='A25_S000_RO':return super().plan_sti(teacher)
        assert self.parent is not None
        self.curves,rates=tables();p=self.grid[1]*self.grid[2];nt=self.grid[0]
        self.plan.update(copy.deepcopy(self.parent['plan']))
        if self.profile in ['NoDecodedFeedback','NoLocalMatch']:
            self.descs=copy.deepcopy(self.parent['descs']);return
        if self.profile in ['NoInnovation','Independent']:
            g=8 if self.profile=='NoInnovation'else 1;a=math.ceil(nt/g)*p;e=a+self.q
            fixed=self.registry.end+L*(H*self.capacity+4*H*D*self.T)
            unit=H*e*8;constant=fixed+L*H*e*4;cap=(self.budget-constant)//unit
            options=[(k,v)for k in range(1,7)for v in range(1,7)if L*(k+v)<=cap]
            if not options:raise RuntimeError('INFEASIBLE_REPRESENTATION_CONTROL')
            rk,rv=max(options,key=lambda x:(sum(x),-abs(x[0]-x[1]),x[0]))
            rr,obj=depth_allocate([rates[s][l]for l in range(L)for s in ['K','V']],cap,lower=[rk,rv]*L)
            choices=[{'rK':rr[2*l],'rV':rr[2*l+1],'innovations':0,'slots':e,'bytes':H*e*(8*(rr[2*l]+rr[2*l+1])+4),'fi':0}for l in range(L)]
            self.plan.update(gop=g,anchors=a,layer_choices=choices,allocation={'objective':obj,'actual_variable_bytes':sum(x['bytes']for x in choices),'scope':'legacy representation control: native calibrated depth refinement, max-feasible balanced lower bound'},representation_control=self.profile,legacy_balanced_depth=[rk,rv])
            self.descs=self.descriptions(choices);return
        floors=self.parent['plan']['floors']
        if self.profile in ['JointContent','UniformJoint']:floors=[(1,1)]*L
        alpha=self.parent['plan']['content_alpha']
        if self.profile=='AR24_NoProbe':alpha={str(g):{'K':1.,'V':1.}for g in [2,4,8,16]}
        c=self.curves['groups'][self.item['duration_group']]['norm'];f0=self.curves['fractions'].index(.125)
        fixed=self.registry.end+L*(H*self.capacity+4*H*D*self.T);available=self.budget-fixed;best=None
        gs=[8]if self.profile in ['A25_S100','FixedGIStableProbe']else [2,4,8,16]
        fis=[f0]if self.profile in ['FixedIStableProbe','FixedGIStableProbe']else list(range(len(self.curves['fractions'])))
        for g in gs:
            a=math.ceil(nt/g)*p;n=self.M-a;options=[]
            for l in range(L):
                oo=[]
                for fi in fis:
                    count=math.ceil(n*self.curves['fractions'][fi]);e=a+count+self.q
                    for rk in range(floors[l][0],7):
                        for rv in range(floors[l][1],7):
                            loss=0.
                            for s,r in [('K',rk),('V',rv)]:
                                z=c[str(g)][s][l];target=z['target'][fi][r-1]
                                if self.profile=='AR24_Additive':target=z['target'][f0][r-1]+z['target'][fi][5]-z['target'][f0][5]
                                loss+=a*z['anchor'][r-1]+n*alpha[str(g)][s]*target+self.q*z['query'][r-1]
                            oo.append({'bytes':H*e*(8*(rk+rv)+4),'loss':loss,'rK':rk,'rV':rv,'innovations':count,'slots':e,'fi':fi})
                options.append(oo)
            try:
                if self.profile=='UniformJoint':
                    legal=[(sum(o[j]['loss']for o in options),sum(o[j]['bytes']for o in options),j)for j in range(len(options[0]))if sum(o[j]['bytes']for o in options)<=available]
                    if not legal:continue
                    obj,cost,j=min(legal);selected=[o[j]for o in options];report={'objective':obj,'actual_variable_bytes':cost,'scope':'complete uniform-across-layer joint enumeration; no floor'}
                else:selected,report=allocate(options,available,4096)
            except RuntimeError as exc:
                if 'INFEASIBLE'not in str(exc):raise
                continue
            key=(report['objective'],report['actual_variable_bytes'],g)
            if best is None or key<best[0]:best=(key,g,a,selected,report)
        if best is None:raise RuntimeError('INFEASIBLE_ABLATION_PLANNER')
        _,g,a,selected,report=best
        self.plan.update(gop=g,anchors=a,layer_choices=selected,allocation=report,floors=floors,content_alpha=alpha,objective='separable fref1/8,rref6'if self.profile=='AR24_Additive'else'original conditional')
        self.descs=self.descriptions(selected)

    @torch.no_grad()
    def encode(self,l,native):
        special=self.profile in ['NoDecodedFeedback','NoLocalMatch','NoInnovation']
        if not special or self.plan['hybrid_dense']:return super().encode(l,native)
        st=self.states[l];vi=self._vi;ti=self._ti;kpost=native['K_post'].cuda();v=native['V'].cuda()
        st['text_FP_K'].copy_(kpost[:,ti]);st['text_FP_V'].copy_(v[:,ti]);raw={'K':native['K_pre'].cuda().float(),'V':v.float()}
        nt,h,w=self.grid;g=self.plan['gop'];p=h*w;o=torch.arange(self.M,device='cuda');ao=o[(o//p)%g==0];to=o[(o//p)%g!=0];a=len(ao);nr=self.plan['layer_choices'][l]['innovations'];decoded={}
        st['T_code'].zero_()
        for s in ['K','V']:
            self.encode_product(l,s,raw[s][:,vi[ao]],st,torch.arange(a,device='cuda'),True);decoded[s]=self.read_product(l,s,st,a,True)
        if self.profile=='NoLocalMatch':
            codes=torch.full((H,len(to)),4,device='cuda',dtype=torch.uint8);reford=references(to,codes,h,w,g);scores=torch.zeros((H,len(to)),device='cuda')
            for s in ['K','V']:
                pred=raw[s].gather(1,vi[reford][...,None].expand(-1,-1,D));scores+=((raw[s][:,vi[to]]-pred)/self.table[s+'_scale'][l].float()[:,None]).square().mean(-1)
        elif self.profile=='NoInnovation':
            scratch={s:torch.zeros_like(raw[s])for s in ['K','V']}
            for s in ['K','V']:scratch[s][:,vi[ao]]=decoded[s].float()
            codes=torch.empty((H,len(to)),device='cuda',dtype=torch.uint8);scores=torch.empty((H,len(to)),device='cuda')
            _match9[(len(to),H)](raw['K'].contiguous(),raw['V'].contiguous(),scratch['K'].contiguous(),scratch['V'].contiguous(),vi.to(torch.int32),self.table['K_scale'][l],self.table['V_scale'][l],to,codes,scores,self.end,len(to),h,w,g,num_warps=4,enable_fp_fusion=False)
        else:codes,scores=fixed_reference(raw,vi.to(torch.int32),to,[self.table['K_scale'][l],self.table['V_scale'][l]],h,w,g)
        st['T_code'][:,vi[to]]=codes
        if nr:
            picked=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:nr].sort(1).values;chosen=to[picked];reford=references(to,codes,h,w,g).gather(1,picked);refslot=(reford//p//g)*p+reford%p;heads=torch.arange(H,device='cuda')[:,None];st['T_code'][heads,vi[chosen]]=codes.gather(1,picked)|128
            for s in ['K','V']:
                current=raw[s].gather(1,vi[chosen][...,None].expand(-1,-1,D))
                pred=raw[s].gather(1,vi[reford][...,None].expand(-1,-1,D))if self.profile=='NoDecodedFeedback'else decoded[s].gather(1,refslot[...,None].expand(-1,-1,D)).float()
                self.encode_product(l,s,current-pred,st,torch.arange(a,a+nr,device='cuda'),False)
