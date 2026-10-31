"""Real unpadded A/I/Q arrays with frozen-B0 query rates and role depths."""
import hashlib,json,math
import torch
from ar26kv.codec import Codec as Base,apply_rope
from ar26kv.runtime import ROOT
from ar26kv.matching import fixed_reference,references
from ar22kv.planner import allocate
from r14kv.fitting import nearest
from .calibration import PATTERNS
L,H,D=28,4,128

class RoleCodec(Base):
    def __init__(self,variant,policy,model,item,pos,vi,ti,end,teacher):
        self.variant=variant;self.policy=policy
        super().__init__('STI-RVQ',model,item,pos,vi,ti,end,teacher)
        self.plan.update(variant=variant,role_policy=policy)

    def plan_sti(self,teacher):
        super().plan_sti(teacher)
        original=[(z['rK'],z['rV'])for z in self.plan['layer_choices']]
        floors=self.plan['floors'];alpha=self.plan['content_alpha']
        path=ROOT/'data/ar28role/ROLE_CURVES.json';curves=json.loads(path.read_text());assert curves['completed']
        head=json.loads((ROOT/'data/ar28/VL_head_curves.json').read_text());group=self.item['duration_group']
        headwise=self.policy.get('headwise',False);metadata=L*H*7*4+4
        qbytes=sum(H*self.q*(8*(rk+rv)+4)for rk,rv in original)
        fixed=self.registry.end+L*(H*self.capacity+4*H*D*self.T)+metadata+qbytes
        available=self.budget-fixed;nt,gh,gw=self.grid;p=gh*gw;best=None
        allowed={self.policy['pattern']:PATTERNS[self.policy['pattern']]}if'pattern'in self.policy else PATTERNS
        for g in [2,4,8,16]:
            a=math.ceil(nt/g)*p;n=self.M-a;options=[]
            for l in range(L):
                for hh in range(H if headwise else 1):
                    oo=[]
                    for key,(ak,av,ik,iv)in allowed.items():
                        if self.policy.get('guard')and(ak<floors[l][0]or av<floors[l][1]):continue
                        for fi,fraction in enumerate(curves['fractions']):
                            count=math.ceil(n*fraction);loss=0.
                            for s,ra,ri in [('K',ak,ik),('V',av,iv)]:
                                anc=head['groups'][group]['norm'][str(g)][s][l]['anchor']
                                target=curves['groups'][group][str(g)][s][l][f'{ra}_{ri}']
                                ae=anc[hh][ra-1]if headwise else sum(v[ra-1]for v in anc)/H
                                te=target[hh][fi]if headwise else sum(v[fi]for v in target)/H
                                loss+=a*ae+n*alpha[str(g)][s]*te
                            cost=(1 if headwise else H)*(a*(8*(ak+av)+4)+count*(8*(ik+iv)+4))
                            oo.append({'bytes':cost,'loss':loss,'rAK':ak,'rAV':av,'rIK':ik,'rIV':iv,
                                       'innovations':count,'fi':fi,'pattern':key})
                    options.append(oo)
            try:chosen,report=allocate(options,available,1024 if headwise else 4096)
            except RuntimeError:continue
            key=(report['objective'],report['actual_variable_bytes'],g)
            if best is None or key<best[0]:best=(key,g,a,chosen,report)
        if best is None:raise RuntimeError('INFEASIBLE_ROLE_ALLOCATION')
        _,g,a,chosen,report=best
        layers=[chosen[l*H:(l+1)*H]for l in range(L)]if headwise else[[z.copy()for _ in range(H)]for z in chosen]
        for l,ll in enumerate(layers):
            for z in ll:z.update(qK=original[l][0],qV=original[l][1])
        self.plan.pop('layer_choices',None)
        self.plan.update(role_separated=True,headwise=headwise,gop=g,anchors=a,layer_head_roles=layers,
            allocation=report,query_rates_frozen_to_B0=original,query_index_gain_bytes=qbytes,
            role_metadata_bytes=metadata,role_curves_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
        header=torch.tensor([[[z[k]for k in ['rAK','rAV','rIK','rIV','qK','qV','innovations']]for z in ll]for ll in layers],dtype=torch.int32,device='cuda')
        self.registry.own('cache/role_plan',{'rates_counts':header,'G':torch.tensor([g],dtype=torch.int32,device='cuda')})
        self.descs=[]
        for ll in layers:
            desc=[]
            for h,z in enumerate(ll):
                for role,count in [('A',a),('I',z['innovations']),('Q',self.q)]:
                    for s in ['K','V']:
                        depth=z['q'+s]if role=='Q'else z['r'+role+s]
                        desc.extend([(f'{s}_{role}_index_h{h}',(count,depth,8),torch.uint8,'payload'),
                                     (f'{s}_{role}_norm_h{h}',(count,1),torch.bfloat16,'metadata')])
            desc.append(('T_code',(H,self.capacity),torch.uint8,'metadata'))
            desc.extend([(s,(H,self.T,D),torch.bfloat16,'payload')for s in ['text_FP_K','text_FP_V']])
            self.descs.append(desc)

    @torch.no_grad()
    def write_piece(self,l,h,s,role,x,state):
        t=self.table;mean=t[s+'_mean'][l,h].float()[None]if role!='I'else 0.
        scale=t[s+'_scale'][l,h].float()[None];index=state[f'{s}_{role}_index_h{h}']
        for b in range(0,len(x),512):
            values=x[b:b+512].float();u=(values-mean)/scale
            gain=u.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6).bfloat16();n=len(values)
            state[f'{s}_{role}_norm_h{h}'][b:b+n]=gain;z=((u/gain.float())@t['H']).reshape(1,-1,16)
            for r in range(index.shape[1]):
                book=t[s+'_books'][r][None].float();ids=nearest(z,book)
                index[b:b+n,r]=ids.reshape(n,8).to(torch.uint8);z-=torch.gather(book,1,ids[...,None].expand(-1,-1,16))

    def read_piece(self,l,h,s,role,state):
        t=self.table;ix=state[f'{s}_{role}_index_h{h}'];n=len(ix);z=torch.zeros((n,8,16),device='cuda')
        for r in range(ix.shape[1]):z+=t[s+'_books'][r].float()[ix[:,r].long()]
        decoded=(z.reshape(n,D)@t['H'].T)*state[f'{s}_{role}_norm_h{h}'].float()*t[s+'_scale'][l,h].float()
        if role!='I':decoded+=t[s+'_mean'][l,h].float()
        return decoded.bfloat16()

    @torch.no_grad()
    def encode(self,l,native):
        if self.plan['hybrid_dense']:return super().encode(l,native)
        st=self.states[l];vi=self._vi;ti=self._ti
        kp=native['K_post'].cuda();v=native['V'].cuda();st['text_FP_K'].copy_(kp[:,ti]);st['text_FP_V'].copy_(v[:,ti])
        raw={'K':native['K_pre'].cuda().float(),'V':v.float()};nt,gh,gw=self.grid;p=gh*gw;g=self.plan['gop']
        ordinal=torch.arange(self.M,device='cuda');ao=ordinal[(ordinal//p)%g==0];to=ordinal[(ordinal//p)%g!=0]
        codes,scores=fixed_reference(raw,vi.to(torch.int32),to,[self.table[s+'_scale'][l]for s in ['K','V']],gh,gw,g)
        refs=references(to,codes,gh,gw,g);st['T_code'].zero_()
        for h,z in enumerate(self.plan['layer_head_roles'][l]):
            anchor={}
            for s in ['K','V']:
                self.write_piece(l,h,s,'A',raw[s][h,vi[ao]],st);anchor[s]=self.read_piece(l,h,s,'A',st)
            st['T_code'][h,vi[to]]=codes[h];count=z['innovations']
            if count:
                picked=torch.argsort(scores[h],descending=True,stable=True)[:count].sort().values
                selected=to[picked];ref=refs[h,picked];slot=(ref//p//g)*p+ref%p
                st['T_code'][h,vi[selected]]=codes[h,picked]|128
                for s in ['K','V']:self.write_piece(l,h,s,'I',raw[s][h,vi[selected]]-anchor[s][slot].float(),st)

    @torch.no_grad()
    def decode(self,l):
        if self.plan['hybrid_dense']:return super().decode(l)
        st=self.states[l];vi=self.global_maps['video'].long();ti=self.global_maps['text'].long()
        nt,gh,gw=self.grid;p=gh*gw;g=self.plan['gop'];o=torch.arange(self.M,device='cuda')
        ao=o[(o//p)%g==0];to=o[(o//p)%g!=0];codes=st['T_code'][:,vi[to]];ref=references(to,codes&127,gh,gw,g)
        slot=(ref//p//g)*p+ref%p;result={s:torch.empty((H,self.end,D),dtype=torch.bfloat16,device='cuda')for s in ['K','V']}
        for h,z in enumerate(self.plan['layer_head_roles'][l]):
            mask=(codes[h]&128)!=0;assert int(mask.sum())==z['innovations']
            for s in ['K','V']:
                a=self.read_piece(l,h,s,'A',st);pred=a[slot[h]].clone()
                if z['innovations']:pred[mask]=(pred[mask].float()+self.read_piece(l,h,s,'I',st).float()).bfloat16()
                result[s][h,vi[ao]]=a;result[s][h,vi[to]]=pred
        result['K']=apply_rope(result['K'],self.global_maps['positions'][:,:self.end],self.global_maps['freq'],self.scale)
        result['K'][:,ti]=st['text_FP_K'];result['V'][:,ti]=st['text_FP_V'];return result['K'],result['V']
