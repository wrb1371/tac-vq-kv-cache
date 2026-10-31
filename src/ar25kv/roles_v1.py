"""Physically segmented A/I/Q shared-RVQ prefixes; small calibrated role families."""
import copy,json,math,os,time
import torch
from ar24kv.pressure_switch_v1 import PressureSwitch
from ar21kv.saturating import SaturatingInnovations
from ar21kv.dynamic import DynamicExactGemm
from ar21kv.dynamic_kernels import read_slots,compose
from ar22kv.planner import allocate
from ar19kv.packed_sparse import PackedSparse,HybridPackedSparse
from r14kv.common import ROOT,sha
from r15kv.persistent_registry import Arena
ROLE_DECODE_CHECKS=[]


def bounds(codec,layer,role):
 a=codec.plan['anchor_tokens'];i=codec.plan['layer_innovation_slots'][layer];q=codec.plan['query_slots']
 return {'A':(0,a),'I':(a,i),'Q':(a+i,q)}[role]

def view(codec,layer,state,side,role):
 start,count=bounds(codec,layer,role)
 return {side+'_index':state[side+'_'+role+'_index'],side+'_norm':state[side+'_norm'][:,start:start+count]}

class TorchRoleReader(PackedSparse):
 def _read_slots(self,layer,state,side,slots,independent):
  a=self.plan['anchor_tokens'];i=self.plan['layer_innovation_slots'][layer]
  # Synchronization is confined to the independent validation oracle.
  first=int(slots[0,0]);role='I' if not independent else 'A' if first<a else 'Q'
  start,_=bounds(self,layer,role)
  return PackedSparse._read_slots(self,layer,view(self,layer,state,side,role),side,slots-start,independent)

class RoleFamily(PressureSwitch):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.da=int(self.policy.get('anchor_depth_delta',0));self.dq=int(self.policy.get('query_depth_delta',0))
  assert self.da in [0,1] and self.dq in [0,1] and self.policy.get('ranking','norm')=='norm'
  source=ROOT/self.policy['role_curve_file'] if self.da else ROOT/self.policy['conditional_file']
  curves=json.loads(source.read_text());assert curves['completed'] and curves['bank_completion_SHA256']==self.package_sha256
  self._role_curve_sha=sha(source);self._role_curves=copy.deepcopy(curves)
  if self.policy.get('separable_target',False):
   f0=curves['fractions'].index(.125)
   for group,ranks in self._role_curves['groups'].items():
    for rank,gops in ranks.items():
     for gop,sides in gops.items():
      for side,layers in sides.items():
       for l,c in enumerate(layers):
        old=curves['groups'][group][rank][gop][side][l]
        c['target']=[[old['target'][f0][r]+row[5]-old['target'][f0][5] for r in range(6)] for row in old['target']]
  self.metadata.update(role_conditional_SHA256=self._role_curve_sha,role_depth_family={'anchor_plus':self.da,'query_plus':self.dq},role_layout='three independently sized index spans; gains/maps/text/positions/full prepaid capacity retained')

 def make_plan(self,meta):
  if self._hybrid_dense:return super().make_plan(meta)
  started=time.perf_counter();m,p=len(self._video_list),self._h*self._w;self._end=self._video_list[-1]+1;q=meta.capacity-self._end;t=len(self._text_list)
  dtype=torch.int16 if max(meta.position_max)<=32767 else torch.int32
  global_bytes=3*meta.capacity*torch.empty((),dtype=dtype).element_size()+256+4*m+4*t
  fixed=self.static_bytes+global_bytes+36*(2*meta.capacity+1024*t);rho=36864*(m+q)/max(1,self.budget_bytes-fixed)
  floors=[(2,2)]*36 if rho>32 else [(1,1)]*36
  if rho<=32 and self.policy.get('guarded_parent_depth',True):
   try:
    self.policy['gop']=self._base_gop;parent=SaturatingInnovations.make_plan(self,meta)
    floors=[(max(1,k-1),max(1,v-1)) for k,v in zip(parent['layer_R_K'],parent['layer_R_V'])]
   except RuntimeError as e:
    if 'INFEASIBLE' not in str(e):raise
  best=None
  for g in self.policy.get('gop_choices',[2,4,8,16]):
   a=math.ceil(self._nt/g)*p;n=m-a;curves=self._role_curves['groups'][self._request_group]['norm'][str(g)];options=[]
   for l in range(36):
    choices=[]
    for fi,f in enumerate(self._role_curves['fractions']):
     i=math.ceil(n*f);e=a+i+q
     for rk in range(floors[l][0],7):
      for rv in range(floors[l][1],7):
       raK,raV=min(6,rk+self.da),min(6,rv+self.da);rqK,rqV=min(6,rk+self.dq),min(6,rv+self.dq)
       cost=16*(a*(raK+raV)+i*(rk+rv)+q*(rqK+rqV))+8*e;loss=0.
       for side,ri,rq in [('K',rk,rqK),('V',rv,rqV)]:
        c=curves[side][l];alpha=self._content_factors[str(g)][side] if self._content_factors else 1.
        loss+=a*c['anchor'][ri-1]+n*alpha*c['target'][fi][ri-1]+q*c['query'][rq-1]
       choices.append({'bytes':cost,'loss':loss,'rK':rk,'rV':rv,'aK':raK,'aV':raV,'qK':rqK,'qV':rqV,'slots':e,'innovation_slots':i,'fraction_index':fi})
    options.append(choices)
   try:chosen,report=allocate(options,self.budget_bytes-fixed,quantum=int(self.policy.get('quantum_bytes',4096)))
   except RuntimeError as e:
    if 'INFEASIBLE' not in str(e):raise
    continue
   key=(report['objective'],fixed+report['actual_variable_bytes'],g)
   if best is None or key<best[0]:best=(key,g,a,chosen,report)
  if best is None:raise RuntimeError('INFEASIBLE_BY_ROLE_PHYSICAL_CATALOG')
  _,g,a,chosen,report=best;self.policy['gop']=g;self._anchor_ord=[i for i in range(m) if (i//p)%g==0]
  descriptions=[]
  for c in chosen:
   desc=[]
   for role,count,k,v in [('A',a,c['aK'],c['aV']),('I',c['innovation_slots'],c['rK'],c['rV']),('Q',q,c['qK'],c['qV'])]:
    desc += [('K_'+role+'_index',(2,count,k,8),torch.uint8,'payload'),('V_'+role+'_index',(2,count,v,8),torch.uint8,'payload')]
   desc += [(s+'_norm',(2,c['slots'],1),torch.bfloat16,'metadata') for s in ['K','V']]
   desc += [('T_code',(2,meta.capacity),torch.uint8,'metadata')]
   desc += [(s,(2,t,128),torch.bfloat16,'payload') for s in ['text_FP_K','text_FP_V']];descriptions.append(desc)
  actual=self.static_bytes+global_bytes+sum(Arena.bytes(d) for d in descriptions);assert actual==fixed+report['actual_variable_bytes']<=self.budget_bytes
  plan={'description':descriptions[0],'descriptions':descriptions,'capacity':meta.capacity,'R_K':chosen[0]['rK'],'R_V':chosen[0]['rV'],'position_dtype':str(dtype),'group_dim':16,'groups':8,'anchor_tokens':a,'innovation_slots':chosen[0]['innovation_slots'],'layer_innovation_slots':[c['innovation_slots'] for c in chosen],'encoded_slots_per_head':chosen[0]['slots'],'layer_encoded_slots_per_head':[c['slots'] for c in chosen],'query_slots':q,'media_original':m,'protected_prefix_text_tokens':t,'gop':g,'ranking':'norm','merged_video_grid':[self._nt,self._h,self._w],'estimate':{'codec_static_bytes':self.static_bytes,'global_request_bytes':global_bytes,'layer_arena_bytes':Arena.bytes(descriptions[0]),'all_layer_arena_bytes':sum(Arena.bytes(d) for d in descriptions),'resident_capacity_bytes':actual,'kind':'INTEGER_ESTIMATE_NOT_MEASUREMENT'},'joint_allocation':report,'planning_CPU_ms':1000*(time.perf_counter()-started),'content_adaptive':self._content_observation is not None,'content_observation':self._content_observation,'conditional_proxy_SHA256':self._role_curve_sha,'guarded_parent_depth':True,'hybrid_dense':False,'ar25_role_depths':{'A_delta':self.da,'Q_delta':self.dq},'ar24_pressure_switch_ratio':rho,'ar24_pressure_switch_branch':'original_allcell_parent_guard' if rho<=32 else 'pressure_two_stage_envelope','representation':'same sampled positions and closed-loop references; physically separate role index prefixes'}
  for side in ['K','V']:
   plan['layer_R_'+side]=[c['r'+side] for c in chosen]
   for role,prefix in [('A','a'),('I','r'),('Q','q')]:plan['layer_R_'+role+'_'+side]=[c[prefix+side] for c in chosen]
  return plan

 @torch.no_grad()
 def _innovation(self,l,state,raw,slots,side):
  if self._hybrid_dense:return super()._innovation(l,state,raw,slots,side)
  if state['n']:role='Q'
  else:
   key='_ar25_anchor_encoded_'+side;role='I' if state.get(key,False) else 'A';state[key]=True
  start,_=bounds(self,l,role)
  return DynamicExactGemm._innovation(self,l,view(self,l,state,side,role),raw,slots-start,side)

 def _read_slots(self,l,state,side,slots,independent):
  # Inherited media append reads anchors only; full decode dispatches roles explicitly.
  assert independent and not state['n']
  return read_slots(l,view(self,l,state,side,'A'),side,slots,True,self.tables[0])

 def torch_decode(self,l,state,positions,inv_freq,attention_scaling=1.):
  if self._hybrid_dense:return HybridPackedSparse.decode(self,l,state,positions,inv_freq,attention_scaling)
  slow=object.__new__(TorchRoleReader);slow.__dict__.update(self.__dict__);slow.plan={**self.plan,'innovation_slots':self.plan['layer_innovation_slots'][l],'encoded_slots_per_head':self.plan['layer_encoded_slots_per_head'][l]}
  return PackedSparse.decode(slow,l,state,positions,inv_freq,attention_scaling)

 def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
  if self._hybrid_dense:return super().decode(l,state,positions,inv_freq,attention_scaling)
  with self._layer(l):
   arrays=[]
   for role in ['A','I','Q']:
    _,count=bounds(self,l,role)
    if role=='Q':count=state['n']-self._end
    slots=torch.arange(count,device=state['T_code'].device)[None].expand(2,-1)
    for side in ['K','V']:
     arrays.append(read_slots(l,view(self,l,state,side,role),side,slots,role!='I',self.tables[0]) if count else torch.empty((2,0,128),device=state['T_code'].device,dtype=torch.bfloat16))
   result=compose(self,state,positions,inv_freq,attention_scaling,arrays)
  if os.environ.get('AR24_INDEPENDENT_DECODE_GATE')=='1' and l in [0,18,35]:
   done=getattr(self,'_role_checked',set())
   if l not in done:
    expected=self.torch_decode(l,state,positions,inv_freq,attention_scaling);diff=[int((a!=b).sum()) for a,b in zip(expected,result)];assert diff==[0,0],(l,diff);ROLE_DECODE_CHECKS.append({'layer':l,'BF16_differences':diff});done.add(l);self._role_checked=done
  return result
