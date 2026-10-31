"""State-lifetime-aware role allocation; actual V+1 conditional calibration."""
from ar25kv.roles_v1 import *

class LifecycleRole(RoleFamily):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.v_only=bool(self.policy.get('value_anchor_plus_one',False))
  self.metadata.update(role_depth_family={'A_K_delta':0,'A_V_delta':int(self.v_only),'Q_fixed':1},allocation_objective_scope='media distortion only; read-only suffix is ephemeral; all reserved Q capacity still physically charged')

 def make_plan(self,meta):
   assert meta.generation_cap==1 and self.policy.get('read_only_single_step'),'LIFECYCLE_FAMILY_REQUIRES_SINGLE_FORWARD_SCORE_OP'
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
        raK,raV=rk,min(6,rv+int(self.v_only));rqK=rqV=1
        cost=16*(a*(raK+raV)+i*(rk+rv)+q*(rqK+rqV))+8*e;loss=0.
        for side,ri,rq in [('K',rk,rqK),('V',rv,rqV)]:
         c=curves[side][l];alpha=self._content_factors[str(g)][side] if self._content_factors else 1.
         loss+=a*c['anchor'][ri-1]+n*alpha*c['target'][fi][ri-1]
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
   plan={'description':descriptions[0],'descriptions':descriptions,'capacity':meta.capacity,'R_K':chosen[0]['rK'],'R_V':chosen[0]['rV'],'position_dtype':str(dtype),'group_dim':16,'groups':8,'anchor_tokens':a,'innovation_slots':chosen[0]['innovation_slots'],'layer_innovation_slots':[c['innovation_slots'] for c in chosen],'encoded_slots_per_head':chosen[0]['slots'],'layer_encoded_slots_per_head':[c['slots'] for c in chosen],'query_slots':q,'media_original':m,'protected_prefix_text_tokens':t,'gop':g,'ranking':'norm','merged_video_grid':[self._nt,self._h,self._w],'estimate':{'codec_static_bytes':self.static_bytes,'global_request_bytes':global_bytes,'layer_arena_bytes':Arena.bytes(descriptions[0]),'all_layer_arena_bytes':sum(Arena.bytes(d) for d in descriptions),'resident_capacity_bytes':actual,'kind':'INTEGER_ESTIMATE_NOT_MEASUREMENT'},'joint_allocation':report,'planning_CPU_ms':1000*(time.perf_counter()-started),'content_adaptive':self._content_observation is not None,'content_observation':self._content_observation,'conditional_proxy_SHA256':self._role_curve_sha,'guarded_parent_depth':True,'hybrid_dense':False,'ar25_role_depths':{'A_K_delta':0,'A_V_delta':int(self.v_only),'Q_fixed':1,'query_objective_weight':0,'scope':'single-forward read-only scoring'},'ar24_pressure_switch_ratio':rho,'ar24_pressure_switch_branch':'original_allcell_parent_guard' if rho<=32 else 'pressure_two_stage_envelope','representation':'same sampled positions and closed-loop references; physically separate role index prefixes'}
   for side in ['K','V']:
    plan['layer_R_'+side]=[c['r'+side] for c in chosen]
    for role,prefix in [('A','a'),('I','r'),('Q','q')]:plan['layer_R_'+role+'_'+side]=[c[prefix+side] for c in chosen]
   return plan
