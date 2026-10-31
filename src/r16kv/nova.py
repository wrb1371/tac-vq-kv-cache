"""Token-affine V1/V2 in W128 blocks; fixed CAL priority, uniform time coverage."""
import math, torch
from r15kv.codecs import NOVABudget
from r15kv.budget import nova_description,pack_indices,unpack_indices
from r15kv.persistent_registry import Arena

def description(cap,sink,recent,groups,index_bits,modes):
    base=[r for r in nova_description(cap,sink,recent,groups,index_bits,1) if r[0]!='V_index']
    low=max(0,cap-min(sink,cap)-min(recent,max(0,cap-sink)))
    lengths=[min(128,low-b) for b in range(0,low,128)]
    n1=sum(n for n,m in zip(lengths,modes) if m==1);n2=low-n1
    return base+[('V1_index',(2,n1,16),torch.uint8,'payload'),('V2_index',(2,n2,32),torch.uint8,'payload'),('V_block_mode',(len(modes),),torch.uint8,'metadata'),('V_block_offset',(len(modes),),torch.int32,'metadata')]

def plan_for(meta,static,groups,index_bits,budget,policy):
    sink,recent=policy.get('band',[32,128]);cap=meta.capacity
    low=max(0,cap-min(sink,cap)-min(recent,max(0,cap-sink)));lengths=[min(128,low-b) for b in range(0,low,128)]
    migration=2*2*min(recent,max(0,cap-sink))*128*2
    mode=policy.get('allocator','A1');sens=policy.get('layer_gains',[1.]*36)
    modes=[[1]*len(lengths) for _ in range(36)]
    desc=[description(cap,sink,recent,groups,index_bits,m) for m in modes]
    total=int(static)+migration+sum(Arena.bytes(d) for d in desc)
    if total>budget:raise RuntimeError('INFEASIBLE_BY_INTEGER_ORACLE_NOVA_R16')
    base_total=total
    if mode=='A0':
        all2=[[2]*len(lengths) for _ in range(36)];d2=[description(cap,sink,recent,groups,index_bits,m) for m in all2]
        cost2=static+migration+sum(Arena.bytes(d) for d in d2)
        # Ordinary global V1/V2; CAL verifies the preferred direction before DEV.
        if cost2<=budget and sum(sens)>0:modes=all2;desc=d2;total=cost2
    else:
        # Within each layer visit time blocks uniformly: nested bit-reversal order.
        nb=len(lengths);width=max(1,(max(1,nb)-1).bit_length())
        order=sorted(range(nb),key=lambda i:int(bin(i)[2:].zfill(width)[::-1],2))
        candidates=[(-max(0,float(sens[l])),rank,l,j) for l in range(36) for rank,j in enumerate(order) if sens[l]>0]
        candidates.sort()
        for _,rank,l,j in candidates:
            # Exact aligned arena delta, including mode/offset tables already prepaid.
            delta=32*lengths[j]
            if total+delta<=budget:total+=delta;modes[l][j]=2
        desc=[description(cap,sink,recent,groups,index_bits,m) for m in modes]
        assert static+migration+sum(Arena.bytes(d) for d in desc)==total
    choices=[{'layer':l,'block':j,'tokens':n,'extra_bytes':32*n} for l in range(36) for j,n in enumerate(lengths) if modes[l][j]==1 and sens[l]>0]
    v2=sum(n for l in range(36) for n,m in zip(lengths,modes[l]) if m==2)
    return {'sink':sink,'recent':recent,'capacity':cap,'window':128,'allocator':mode,'modes':modes,'descriptions':desc,'description':desc[0],
            'estimate':{'codec_static_bytes':int(static),'global_request_bytes':migration,'resident_capacity_bytes':int(total),'kind':'INTEGER_ESTIMATE_NOT_MEASUREMENT'},
            'migration_bytes':migration,'base_V1_resident_bytes':base_total,'V2_tokens_across_layers':v2,'low_tokens_across_layers':36*low,
            'remaining_bytes':int(budget-total),'minimum_legal_upgrade_bytes':min((x['extra_bytes'] for x in choices),default=None),
            'active_selection':bool(0<v2<36*low),'stop_reason':'all positive upgrades complete' if not choices else 'ordinary global action infeasible' if mode=='A0' else 'remaining budget below legal positive upgrade',
            'no_current_video_semantic_score':True,'no_test_labels':True}

class NOVAAllocated(NOVABudget):
    method='NOVA-R16-token-affine-W128-V1V2'
    def make_plan(self,meta):
        planner_static=self.policy.get('matched_plan_codec_bytes',self.static_bytes)
        try:
            plan=plan_for(meta,planner_static,self.groups,self.index_bits,self.budget_bytes,self.policy)
            plan['estimate']['codec_static_bytes']=self.static_bytes;plan['estimate']['resident_capacity_bytes']-=planner_static-self.static_bytes
            if planner_static!=self.static_bytes:plan['matched_layout_reserve_not_GPU_padding']=planner_static-self.static_bytes
            return plan
        except RuntimeError as e:
            if 'INFEASIBLE_BY_INTEGER_ORACLE' not in str(e):raise
            # Same frozen ordinary fallback for both A0/A1 and all N packages.
            plan=NOVABudget.make_plan(self,meta);plan['legacy_fallback']=True;plan['allocator']=self.policy.get('allocator','A1');plan['fallback_reason']='affine1 minimum cannot fit; use existing global sign1 with the same fixed band';return plan
    def allocate_state(self,l,registry):
        if self.plan.get('legacy_fallback'):return NOVABudget.allocate_state(self,l,registry)
        arena=Arena(self.plan['descriptions'][l],'cuda',registry,f'cache/layer{l:02}/arena');s={'arena':arena,'n':0,'nf':0,'nr':0,'nlow':0,**arena.views}
        modes=self.plan['modes'][l];offsets=[];counts={1:0,2:0};low=max(0,self.meta.capacity-self.plan['sink']-self.plan['recent'])
        for j,m in enumerate(modes):offsets.append(counts[m]);counts[m]+=min(128,low-j*128)
        if modes:
            s['V_block_mode'].copy_(torch.tensor(modes,dtype=torch.uint8,device='cuda'));s['V_block_offset'].copy_(torch.tensor(offsets,dtype=torch.int32,device='cuda'))
        return s
    def rotate_v(self,l,v,start=0):return torch.bmm(v,self.tables[l]['V_rotation'].expand(2,-1,-1)).float()
    def inverse_v(self,l,vr,start=0):return torch.bmm(vr,self.tables[l]['V_rotation'].float().transpose(1,2).expand(2,-1,-1))
    def _routing(self,state,start,n):
        ids=torch.arange(start,start+n,device='cuda');block=ids//128;mode=state['V_block_mode'].index_select(0,block);slot=state['V_block_offset'].index_select(0,block).long()+ids%128
        return mode,slot
    @torch.no_grad()
    def write_values(self,l,state,v,start):
        if self.plan.get('legacy_fallback'):return NOVABudget.write_values(self,l,state,v,start)
        for b in range(0,v.shape[1],1024):
            vr=self.rotate_v(l,v[:,b:b+1024]);n=vr.shape[1];sl=slice(start+b,start+b+n);modes,slots=self._routing(state,start+b,n)
            threshold=vr.abs().sort(-1).values[...,int(.92*128):int(.92*128)+1];z=vr.clamp(-threshold,threshold)
            lo=z.amin(-1,keepdim=True);hi=z.amax(-1,keepdim=True)
            for bits in [1,2]:
                ix=(modes==bits).nonzero(as_tuple=True)[0]
                if not len(ix):continue
                zz=z.index_select(1,ix);scale=((hi.index_select(1,ix)-lo.index_select(1,ix)).clamp_min(1e-8)/(2**bits-1)).bfloat16();zero=(-lo.index_select(1,ix)/scale.float()).bfloat16()
                q=(zz/scale.float()+zero.float()).round().clamp(0,2**bits-1).to(torch.uint8);native=ix+start+b
                state['V_scale'].index_copy_(1,native,scale);state['V_zero'].index_copy_(1,native,zero)
                state[f'V{bits}_index'].index_copy_(1,slots.index_select(0,ix),pack_indices(q,bits))
    @torch.no_grad()
    def synthesize_low(self,l,state):
        if self.plan.get('legacy_fallback'):return NOVABudget.synthesize_low(self,l,state)
        n=state['nlow'];t=self.tables[l];ix=unpack_indices(state['K_index'][:,:n],self.index_bits,self.groups).long();rows=[]
        for h in range(2):rows.append(t['centroids'][h].float()[torch.arange(self.groups,device='cuda')[None,:],ix[h]])
        r=torch.stack(rows).reshape(2,n,128)*state['rho'][:,:n].float();k=torch.bmm(r,t['A'].float().transpose(1,2))+t['mean'].float()[:,None]
        modes,slots=self._routing(state,0,n);q=torch.empty(2,n,128,device='cuda')
        for bits in [1,2]:
            ids=(modes==bits).nonzero(as_tuple=True)[0]
            if len(ids):q.index_copy_(1,ids,unpack_indices(state[f'V{bits}_index'].index_select(1,slots.index_select(0,ids)),bits,128).float())
        vr=(q-state['V_zero'][:,:n].float())*state['V_scale'][:,:n].float()
        return k.bfloat16(),self.inverse_v(l,vr).bfloat16()
