"""Paid rate-conditioned decoder correction and exact-budget layer allocation."""
import torch
from ar17kv.psrvq import ProductRVQ
from ar17kv.text_protection import PaidPrefixText
from r15kv.persistent_registry import Arena

class AffineProduct(ProductRVQ):
    method='PSRVQ-rate-conditioned-affine'
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        from r14kv.common import apply_key_rope
        t=self.tables[0];n=state['n'];d=self.package['group_dim'];g=128//d;result=[]
        for side in ['K','V']:
            depth=state[side+'_index'].shape[2];z=torch.zeros(2,n,g,d,device='cuda')
            for r in range(depth):z+=t[side+'_books'][r].float()[state[side+'_index'][:,:n,r].long()]
            z=torch.matmul(z.reshape(2,n,128),t['H'].T);x=(z*state[side+'_norm'][:,:n].float()*t[side+'_scale'][l].float()[:,None]+t[side+'_mean'][l].float()[:,None]).bfloat16()
            if side+'_gain' in t:x=(x.float()*t[side+'_gain'][l,:,depth-1].float()[:,None]+t[side+'_bias'][l,:,depth-1].float()[:,None]).bfloat16()
            result.append(x[None])
        phase=positions.float()[:,:,None]*inv_freq[None,None,:];out=[];widths=[16,24,24]
        for trig in [torch.cos,torch.sin]:
            a=(trig(phase)*attention_scaling).bfloat16();b=torch.cat([a[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);out.append(torch.cat([b,b],-1)[None,None])
        return apply_key_rope(result[0],*out),result[1]

class AllocatedProduct(AffineProduct):
    method='PSRVQ-layer-sensitive-budget-allocation'
    def make_plan(self,meta):
        maximum=self.package['Rmax_K'];g=128//self.package['group_dim'];dtype=torch.int16 if max(meta.position_max)+max(0,meta.generation_cap-1)<=32767 else torch.int32;posbytes=3*meta.capacity*torch.empty((),dtype=dtype).element_size()+256
        normbytes=36*2*2*meta.capacity*2;stepbytes=2*g*meta.capacity;limit=min(72*maximum,(self.budget_bytes-self.static_bytes-posbytes-normbytes)//stepbytes)
        if limit<72:raise RuntimeError('INFEASIBLE_BY_INTEGER_ORACLE_LAYER_PSRVQ')
        costs=self.tables[0]['rate_distortion'].detach().cpu().double().permute(1,0,2).reshape(72,maximum).tolist()
        dp={0:(0.,[])}
        for curve in costs:
            nxt={}
            for used,(value,path) in dp.items():
                for r in range(1,maximum+1):
                    new=used+r
                    if new>limit:break
                    score=value+curve[r-1]
                    if new not in nxt or score<nxt[new][0]:nxt[new]=(score,path+[r])
            dp=nxt
        used,(objective,chosen)=min(dp.items(),key=lambda kv:(kv[1][0],-kv[0]));descs=[]
        for l in range(36):
            rk,rv=chosen[2*l:2*l+2];descs.append([('K_index',(2,meta.capacity,rk,g),torch.uint8,'payload'),('V_index',(2,meta.capacity,rv,g),torch.uint8,'payload')]+[(side+'_norm',(2,meta.capacity,1),torch.bfloat16,'metadata') for side in ['K','V']])
        total=self.static_bytes+posbytes+sum(Arena.bytes(d) for d in descs);assert total<=self.budget_bytes
        return {'R_K':chosen[::2],'R_V':chosen[1::2],'description':descs[0],'descriptions':descs,'group_dim':self.package['group_dim'],'groups':g,'capacity':meta.capacity,'position_dtype':str(dtype),'loaded_Rmax_K':maximum,'loaded_Rmax_V':maximum,'allocator':'exact multiple-choice DP of frozen CAL scalar-gradient distortion; <=budget','allocation_objective':objective,'allocation_stage_sum':used,'estimate':{'codec_static_bytes':self.static_bytes,'global_request_bytes':posbytes,'layer_arena_bytes_list':[Arena.bytes(d) for d in descs],'resident_capacity_bytes':total,'kind':'INTEGER_ESTIMATE_NOT_MEASUREMENT'}}
    def allocate_state(self,l,registry):
        arena=Arena(self.plan['descriptions'][l],'cuda',registry,f'cache/layer{l:02}/arena');return {'arena':arena,'n':0,**arena.views}

class ProtectedAffine(PaidPrefixText,AffineProduct):
    method='PSRVQ-paid-text-rate-affine';needs_pre_rope_K=True
class ProtectedAllocated(PaidPrefixText,AllocatedProduct):
    method='PSRVQ-paid-text-layer-allocation';needs_pre_rope_K=True
