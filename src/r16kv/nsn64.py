"""NSNQuant residual64 mechanism transfer: double-quantized NSN, official books, scale repair.

Torch reference implementation, explicit three-axis MRoPE, original-space FA2.
No claim of reproducing the authors' fused kernel latency or exact rounding.
"""
import math,torch
from r15kv.codecs import PackageCodec
from r15kv.budget import pack_indices,unpack_indices,estimate_resident_bytes
from r14kv.fitting import nearest

def q4(x,group):
    shape=x.shape;z=x.half().reshape(-1,group);lo=z.amin(-1,keepdim=True);scale=((z.amax(-1,keepdim=True)-lo)/15).clamp_min(1e-5).half()
    q=((z-lo)/scale).clamp(0,15).round().to(torch.uint8);rec=(q.half()*scale+lo).reshape(shape)
    return pack_indices(q.reshape(shape),4),scale.reshape(*shape[:-1],-1),lo.reshape(*shape[:-1],-1),rec
def dq4(packed,scale,lo,n,group):
    q=unpack_indices(packed,4,n).half();return (q.reshape(-1,group)*scale.reshape(-1,1)+lo.reshape(-1,1)).reshape(*packed.shape[:-1],n)
def layout(cap,bits):
    blocks=cap//64;qt=blocks*64;out=[]
    for side in ['K','V']:
        out += [(side+'_index',(2,qt,16),torch.uint8,'payload'),(side+'_sign',(2,qt,16 if bits==2 else 0),torch.uint8,'payload'),
          (side+'_norm1_index',(2,blocks,32),torch.uint8,'metadata'),(side+'_norm1_scale',(2,blocks,1),torch.float16,'metadata'),(side+'_norm1_offset',(2,blocks,1),torch.float16,'metadata'),
          (side+'_mean_index',(2,blocks,64),torch.uint8,'metadata'),(side+'_mean_scale',(2,blocks,4),torch.float16,'metadata'),(side+'_mean_offset',(2,blocks,4),torch.float16,'metadata'),
          (side+'_norm2',(2,qt,1),torch.float16,'metadata'),(side+'_pending',(2,64,128),torch.bfloat16,'payload')]
    return out
class NSN64Reference(PackageCodec):
    method='NSNQuant-R16-paper-default-residual64-three-axis-reference-transfer';needs_positions=True
    def make_plan(self,meta):
        dtype=torch.int16 if max(meta.position_max)+max(0,meta.generation_cap-1)<=32767 else torch.int32
        posbytes=3*meta.capacity*torch.empty((),dtype=dtype).element_size()+256
        for bits in self.policy.get('bit_menu',[2,1]):
            desc=layout(meta.capacity,bits);est=estimate_resident_bytes(self.static_bytes,posbytes,desc)
            if est['resident_capacity_bytes']<=self.budget_bytes:return {'bits':bits,'window':64,'description':desc,'estimate':est,'capacity':meta.capacity,'position_dtype':str(dtype),'official_fused_kernel':False,'double_quantization':'FP16 metadata, actual4bit norms and means','V_rotation':'explicit Hadamard before NSN, inverse at reference reader; no changed model weights'}
        raise RuntimeError('INFEASIBLE_BY_INTEGER_ORACLE_NSN')
    def allocate_state(self,l,registry):
        state=super().allocate_state(l,registry);state.update(quantized=0,pending=0);return state
    def rope(self,x,positions):
        from r14kv.common import apply_key_rope
        phase=positions.float()[:,:,None]*self.position_inv_freq[None,None,:];widths=[16,24,24]
        cs=[(phase.cos()*self.position_attention_scaling).to(x.dtype),(phase.sin()*self.position_attention_scaling).to(x.dtype)];out=[]
        for a in cs:
            z=torch.cat([a[i,:,sum(widths[:i]):sum(widths[:i+1])] for i in range(3)],-1);out.append(torch.cat([z,z],-1)[None,None])
        return apply_key_rope(x[None],*out)[0]
    def _encode(self,l,state,side,raw,start):
        # Complete windows only; preserve the authors' norm/shift/norm order.
        n=raw.shape[1];assert n%64==0 and start%64==0;h=self.tables[0]['H'];bits=self.plan['bits'];book=self.tables[0]['book'+str(bits)]
        block0=start//64;nb=n//64;z=raw.half()
        if side=='V':z=torch.matmul(z,h)
        norm=z.norm(dim=-1,keepdim=True)/math.sqrt(128);ni,ns,no,nr=q4(norm.reshape(2,nb,64),64);nr=nr.reshape(2,n,1).clamp_min(1e-5)
        z=(z/nr).half();mean=z.reshape(2,nb,64,128).mean(-2);mi,ms,mo,mr=q4(mean,32);z=(z.reshape(2,nb,64,128)-mr[:,:,None]).reshape(2,n,128).half()
        norm2=(z.norm(dim=-1,keepdim=True)/math.sqrt(128)).clamp_min(1e-5).half();z=(z/norm2).half()
        if side=='K':z=torch.matmul(self.rope(z,self.position_storage[:,start:start+n]),h).half()
        grouped=z.reshape(-1,8);search=grouped.abs() if bits==2 else grouped;ix=nearest(search.float()[None],book.float()[None],chunk=8192)[0];restored=book.index_select(0,ix).reshape(2,n,128)
        if bits==2:
            signs=z>=0;restored=torch.where(signs,restored,-restored);state[side+'_sign'][:,start:start+n].copy_(pack_indices(signs,1))
        # Author strategy3: preserve the original-direction projection.
        dot=(z.float()*restored.float()).sum(-1,keepdim=True);assert torch.isfinite(dot).all() and (dot>0).all(), 'NSN scale adjustment invalid projection'
        adjusted=(norm2.float()*z.float().square().sum(-1,keepdim=True)/dot).half();assert torch.isfinite(adjusted).all()
        state[side+'_index'][:,start:start+n].copy_(ix.reshape(2,n,16).to(torch.uint8));state[side+'_norm2'][:,start:start+n].copy_(adjusted)
        for key,val in [('norm1_index',ni),('norm1_scale',ns),('norm1_offset',no),('mean_index',mi),('mean_scale',ms),('mean_offset',mo)]:state[side+'_'+key][:,block0:block0+nb].copy_(val)
    def append(self,l,state,kpost,v,pre=None):
        assert pre is not None;raw={'K':pre[0],'V':v[0]};n=v.shape[-2];done=0
        while done<n:
            # Batch complete windows directly when the pending buffer is empty.
            if state['pending']==0 and n-done>=64:
                take=min((n-done)//64,16)*64
                for side in ['K','V']:self._encode(l,state,side,raw[side][:,done:done+take],state['quantized'])
                state['quantized']+=take;done+=take
            else:
                take=min(64-state['pending'],n-done);p=state['pending']
                for side in ['K','V']:state[side+'_pending'][:,p:p+take].copy_(raw[side][:,done:done+take])
                state['pending']+=take;done+=take
                if state['pending']==64:
                    for side in ['K','V']:self._encode(l,state,side,state[side+'_pending'],state['quantized'])
                    state['quantized']+=64;state['pending']=0
        state['n']+=n;assert state['n']==state['quantized']+state['pending'];return state
    def _restore(self,state,side):
        n=state['quantized'];nb=n//64;bits=self.plan['bits'];book=self.tables[0]['book'+str(bits)];h=self.tables[0]['H']
        if not n:return torch.empty(2,0,128,device='cuda',dtype=torch.float16)
        ix=state[side+'_index'][:,:n].long();z=book[ix].reshape(2,n,128)
        if bits==2:z=torch.where(unpack_indices(state[side+'_sign'][:,:n],1,128).bool(),z,-z)
        z=(z*state[side+'_norm2'][:,:n]).half()
        mean=dq4(state[side+'_mean_index'][:,:nb],state[side+'_mean_scale'][:,:nb],state[side+'_mean_offset'][:,:nb],128,32)[:,:,None].expand(-1,-1,64,-1).reshape(2,n,128)
        if side=='K':z=(torch.matmul(z,h)+self.rope(mean,self.position_storage[:,:n])).half()
        else:z=(z+mean).half()
        norm=dq4(state[side+'_norm1_index'][:,:nb],state[side+'_norm1_scale'][:,:nb],state[side+'_norm1_offset'][:,:nb],64,64).reshape(2,n,1).clamp_min(1e-5);z=(z*norm).half()
        if side=='V':z=torch.matmul(z,h)
        return z
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        k=self._restore(state,'K');v=self._restore(state,'V');p=state['pending'];start=state['quantized']
        if p:
            k=torch.cat([k,self.rope(state['K_pending'][:,:p],self.position_storage[:,start:start+p])],1);v=torch.cat([v,state['V_pending'][:,:p].half()],1)
        return k.bfloat16()[None],v.bfloat16()[None]
    def finish_request(self):
        self.position_storage=None;self.position_inv_freq=None;super().finish_request()
