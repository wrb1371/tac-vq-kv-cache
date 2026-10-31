"""Fused RVQ lookup + Walsh butterfly + parameter restoration (shadow prototype)."""
import torch,triton
import triton.language as tl
from ar21kv.dynamic_kernels import compose

@triton.jit(do_not_specialize=['N','IS0','IS1','NS0'])
def _read(I,B,Slots,Norm,Scale,Mean,H,O,N,IS0,IS1,NS0,R:tl.constexpr,INDEPENDENT:tl.constexpr,BM:tl.constexpr):
 row=tl.program_id(0)*BM+tl.arange(0,BM)
 col=tl.arange(0,128);head=row//N;token=row%N
 slot=tl.load(Slots+head*N+token,row<2*N,0)
 z=tl.full((BM,128),0,tl.float32)
 for stage in tl.static_range(R):
  code=tl.load(I+head[:,None]*IS0+slot[:,None]*IS1+stage*8+col[None,:]//16,row[:,None]<2*N,0).to(tl.int32)
  z+=tl.load(B+(stage*256+code)*16+col[None,:]%16,row[:,None]<2*N,0).to(tl.float32)
 for shift in tl.static_range(7):
  stride=1<<shift
  pairs=tl.permute(tl.reshape(z,(BM,128//(2*stride),2,stride)),(0,1,3,2))
  left,right=tl.split(pairs)
  pairs=tl.join(left+right,left-right)
  z=tl.reshape(tl.permute(pairs,(0,1,3,2)),(BM,128))
 z=z*tl.load(H)
 gain=tl.load(Norm+head*NS0+slot,row<2*N,0).to(tl.float32)
 scale=tl.load(Scale+head[:,None]*128+col[None,:],row[:,None]<2*N,0).to(tl.float32)
 v=(z*gain[:,None])*scale
 if INDEPENDENT:v=v+tl.load(Mean+head[:,None]*128+col[None,:],row[:,None]<2*N,0).to(tl.float32)
 tl.store(O+row[:,None]*128+col[None,:],v,row[:,None]<2*N)

def read_slots_fwht(layer,state,side,slots,independent,table):
 slots=slots.contiguous();n=slots.shape[1];ix=state[side+'_index'];norm=state[side+'_norm'];out=torch.empty((2,n,128),device=ix.device,dtype=torch.bfloat16)
 if n:_read[(triton.cdiv(2*n,4),)](ix,table[side+'_books'],slots,norm,table[side+'_scale'][layer],table[side+'_mean'][layer],table['H'],out,n,ix.stride(0),ix.stride(1),norm.stride(0),ix.shape[2],independent,4,num_warps=4,enable_fp_fusion=False)
 return out

def decode_fwht(codec,layer,state,positions,inv_freq,attention_scaling=1.):
 assert not codec._hybrid_dense
 with codec._layer(layer):
  a=codec.plan['anchor_tokens'];i=codec.plan['innovation_slots'];q=state['n']-codec._end;arrays=[]
  for start,count,independent in [(0,a,True),(a,i,False),(a+i,q,True)]:
   slots=torch.arange(start,start+count,device=state['T_code'].device)[None].expand(2,-1)
   for side in ['K','V']:arrays.append(read_slots_fwht(layer,state,side,slots,independent,codec.tables[0]) if count else torch.empty((2,0,128),device=state['T_code'].device,dtype=torch.bfloat16))
  return compose(codec,state,positions,inv_freq,attention_scaling,arrays)
