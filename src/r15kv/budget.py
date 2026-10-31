"""Integer-byte pre-screening; measured storage, never this oracle, certifies a run."""
from dataclasses import dataclass
import torch
from .persistent_registry import Arena
@dataclass(frozen=True)
class RequestMeta:
    prompt_tokens:int
    generation_cap:int=1
    position_max:tuple=(0,0,0)
    input_sha256:str=''
    @property
    def capacity(self):return self.prompt_tokens+max(0,self.generation_cap-1)
def pack_indices(x,bits):
    assert bits in (1,2,4,8)
    x=x.to(torch.uint8)
    if bits==8:return x.contiguous()
    per=8//bits;pad=(-x.shape[-1])%per
    if pad:x=torch.nn.functional.pad(x,(0,pad))
    x=x.reshape(*x.shape[:-1],-1,per);out=torch.zeros(x.shape[:-1],device=x.device,dtype=torch.uint8)
    for i in range(per):out|=x[...,i]<<(bits*i)
    return out
def unpack_indices(x,bits,n):
    if bits==8:return x[...,:n]
    return torch.stack([(x>>(bits*i))&((1<<bits)-1) for i in range(8//bits)],-1).flatten(-2)[...,:n]
def gs_split(total,kmax,vmax,bias=0):
    choices=[(k,total-k) for k in range(1,kmax+1) if 1<=total-k<=vmax]
    return min(choices,key=lambda z:(abs(z[0]-z[1]-bias),-z[0])) if choices else None
def nova_description(cap,sink,recent,groups,index_bits,vbits,sign=False):
    fs=min(sink,cap);rs=min(recent,max(0,cap-fs));low=max(0,cap-fs-rs)
    out=[('firstK',(2,fs,128),torch.bfloat16,'payload'),('firstV',(2,fs,128),torch.bfloat16,'payload'),('recentK',(2,rs,128),torch.bfloat16,'payload'),('recentV',(2,rs,128),torch.bfloat16,'payload'),('K_index',(2,low,(groups*index_bits+7)//8),torch.uint8,'payload'),('rho',(2,low,1),torch.bfloat16,'metadata'),('V_index',(2,low,128*vbits//8),torch.uint8,'payload'),('V_scale',(2,low,1),torch.bfloat16,'metadata')]
    if not sign:out.append(('V_zero',(2,low,1),torch.bfloat16,'metadata'))
    return out
def gsrq_description(cap,rk,rv):
    return [('K_index',(2,cap,rk),torch.uint8,'payload'),('V_index',(2,cap,rv),torch.uint8,'payload')]
def estimate_resident_bytes(codec_bytes,global_bytes,description):
    return {'codec_static_bytes':int(codec_bytes),'global_request_bytes':int(global_bytes),'layer_arena_bytes':Arena.bytes(description),'resident_capacity_bytes':int(codec_bytes)+int(global_bytes)+36*Arena.bytes(description),'kind':'INTEGER_ESTIMATE_NOT_MEASUREMENT'}
