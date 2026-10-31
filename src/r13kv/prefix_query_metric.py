"""Preparation only: position-conditioned pre-RoPE assignment on fixed books.

The objective is block-diagonal quadratic logit distortion with FP32 linear
RoPE, not the native BF16+global-unit-normalized decoder's full attention loss.
A future online encoder could obtain M from already available prefix queries;
this operator does not implement model hooks or generate any answers.
"""
import torch
from .prerope_packing import paired_groups

def prefix_query_metric(native_queries):
    b,h,n,d=native_queries.shape;assert (b,h,d)==(1,16,128) and n>0
    q=paired_groups(native_queries.float())[0].reshape(2,8,n,32,4).permute(0,3,1,2,4).reshape(2,32,8*n,4)
    moment=(q.transpose(-1,-2)@q/(8*n)).detach().double().cpu()
    val,vec=torch.linalg.eigh(moment);floor=val.amax(-1,keepdim=True).abs().clamp_min(1e-12)*1e-4
    m=vec@torch.diag_embed(val.maximum(floor))@vec.transpose(-1,-2)
    m=m/(m.diagonal(dim1=-2,dim2=-1).sum(-1)[...,None,None]/4).clamp_min(1e-20)
    return m.float().contiguous()

def position_rotations(coordinates,cos_lut,sin_lut):
    assert coordinates.ndim==2 and coordinates.shape[0]==3
    axis=torch.tensor([0]*16+[1]*24+[2]*24,device=coordinates.device)
    freq=torch.arange(64,device=coordinates.device);pos=coordinates.long()[axis].T
    cos=cos_lut[pos,freq[None]].float().reshape(-1,32,2).permute(1,0,2)
    sin=sin_lut[pos,freq[None]].float().reshape(-1,32,2).permute(1,0,2)
    r=torch.zeros(32,coordinates.shape[1],4,4,device=coordinates.device,dtype=torch.float32)
    for pair in range(2):
        r[...,pair,pair]=cos[...,pair];r[...,pair,pair+2]=-sin[...,pair]
        r[...,pair+2,pair]=sin[...,pair];r[...,pair+2,pair+2]=cos[...,pair]
    return r

def assign_position_weighted(unit,books,metric,coordinates,cos_lut,sin_lut,chunk=512):
    assert unit.shape[0:2]==(1,2) and unit.shape[-1]==128
    assert books.shape==(2,32,64,4) and books.dtype==torch.float16
    assert metric.shape==(2,32,4,4) and not metric.is_cuda and metric.dtype==torch.float32
    n=unit.shape[2];assert coordinates.shape==(3,n)
    result=torch.empty(1,2,n,32,device=unit.device,dtype=torch.uint8);vectors=paired_groups(unit)
    for head in range(2):
        c=books[head].float();outer=(c[...,None]*c[...,None,:]).flatten(-2)
        m=metric[head].to(unit.device)
        for start in range(0,n,chunk):
            end=min(n,start+chunk);rot=position_rotations(coordinates[:,start:end],cos_lut,sin_lut)
            local=rot.transpose(-1,-2)@m[:,None]@rot
            x=vectors[0,head,start:end].permute(1,0,2).float();xm=torch.einsum('gti,gtij->gtj',x,local)
            cx=torch.bmm(xm,c.transpose(1,2));cc=torch.bmm(local.flatten(-2),outer.transpose(1,2))
            distance=(xm*x).sum(-1,keepdim=True)+cc-2*cx
            result[0,head,start:end]=distance.argmin(-1).T.to(torch.uint8)
    return result
