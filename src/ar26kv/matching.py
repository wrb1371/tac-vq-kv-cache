"""Exact old raw-reference matching kernel, launched over true physical KV heads."""
import torch
from ar21kv.dynamic_kernels import _match9
def fixed_reference(raw,video_indices,targets,scales,h,w,gop,importance=None):
    assert importance is None
    k=raw['K'].contiguous();v=raw['V'].contiguous();heads,n,_=k.shape;count=len(targets)
    codes=torch.empty((heads,count),dtype=torch.uint8,device=k.device);scores=torch.empty((heads,count),dtype=torch.float32,device=k.device)
    _match9[(count,heads)](k,v,k,v,video_indices,scales[0],scales[1],targets,codes,scores,n,count,h,w,gop,num_warps=4,enable_fp_fusion=False)
    return codes,scores

def references(targets,codes,h,w,gop):
    p=h*w;ry=(targets[None]//w%h+codes.long()//3-1).clamp(0,h-1);rx=(targets[None]%w+codes.long()%3-1).clamp(0,w-1)
    return targets[None]//p//gop*gop*p+ry*w+rx
