"""Four6-bit indices occupy exactly three bytes, with no hidden byte index."""
import torch

def pack6(indices):
    assert indices.dtype==torch.uint8 and indices.shape[-1]%4==0
    assert not indices.numel() or int(indices.max())<64
    x=indices.reshape(*indices.shape[:-1],-1,4)
    a=x[...,0]|((x[...,1]&3)<<6)
    b=(x[...,1]>>2)|((x[...,2]&15)<<4)
    c=(x[...,2]>>4)|(x[...,3]<<2)
    return torch.stack((a,b,c),-1).flatten(-2).contiguous()

def unpack6(packed):
    assert packed.dtype==torch.uint8 and packed.shape[-1]%3==0
    x=packed.reshape(*packed.shape[:-1],-1,3)
    a=x[...,0]&63
    b=(x[...,0]>>6)|((x[...,1]&15)<<2)
    c=(x[...,1]>>4)|((x[...,2]&3)<<4)
    d=x[...,2]>>2
    return torch.stack((a,b,c,d),-1).flatten(-2)

def paired_groups(x):
    assert x.shape[-1]==128
    return torch.stack((x[...,:64:2],x[...,1:64:2],x[...,64::2],x[...,65::2]),-1)

def undo_paired_groups(x):
    assert x.shape[-2:]==(32,4)
    result=torch.empty(*x.shape[:-2],128,device=x.device,dtype=x.dtype)
    result[...,:64:2]=x[...,0];result[...,1:64:2]=x[...,1]
    result[...,64::2]=x[...,2];result[...,65::2]=x[...,3]
    return result
