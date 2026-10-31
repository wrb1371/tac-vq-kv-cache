"""Known unsigned E2M2-style finite FP4 levels for nonnegative range spans.

This is a component experiment, not a claim to invent unsigned FP4.
The parent's signed minimum representation and E8M0 block exponents stay intact.
"""
import torch
import triton
import triton.language as tl

U4_LEVELS=(0.,.25,.5,.75,1.,1.25,1.5,1.75,2.,2.5,3.,3.5,4.,5.,6.,7.)


def encode_unsigned4(values):
    width=int(values.shape[-1]);assert width%32==0 and width%2==0
    source=values.float();assert bool((source>=0).all()),'span is nonnegative'
    blocks=source.reshape(*source.shape[:-1],-1,32)
    maximum=blocks.amax(-1,keepdim=True)
    exponent=torch.ceil(torch.log2((maximum/7).clamp_min(2.**-127))).clamp_(-127,128)
    exponent=torch.where(maximum==0,torch.zeros_like(exponent),exponent)
    scale=torch.pow(2.,exponent);unit=blocks/scale
    levels=torch.tensor(U4_LEVELS,dtype=torch.float32,device=values.device)
    index=(unit.unsqueeze(-1)-levels).abs().argmin(-1)
    code=index.to(torch.uint8).reshape_as(source)
    packed=code[...,0::2]|(code[...,1::2]<<4)
    exponent_byte=(exponent.squeeze(-1).to(torch.int16)+127).to(torch.uint8)
    reconstructed=(levels[index]*scale).reshape_as(source)
    return packed.contiguous(),exponent_byte.contiguous(),reconstructed


def decode_unsigned4(packed,exponent_byte,width):
    assert packed.shape[-1]==width//2
    code=torch.empty((*packed.shape[:-1],width),dtype=torch.uint8,device=packed.device)
    code[...,0::2]=packed&15;code[...,1::2]=packed>>4
    levels=torch.tensor(U4_LEVELS,dtype=torch.float32,device=packed.device)
    scale=torch.pow(2.,exponent_byte.to(torch.int16).float()-127).repeat_interleave(32,-1)
    return levels[code.long()]*scale


@triton.jit
def decode_unsigned4_triton(code,exponent_byte):
    # The finite level set has no sign bit; every nibble is a valid positive value.
    low=tl.where(code==0,0.,tl.where(code==1,.25,tl.where(code==2,.5,tl.where(code==3,.75,
        tl.where(code==4,1.,tl.where(code==5,1.25,tl.where(code==6,1.5,1.75)))))))
    high=tl.where(code==8,2.,tl.where(code==9,2.5,tl.where(code==10,3.,tl.where(code==11,3.5,
        tl.where(code==12,4.,tl.where(code==13,5.,tl.where(code==14,6.,7.)))))))
    level=tl.where(code<8,low,high)
    return level*tl.exp2(exponent_byte.to(tl.float32)-127.)
