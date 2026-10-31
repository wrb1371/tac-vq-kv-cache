"""Tile reconstruction + online attention: no full historical K/V HBM tensor.

This is fused tile dequantization, not pure index-domain arithmetic. The optional
canonical Hadamard butterfly changes floating-point summation order explicitly.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _shape(I, B, Norm, Scale, Mean, H, slots, head, IS0:tl.constexpr, IS1:tl.constexpr,
           NS0:tl.constexpr, R:tl.constexpr, BN:tl.constexpr, INDEPENDENT:tl.constexpr, DENSE:tl.constexpr):
    d=tl.arange(0,128)
    good=slots>=0
    z=tl.full((BN,128),0,tl.float32)
    for r in tl.static_range(R):
        ix=tl.load(I+head*IS0+slots[:,None]*IS1+r*8+d[None,:]//16,good[:,None],0).to(tl.int32)
        z+=tl.load(B+(r*256+ix)*16+d[None,:]%16,good[:,None],0).to(tl.float32)
    if DENSE:
        ht=tl.load(H+d[:,None]+d[None,:]*128)
        z=tl.dot(z,ht,input_precision='ieee')
    else:
        for bit in tl.static_range(7):
            pairs=z.reshape(BN,128//(2*(1<<bit)),2,1<<bit).permute(0,1,3,2)
            left,right=tl.split(pairs)
            z=tl.join(left+right,left-right).permute(0,1,3,2).reshape(BN,128)
        z=z*0.08838834764831845
    norm=tl.load(Norm+head*NS0+slots,good,0).to(tl.float32)
    scale=tl.load(Scale+head*128+d).to(tl.float32)
    z=(z*norm[:,None])*scale[None,:]
    if INDEPENDENT:
        z+=tl.load(Mean+head*128+d)[None,:].to(tl.float32)
    return z.to(tl.bfloat16)


@triton.jit(do_not_specialize=['N'])
def _partial(Q, IK, IV, BK, BV, NK, NV, SK, SV, MK, MV, H, Base, Residual, TK, TV, Pos, Freq, KNew, VNew, Temp, Max, Den,
             N, QN:tl.constexpr, KS0:tl.constexpr, KS1:tl.constexpr, VS0:tl.constexpr, VS1:tl.constexpr,
             KNS0:tl.constexpr,VNS0:tl.constexpr,RK:tl.constexpr,RV:tl.constexpr,NT:tl.constexpr,PS0:tl.constexpr,
             SPLITS:tl.constexpr, QT:tl.constexpr, CHUNK:tl.constexpr, BN:tl.constexpr, BM:tl.constexpr,DENSE:tl.constexpr,SCALE:tl.constexpr):
    qtile=tl.program_id(0);head=tl.program_id(1);part=tl.program_id(2)
    rr=qtile*BM+tl.arange(0,BM)
    qhead=head*8+rr//QN
    qi=rr%QN
    valid=rr<8*QN
    d=tl.arange(0,128)
    q=tl.load(Q+(qi[:,None]*16+qhead[:,None])*128+d[None,:],valid[:,None],0)
    mi=tl.full((BM,),-float('inf'),tl.float32)
    li=tl.full((BM,),0,tl.float32)
    acc=tl.full((BM,128),0,tl.float32)
    for offset in range(0,CHUNK,BN):
        j=part*CHUNK+offset+tl.arange(0,BN)
        past=j<N;incoming=(j>=N)&(j<N+QN)
        # Clamp speculative descriptors; incoming tokens override their values.
        # Avoid a Triton3.1 masked-load layout conversion bug in this inlined path.
        base=tl.load(Base+head*N+tl.minimum(j,N-1))
        res=tl.load(Residual+head*N+tl.minimum(j,N-1))
        kb=_shape(IK,BK,NK,SK,MK,H,base,head,KS0,KS1,KNS0,RK,BN,True,DENSE)
        kr=_shape(IK,BK,NK,SK,MK,H,res,head,KS0,KS1,KNS0,RK,BN,False,DENSE)
        vb=_shape(IV,BV,NV,SV,MV,H,base,head,VS0,VS1,VNS0,RV,BN,True,DENSE)
        vr=_shape(IV,BV,NV,SV,MV,H,res,head,VS0,VS1,VNS0,RV,BN,False,DENSE)
        k=(kb.to(tl.float32)+kr.to(tl.float32)).to(tl.bfloat16)
        v=(vb.to(tl.float32)+vr.to(tl.float32)).to(tl.bfloat16)
        ax=tl.where(d%64<16,0,tl.where(d%64<40,1,2))
        pos=tl.load(Pos+ax[None,:]*PS0+j[:,None],past[:,None],0).to(tl.float32)
        freq=tl.load(Freq+d%64)
        co=(libdevice.cos(pos*freq[None,:])*SCALE).to(tl.bfloat16).to(tl.float32)
        si=(libdevice.sin(pos*freq[None,:])*SCALE).to(tl.bfloat16).to(tl.float32)
        first,second=tl.split(k.reshape(BN,2,64).permute(0,2,1))
        rotated=tl.join(second,first).permute(0,2,1).reshape(BN,128).to(tl.float32)
        left=(k.to(tl.float32)*co).to(tl.bfloat16).to(tl.float32)
        right=(tl.where(d[None,:]<64,-rotated,rotated)*si).to(tl.bfloat16).to(tl.float32)
        k=(left+right).to(tl.bfloat16)
        text=past&(base<=-2)
        ti=-base-2
        k=tl.where(text[:,None],tl.load(TK+(head*NT+ti[:,None])*128+d[None,:],text[:,None],0),k)
        v=tl.where(text[:,None],tl.load(TV+(head*NT+ti[:,None])*128+d[None,:],text[:,None],0),v)
        k=tl.where(incoming[:,None],tl.load(KNew+((head*QN+j[:,None]-N)*128)+d[None,:],incoming[:,None],0),k)
        v=tl.where(incoming[:,None],tl.load(VNew+((head*QN+j[:,None]-N)*128)+d[None,:],incoming[:,None],0),v)
        score=tl.dot(q,tl.trans(k))*0.08838834764831845
        allowed=(j[None,:]<N+QN)&(j[None,:]<=N+qi[:,None])&valid[:,None]
        score=tl.where(allowed,score,-1.0e9)
        mn=tl.maximum(mi,tl.max(score,1))
        weight=tl.exp(score-mn[:,None])
        weight=tl.where(allowed,weight,0.0)
        alpha=tl.exp(mi-mn)
        li=li*alpha+tl.sum(weight,1)
        acc=acc*alpha[:,None]+tl.dot(weight.to(tl.bfloat16),v)
        mi=mn
    row=((head*QT+qtile)*SPLITS+part)*BM+tl.arange(0,BM)
    tl.store(Temp+row[:,None]*128+d[None,:],acc)
    tl.store(Max+row,mi);tl.store(Den+row,li)


@triton.jit
def _merge(Temp,Max,Den,Out,QN:tl.constexpr,QT:tl.constexpr,SPLITS:tl.constexpr,BM:tl.constexpr,BS:tl.constexpr):
    row=tl.program_id(0);head=tl.program_id(1)
    d=tl.arange(0,128);s=tl.arange(0,BS)
    tile=row//BM;lane=row%BM
    off=((head*QT+tile)*SPLITS+s)*BM+lane
    m=tl.load(Max+off,s<SPLITS,-1.0e9);l=tl.load(Den+off,s<SPLITS,0)
    biggest=tl.max(m,0);weight=tl.exp(m-biggest)
    x=tl.load(Temp+off[:,None]*128+d[None,:],s[:,None]<SPLITS,0)
    value=tl.sum(x*weight[:,None],0)/tl.sum(l*weight,0)
    qhead=head*8+row//QN;query=row%QN
    tl.store(Out+(query*16+qhead)*128+d,value)


def maps(codec,state):
    """Transient slot descriptors only; never retained as a cache or codec table."""
    n=state['n'];m=len(codec._video_list);a=codec.plan['anchor_tokens'];nr=codec.plan['innovation_slots']
    base=torch.full((2,n),-1,device='cuda',dtype=torch.int32);res=torch.full_like(base,-1)
    order=torch.arange(m,device='cuda');frame=order//(codec._h*codec._w);sp=order%(codec._h*codec._w)
    codes=state['T_code'][:,codec._vid_gpu.long()];anchor=frame%int(codec.policy['gop'])==0
    yy=(sp//codec._w+(codes&127).long()//3-1).clamp(0,codec._h-1);xx=(sp%codec._w+(codes&127).long()%3-1).clamp(0,codec._w-1)
    refs=frame//int(codec.policy['gop'])*(codec._h*codec._w)+yy*codec._w+xx
    refs=torch.where(anchor[None],frame//int(codec.policy['gop'])*(codec._h*codec._w)+sp,refs)
    active=((codes&128)!=0)&~anchor[None];rank=active.int().cumsum(1,dtype=torch.int32)-1
    base[:,codec._vid_gpu.long()]=refs.int();res[:,codec._vid_gpu.long()]=torch.where(active,rank+a,-1).int()
    base[:,codec._text_gpu.long()]=-torch.arange(len(codec._text_list),device='cuda',dtype=torch.int32)-2
    if n>codec._end:base[:,codec._end:n]=torch.arange(a+nr,a+nr+n-codec._end,device='cuda',dtype=torch.int32)
    return base,res


def attend(codec,l,state,positions,inv_freq,q,knew,vnew,attention_scaling=1.,dense=False):
    assert q.shape[0]==1 and q.shape[2:]==(16,128)
    q=q.contiguous();knew=knew.contiguous();vnew=vnew.contiguous();base,res=maps(codec,state)
    n=state['n'];qn=q.shape[1];bm=16;bn=16;chunk=128;qt=triton.cdiv(8*qn,bm);splits=triton.cdiv(n+qn,chunk)
    partial=torch.empty((2,qt,splits,bm,128),device=q.device,dtype=torch.float32);maximum=torch.empty((2,qt,splits,bm),device=q.device);den=torch.empty_like(maximum)
    t=codec.tables[0];ik=state['K_index'];iv=state['V_index'];nk=state['K_norm'];nv=state['V_norm']
    _partial[(qt,2,splits)](q,ik,iv,t['K_books'],t['V_books'],nk,nv,t['K_scale'][l],t['V_scale'][l],t['K_mean'][l],t['V_mean'][l],t['H'],base,res,state['text_FP_K'],state['text_FP_V'],positions,inv_freq,knew,vnew,partial,maximum,den,
        n,qn,ik.stride(0),ik.stride(1),iv.stride(0),iv.stride(1),nk.stride(0),nv.stride(0),ik.shape[2],iv.shape[2],len(codec._text_list),positions.stride(0),splits,qt,chunk,bn,bm,dense,attention_scaling,num_warps=4,num_stages=1,enable_fp_fusion=False)
    out=torch.empty_like(q)
    _merge[(8*qn,2)](partial,maximum,den,out,qn,qt,splits,bm,triton.next_power_of_2(splits))
    return out
