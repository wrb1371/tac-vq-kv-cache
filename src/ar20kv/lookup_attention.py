"""Research prototype: true index-domain attention under relaxed FP32 rounding.

K scores use three coordinate-specific ADC lookup tables. V uses weighted code
histograms and one inverse transform per output. No dense historical K/V is
assembled here. The frozen representation's intermediate BF16 roundings cannot
be algebraically commuted, so this is explicitly not a bit-equivalent reader.
"""
import torch
from .code_flash import maps


def lookup_attention(codec,l,state,positions,inv_freq,q,knew,vnew):
    assert state['n']==codec._end and q.shape[0]==1
    n=state['n'];qn=q.shape[1];f=qn*16;device=q.device
    physical=(torch.arange(f,device=device)%16)//8
    query=q.reshape(f,128).float();rot=torch.cat([-query[:,64:],query[:,:64]],-1)
    base,res=maps(codec,state);vid=codec._vid_gpu.long();text=codec._text_gpu.long()
    b=base[physical[:,None],vid[None]];r=res[physical[:,None],vid[None]];active=r>=0
    t=codec.tables[0];scores=torch.zeros(f,len(vid),device=device)
    kix=state['K_index'];bk=kix[physical[:,None],b];rk=kix[physical[:,None],r.clamp_min(0)]
    normb=state['K_norm'][physical[:,None],b,0].float();normr=state['K_norm'][physical[:,None],r.clamp_min(0),0].float()*active
    stages=kix.shape[2];ff=torch.arange(f,device=device)[:,None,None,None];rr=torch.arange(stages,device=device)[None,None,:,None];gg=torch.arange(8,device=device)[None,None,None,:]
    d=torch.arange(128,device=device);axis=torch.where(d%64<16,0,torch.where(d%64<40,1,2))
    largest_LUT_bytes=0
    for ax in range(3):
        coords,inverse=torch.unique(positions[ax,vid],sorted=True,return_inverse=True)
        phase=coords.float()[:,None]*inv_freq[d%64][None]
        effective=(query[:,None]*phase.cos()[None]-rot[:,None]*phase.sin()[None])*(axis==ax)[None,None]
        transformed=(effective*t['K_scale'][l][physical].float()[:,None])@t['H']
        lut=torch.einsum('fugd,rcd->furgc',transformed.reshape(f,len(coords),8,16),t['K_books'][:stages].float())
        largest_LUT_bytes=max(largest_LUT_bytes,lut.numel()*lut.element_size())
        index=inverse[None,:,None,None]
        anchor=lut[ff,index,rr,gg,bk.long()].sum((-1,-2))*normb
        innovation=lut[ff,index,rr,gg,rk.long()].sum((-1,-2))*normr
        mean=torch.einsum('fud,fd->fu',effective,t['K_mean'][l][physical].float())[:,inverse]
        scores+=anchor+innovation+mean
    full=torch.empty(f,n+qn,device=device);full[:,vid]=scores
    full[:,text]=torch.einsum('fd,fnd->fn',query,state['text_FP_K'][physical].float())
    kn=knew[0][physical].float();vn=vnew[0][physical].float()
    incoming=torch.einsum('fd,fnd->fn',query,kn)
    allowed=torch.arange(qn,device=device)[None]<=torch.arange(f,device=device)[:,None]//16
    full[:,n:]=incoming.masked_fill(~allowed,-float('inf'))
    prob=(full/(128**.5)).softmax(-1)
    vi=state['V_index'];stagesv=vi.shape[2];bc=vi[physical[:,None],b];rc=vi[physical[:,None],r.clamp_min(0)]
    wb=prob[:,vid]*state['V_norm'][physical[:,None],b,0].float()
    wr=prob[:,vid]*state['V_norm'][physical[:,None],r.clamp_min(0),0].float()*active
    histogram=torch.zeros(f,stagesv,8,256,device=device)
    histogram.scatter_add_(3,bc.permute(0,2,3,1).long(),wb[:,None,None].expand(-1,stagesv,8,-1))
    histogram.scatter_add_(3,rc.permute(0,2,3,1).long(),wr[:,None,None].expand(-1,stagesv,8,-1))
    compressed_output=torch.einsum('frgc,rcd->fgd',histogram,t['V_books'][:stagesv].float()).reshape(f,128)
    output=(compressed_output@t['H'].T)*t['V_scale'][l][physical].float()+t['V_mean'][l][physical].float()*prob[:,vid].sum(-1,keepdim=True)
    output+=torch.einsum('fn,fnd->fd',prob[:,text],state['text_FP_V'][physical].float())
    output+=torch.einsum('fn,fnd->fd',prob[:,n:],vn)
    return output.reshape(1,qn,16,128),{'largest_single_axis_LUT_bytes':largest_LUT_bytes,'whole_history_KV_materialized':False,'rounding_scope':'relaxed FP32; no original intermediate BF16 rounds; numerical prototype only'}


def relaxed_explicit(codec,l,state,positions,inv_freq,q,knew,vnew):
    """Independent algebra reference that explicitly assembles unrounded K/V."""
    base,res=maps(codec,state);t=codec.tables[0];n=state['n'];out=[];head=torch.arange(2,device=q.device)[:,None]
    for side in ['K','V']:
        used=codec.plan['anchor_tokens']+codec.plan['innovation_slots']
        ix=state[side+'_index'][:,:used];z=torch.zeros(2,used,8,16,device=q.device)
        for stage in range(ix.shape[2]):z+=t[side+'_books'][stage].float()[ix[:,:,stage].long()]
        z=((z.reshape(2,-1,128)@t['H'].T)*state[side+'_norm'][:,:used].float())*t[side+'_scale'][l].float()[:,None]
        v=z[head,base.clamp_min(0)]+t[side+'_mean'][l].float()[:,None]
        v+=z[head,res.clamp_min(0)]*(res>=0)[...,None]
        if side=='K':
            d=torch.arange(128,device=q.device);ax=torch.where(d%64<16,0,torch.where(d%64<40,1,2));phase=positions[ax].T.float()*inv_freq[d%64][None]
            v=v*phase.cos()[None]+torch.cat([-v[:,:,64:],v[:,:,:64]],-1)*phase.sin()[None]
        v[:,codec._text_gpu.long()]=state['text_FP_'+side].float();out.append(v)
    k,v=out;k=torch.cat([k,knew[0].float()],1);v=torch.cat([v,vnew[0].float()],1)
    qn=q.shape[1];physical=torch.arange(16,device=q.device)//8
    scores=torch.einsum('qhd,hnd->hqn',q[0].float(),k[physical])/(128**.5)
    allowed=torch.arange(n+qn,device=q.device)[None]<=n+torch.arange(qn,device=q.device)[:,None]
    weights=scores.masked_fill(~allowed[None],-float('inf')).softmax(-1)
    return torch.einsum('hqn,hnd->qhd',weights,v[physical])[None]
