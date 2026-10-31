"""Author global scalar K paired with the full Omni modality-specific V codec.

This hybrid is an explicit representation control, not author TurboQuant as a
whole. It removes only local K min/span and keeps norms, positions and V.
"""
import torch
from model.codec.packed_cache import PackedLayer,CODEC_LM,pack_unsigned
from model.codec.turboquant import lloyd_max


def append_global_key(cache,x,offset,layer_idx):
    b,h,t,d=x.shape
    if layer_idx not in cache._packed['k']:cache._packed['k'][layer_idx]=PackedLayer(b,h,d,x.dtype,x.device)
    source=x.float();norm=source.norm(dim=-1).clamp_min(1e-12);rotation=cache._shared_rot(layer_idx,h,d,x.device,'k')
    unit=torch.einsum('bhtd,hcd->bhtc',source/norm.unsqueeze(-1),rotation);tags=cache._chunk_tags(offset,t)
    boundaries=lloyd_max(d)[1].to(x.device)
    for mod in tags.unique().tolist():
        ids=(tags==mod).nonzero(as_tuple=True)[0];dev=ids.to(x.device);values=unit.index_select(2,dev);norms=norm.index_select(2,dev)
        if x.is_cuda:
            from model.codec.triton_store import pack_lm
            index,norm16=pack_lm(values,norms,boundaries)
        else:index=pack_unsigned(torch.bucketize(values.contiguous(),boundaries,right=True));norm16=norms.half()
        cache._packed['k'][layer_idx].group(mod,CODEC_LM).append(index,norm16,offset+ids)
        elements=b*h*len(ids)*d;cache._elems['k'][mod]=cache._elems['k'].get(mod,0)+elements;cache._bits['k'][mod]=cache._bits['k'].get(mod,0)+elements*(2+16/d)
    cache._packed_fused_store|=x.is_cuda
