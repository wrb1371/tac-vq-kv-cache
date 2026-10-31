"""A bounded 4-time by 8-spatial-token prefill ordering, retaining original positions.

Interior 32-token blocks mix nearby grid times; edge fragments concatenate in
the existing stream without unpaid padding or an invented frame identity.
"""
import torch
from model.codec.modality import VIDEO
from model.codec.packed_cache import CODEC_LOCAL_LM,CODEC_LM,PackedLayer,pack_unsigned
from model.codec.turboquant import lloyd_max


def reorder_video_indices(indices,grid):
    assert grid is not None and len(grid)==3,'use the real video_grid_thw'
    t,h,w=map(int,grid);assert t>0 and h%2==w%2==0
    p=h*w//4;assert indices.numel()==t*p,(indices.numel(),grid)
    original=torch.arange(t*p).reshape(t,p)
    order=torch.cat([original[i:i+4,j:j+8].reshape(-1)
                     for i in range(0,t,4) for j in range(0,p,8)])
    assert order.unique().numel()==indices.numel(),'not a bijection'
    return indices.index_select(0,order)


def pack_reordered(cache,x,offset,layer_idx,which):
    # Writer structure follows the pinned author's _pack_append; only video
    # gather order changes. Existing positions are the authoritative mask keys.
    batch,heads,tokens,dim=x.shape
    if layer_idx not in cache._packed[which]:
        cache._packed[which][layer_idx]=PackedLayer(batch,heads,dim,x.dtype,x.device)
    layer=cache._packed[which][layer_idx]
    local=which=='k' and cache.cfg.method=='omnikvquant'
    codec=CODEC_LOCAL_LM if local else CODEC_LM
    shared=which=='k' or cache.cfg.method=='turboquant'
    values=norms=None
    if shared:
        source=x.float();norms=source.norm(dim=-1).clamp_min(1e-12)
        rotation=cache._shared_rot(layer_idx,heads,dim,x.device,which)
        values=torch.einsum('bhtd,hcd->bhtc',source/norms.unsqueeze(-1),rotation)
    tags=cache._chunk_tags(offset,tokens)
    for mod in tags.unique().tolist():
        indices=(tags==mod).nonzero(as_tuple=True)[0]
        if mod==VIDEO:
            assert offset==0,'this component supports a full offline prefill, not video-stream appends'
            indices=reorder_video_indices(indices,cache.video_grid)
        device_indices=indices.to(x.device);positions=offset+indices
        if shared:sub=values.index_select(2,device_indices);norm=norms.index_select(2,device_indices)
        else:
            sub=x.index_select(2,device_indices).float();norm=sub.norm(dim=-1).clamp_min(1e-12)
            rotation=cache._rot(layer_idx,mod,heads,dim,x.device,which)
            sub=torch.einsum('bhtd,hcd->bhtc',sub/norm.unsqueeze(-1),rotation)
        group=layer.group(mod,codec)
        if local:group.append_local_lm(sub,norm,positions)
        else:
            boundaries=lloyd_max(dim)[1].to(x.device)
            if x.is_cuda:
                from model.codec.triton_store import pack_lm
                index,norm16=pack_lm(sub,norm,boundaries)
            else:index=pack_unsigned(torch.bucketize(sub.contiguous(),boundaries,right=True));norm16=norm.to(torch.float16)
            group.append(index,norm16,positions)
        cache._packed_fused_store|=x.is_cuda
        elements=batch*heads*indices.numel()*dim
        effective=2.+16./dim+(8.5/32 if local else 0)
        cache._elems[which][mod]=cache._elems[which].get(mod,0)+elements
        cache._bits[which][mod]=cache._bits[which].get(mod,0)+elements*effective
