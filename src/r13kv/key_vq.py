"""Actual byte-index g4 K vector quantization, a preparation-stage component.

One FP16 256 x 4 book per layer/physical KV head, shared across dimension
groups and modalities. Post-RoPE keys retain the parent's Hadamard transform;
the parent's complete modality-specific V codec is unchanged. No novelty or
accuracy claim is implied by this implementation.
"""
import torch
from model.codec.packed_cache import PackedGroup, PackedLayer, CODEC_LM


def assign_vectors(values, book, chunk=8192):
    """Assignment uses the deployed FP16 book, with bounded FP32 workspace."""
    assert book.dtype == torch.float16 and book.shape == (values.shape[1],256,4)
    b,h,t,d=values.shape
    vectors=values.float().reshape(b,h,t,d//4,4)
    indices=torch.empty((b,h,t,d//4),device=values.device,dtype=torch.uint8)
    for head in range(h):
        x=vectors[:,head].reshape(-1,4);c=book[head].float()
        flat=indices[:,head].reshape(-1)
        for start in range(0,len(x),chunk):
            sub=x[start:start+chunk]
            distance=sub.square().sum(-1,keepdim=True)+c.square().sum(-1)[None]-2*sub@c.T
            flat[start:start+len(sub)]=distance.argmin(-1).to(torch.uint8)
    return indices


def reconstruct_vectors(indices,book,dim):
    b,h,t,g=indices.shape;assert dim==4*g
    values=torch.stack([book[i][indices[:,i].long()] for i in range(h)],dim=1)
    values=values.float().reshape(b,h,t,dim)
    return values/values.norm(dim=-1,keepdim=True).clamp_min(1e-12)


class KeyVQLayer(PackedLayer):
    # Byte indices have the same width as the parent's four packed 2-bit
    # scalars: each byte now addresses one four-dimensional vector.
    def group(self,mod,codec=CODEC_LM):
        return super().group(mod,CODEC_LM)


def append_key_vq(cache,x,offset,layer_idx):
    batch,heads,tokens,dim=x.shape
    assert dim==128 and not cache.closed_key_norm and not cache.unsigned_span and not cache.reorder_video
    if layer_idx not in cache._packed['k']:
        cache._packed['k'][layer_idx]=KeyVQLayer(batch,heads,dim,x.dtype,x.device)
    key=('key_vq',layer_idx)
    if key not in cache._packed_codebooks:
        cache._packed_codebooks[key]=cache.key_vq_books[layer_idx].to(device=x.device)
    book=cache._packed_codebooks[key]
    source=x.float();norm=source.norm(dim=-1).clamp_min(1e-12)
    rotation=cache._shared_rot(layer_idx,heads,dim,x.device,'k')
    unit=torch.einsum('bhtd,hcd->bhtc',source/norm.unsqueeze(-1),rotation)
    tags=cache._chunk_tags(offset,tokens)
    for mod in tags.unique().tolist():
        ids=(tags==mod).nonzero(as_tuple=True)[0];device_ids=ids.to(x.device)
        sub=unit.index_select(2,device_ids)
        indices=assign_vectors(sub,book)
        group=cache._packed['k'][layer_idx].group(mod)
        group.append(indices,norm.index_select(2,device_ids).half(),offset+ids)
        elements=batch*heads*len(ids)*dim
        cache._elems['k'][mod]=cache._elems['k'].get(mod,0)+elements
        cache._bits['k'][mod]=cache._bits['k'].get(mod,0)+elements*(2+16/dim)
    # Store is a bounded Torch reference encoder, not a fused-store claim.


def unpack_key_vq(cache,layer_idx):
    layer=cache._packed['k'][layer_idx];total=cache.get_seq_length(layer_idx)
    out=torch.empty(layer.batch,layer.heads,total,layer.dim,device=layer.device,dtype=torch.float32)
    filled=torch.zeros(total,device=layer.device,dtype=torch.bool)
    book=cache._packed_codebooks[('key_vq',layer_idx)]
    rotation=cache._shared_rot(layer_idx,layer.heads,layer.dim,layer.device,'k')
    for group in layer.groups.values():
        n=group.length;unit=reconstruct_vectors(group.index[:,:,:n],book,layer.dim)
        value=unit*group.norm[:,:,:n].float().unsqueeze(-1)
        restored=torch.einsum('bhtc,hcd->bhtd',value,rotation)
        positions=group.positions[:n].long();out.index_copy_(2,positions,restored);filled[positions]=True
    key,_,positions=cache.current_token(layer_idx)
    if key is not None:
        out.index_copy_(2,positions.long(),key.float());filled[positions.long()]=True
    assert filled.all()
    return out.to(layer.dtype)
