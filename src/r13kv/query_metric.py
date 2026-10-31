"""Encoder-only query-weighted assignment for deployed FP16 trueVQ books.

The metric remains on CPU. Per-layer/group metrics and distance workspace
are transient encoder tensors; decoder, books, byte indices and norm layout
are unchanged. The quadratic objective is block-diagonal, not full attention.
"""
import torch
from r13kv.key_vq import KeyVQLayer

def assign_weighted(values,book,metric,chunk=512):
    b,h,t,d=values.shape;assert b==1 and d==128 and book.shape==(2,256,4) and metric.shape==(2,32,4,4)
    assert book.dtype==torch.float16 and not metric.is_cuda
    vectors=values.float().reshape(1,2,t,32,4);indices=torch.empty(1,2,t,32,device=values.device,dtype=torch.uint8)
    for head in range(2):
        m=metric[head].to(values.device);c=book[head].float()[None].expand(32,256,4)
        cm=torch.bmm(c,m);c2=(cm*c).sum(-1);x=vectors[0,head].permute(1,0,2)
        for start in range(0,t,chunk):
            sub=x[:,start:start+chunk];xm=torch.bmm(sub,m)
            distance=(xm*sub).sum(-1,keepdim=True)+c2[:,None,:]-2*torch.bmm(sub,cm.transpose(1,2))
            indices[0,head,start:start+sub.shape[1]].copy_(distance.argmin(-1).T.to(torch.uint8))
    return indices


def append_query_metric(cache,x,offset,layer_idx):
    b,h,t,d=x.shape;assert (b,h,d)==(1,2,128) and not (cache.closed_key_norm or cache.unsigned_span or cache.reorder_video)
    if layer_idx not in cache._packed['k']:cache._packed['k'][layer_idx]=KeyVQLayer(b,h,d,x.dtype,x.device)
    key=('key_vq',layer_idx)
    if key not in cache._packed_codebooks:cache._packed_codebooks[key]=cache.key_vq_books[layer_idx].to(x.device)
    book=cache._packed_codebooks[key];source=x.float();norm=source.norm(dim=-1).clamp_min(1e-12)
    rotation=cache._shared_rot(layer_idx,h,d,x.device,'k');unit=torch.einsum('bhtd,hcd->bhtc',source/norm.unsqueeze(-1),rotation)
    tags=cache._chunk_tags(offset,t)
    for mod in tags.unique().tolist():
        ids=(tags==mod).nonzero(as_tuple=True)[0];dev=ids.to(x.device)
        labels=assign_weighted(unit.index_select(2,dev),book,cache.assignment_metric[layer_idx])
        cache._packed['k'][layer_idx].group(mod).append(labels,norm.index_select(2,dev).half(),offset+ids)
        elements=b*h*len(ids)*d;cache._elems['k'][mod]=cache._elems['k'].get(mod,0)+elements;cache._bits['k'][mod]=cache._bits['k'].get(mod,0)+elements*(2+16/d)
