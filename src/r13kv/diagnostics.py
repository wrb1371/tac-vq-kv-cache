"""CAL-only range reconstruction diagnostics; never used to choose an answer."""
import math,torch
from model.codec.packed_cache import CODEC_LOCAL_LM,_decode_mxfp4
from model.codec.modality import MODALITY_NAMES
from .metadata import decode_unsigned4

def local_range_errors(cache,x,layer_idx):
    layer=cache._packed['k'][layer_idx]
    if not any(g.codec==CODEC_LOCAL_LM for g in layer.groups.values()):return []
    original=x.float();norm=original.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    rotation=cache._shared_rot(layer_idx,x.shape[1],x.shape[-1],x.device,'k')
    unit=torch.einsum('bhtd,hcd->bhtc',original/norm,rotation).half().float();rows=[]
    for mod,g in layer.groups.items():
        if g.codec!=CODEC_LOCAL_LM:continue
        n=g.length;groups=math.ceil(n/32);sub=unit.index_select(2,g.positions[:n].long())
        padded=torch.zeros(1,g.heads,groups*32,g.dim,device=x.device)
        padded[:,:,:n]=sub;values=padded.reshape(1,g.heads,groups,32,g.dim)
        valid=(torch.arange(groups*32,device=x.device)<n).reshape(1,1,groups,32,1)
        minimum=torch.where(valid,values,float('inf')).amin(3)
        maximum=torch.where(valid,values,float('-inf')).amax(3);span=(maximum-minimum).clamp_min(1e-12)
        decoded_min=_decode_mxfp4(g.local_min[:,:,:groups],g.local_min_scale[:,:,:groups],g.dim)
        decoder=decode_unsigned4 if cache.unsigned_span else _decode_mxfp4
        decoded_span=decoder(g.local_span[:,:,:groups],g.local_span_scale[:,:,:groups],g.dim)
        rows.append({'layer':layer_idx,'modality':MODALITY_NAMES[mod],'live_groups':groups,
           'minimum_relative_MSE':float((decoded_min-minimum).square().mean()/minimum.square().mean().clamp_min(1e-20)),
           'positive_span_relative_MSE':float((decoded_span-span).square().mean()/span.square().mean().clamp_min(1e-20)),
           'reference':'exact physical writer order, FP16 normalized rotated input, valid live32-token groups',
           'calibration_diagnostic_only':True})
    return rows
