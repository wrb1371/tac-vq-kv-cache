"""Per-token least-squares V radius, through the existing FP16 norm slot.

The byte indices, modality rotation and native fused reader are unchanged.
This is standard scalar projection applied to the complete parent V codec;
it is not a new vector-quantization algorithm. No calibration or task labels.
"""
import torch
from model.codec.turboquant import lloyd_max
from model.codec.packed_cache import unpack_unsigned


def append_optimal_radius(cache,x,offset,layer_idx,original_pack):
    # A V codec change may coexist with inherited K components, including a
    # complete K representation replacement. Reordering must use identical
    # physical position mapping, so read source positions from stored groups.
    old_layer=cache._packed['v'].get(layer_idx)
    before={mod:g.length for mod,g in old_layer.groups.items()} if old_layer else {}
    original_pack()
    levels=lloyd_max(x.shape[-1])[0].to(x.device)
    for mod,g in cache._packed['v'][layer_idx].groups.items():
        first=before.get(mod,0);last=g.length
        if first==last:continue
        indices=unpack_unsigned(g.index[:,:,first:last]).long()
        unit=levels[indices];unit=unit/unit.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        rotation=cache._rot(layer_idx,mod,g.heads,g.dim,x.device,'v')
        decoded_direction=torch.einsum('bhtc,hcd->bhtd',unit,rotation)
        positions=g.positions[first:last].long()-offset
        original=x.index_select(2,positions).float()
        radius=(original*decoded_direction).sum(-1)/decoded_direction.square().sum(-1).clamp_min(1e-12)
        g.norm[:,:,first:last].copy_(radius.to(torch.float16))
