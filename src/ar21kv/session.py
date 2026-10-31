"""Reuse immutable temporal media arrays; rewind only the reserved query tail."""
import hashlib
import torch
def rewind_query_tail(cache,media_end):
    assert not cache.pre and not cache.incoming and len(cache.states)==36
    assert all(s['n']>=media_end for s in cache.states.values())
    for s in cache.states.values():s['n']=media_end
    cache._seen_tokens=media_end;cache.seen_positions=media_end
    cache.registry.observe('resident_media_query_tail_rewound')
def media_fingerprint(cache,media_end):
    h=hashlib.sha256();count=0
    for l,s in sorted(cache.states.items()):
        for name,t in sorted(s['arena'].views.items()):
            if name in ['K_BF16','V_BF16','T_code']:view=t[:,:media_end]
            elif name in ['K_index','V_index','K_norm','V_norm']:
                view=t[:,:cache.codec.plan['anchor_tokens']+cache.codec.plan['innovation_slots']]
            elif name in ['text_FP_K','text_FP_V']:view=t
            else:raise AssertionError(('unsupported immutable session field',name))
            body=view.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes();h.update(str(l).encode()+name.encode()+body);count+=len(body)
    if cache.positions is not None:h.update(cache.positions[:,:media_end].contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return {'SHA256':h.hexdigest(),'media_state_bytes_fingerprinted':count,'scope':'validation digest, not a CPU prefix restore or deployed duplicate'}
