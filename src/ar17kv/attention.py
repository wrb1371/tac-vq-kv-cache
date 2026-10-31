"""Explicit pre-RoPE capability for new codecs; no method-name scheduling alias."""
from r16kv.attention import install_reference_attention as previous
from r14kv.cache import PackedReferenceCache
def install_reference_attention(model):
    handles=previous(model)
    for l,layer in enumerate(model.thinker.model.layers):
        attn=layer.self_attn
        def capture(module,args,output,attn=attn,l=l):
            cache=attn._r14_capture_owner
            if isinstance(cache,PackedReferenceCache) and getattr(cache.codec,'needs_pre_rope_K',False):cache.pre[l]=output.view(1,output.shape[1],2,128).transpose(1,2)
        handles.append(attn.k_proj.register_forward_hook(capture))
    return handles
