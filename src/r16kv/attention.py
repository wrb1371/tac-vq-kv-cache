"""Reference FA2 unchanged; capture exact pre-MRoPE projections for NSN too."""
from r14kv.cache import install_reference_attention as legacy_install,PackedReferenceCache
def install_reference_attention(model):
    handles=legacy_install(model)
    for l,layer in enumerate(model.thinker.model.layers):
        attn=layer.self_attn
        if getattr(attn,'_r16_NSN_raw_capture_installed',False):continue
        def capture(module,args,output,attn=attn,l=l):
            cache=attn._r14_capture_owner
            if isinstance(cache,PackedReferenceCache) and cache.codec.method.startswith('NSNQuant'):
                assert cache.codec.needs_positions
                cache.pre[l]=output.view(1,output.shape[1],2,128).transpose(1,2)
        handles.append(attn.k_proj.register_forward_hook(capture));attn._r16_NSN_raw_capture_installed=True
    return handles
