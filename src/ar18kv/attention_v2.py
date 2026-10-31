"""Explicit pre-RoPE capability/family capture, dense mode has no raw-K requirement."""
import types
from r14kv.cache import PackedReferenceCache

def install_reference_attention(model):
    handles=[]
    for l,layer in enumerate(model.thinker.model.layers):
        attn=layer.self_attn
        assert not hasattr(attn,'_r14_original_forward')
        attn._r14_original_forward=attn.forward;attn._r14_capture_owner=None
        def capture(module,args,output,attn=attn,l=l):
            cache=attn._r14_capture_owner
            if not isinstance(cache,PackedReferenceCache) or getattr(cache.codec,'_hybrid_dense',False):return
            codec=cache.codec;need=getattr(codec,'needs_pre_rope_K',False) or codec.package['family'] in ['GSRQ','NSN']
            if need:cache.pre[l]=output.view(1,output.shape[1],2,128).transpose(1,2)
        handles.append(attn.k_proj.register_forward_hook(capture))
        def wrapped(self,*args,**kwargs):
            cache=kwargs.get('past_key_value')
            if not isinstance(cache,PackedReferenceCache):return self._r14_original_forward(*args,**kwargs)
            if self.layer_idx==0:cache.capture_positions(kwargs['position_ids'])
            self._r14_capture_owner=cache
            try:
                result=self._r14_original_forward(*args,**kwargs);cache.encode_after_attention(self.layer_idx);return result
            finally:self._r14_capture_owner=None
        attn.forward=types.MethodType(wrapped,attn)
    return handles
