"""Common inference-only execution optimization, not a codec mutation.

The author prefix helper discards every prefix logit. Restricting only its
final linear projection leaves every Transformer/KV operation unchanged.
"""
from contextlib import contextmanager


@contextmanager
def final_prefix_projection(model):
    assert not model.training,'inference-only; never apply to labeled training loss'
    def trim(module,args):
        hidden=args[0]
        return (hidden[:,-1:,:],) if hidden.ndim==3 and hidden.shape[1]>1 else args
    handle=model.thinker.lm_head.register_forward_pre_hook(trim)
    try:yield
    finally:handle.remove()


def install_final_prefix_projection():
    import model.modeling_qwen2_5_omni as native
    if hasattr(native,'_r13_original_prefill'):return native.prefill_mcq_prefix
    native._r13_original_prefill=native.prefill_mcq_prefix
    def prefill(model,inputs,cache):
        with final_prefix_projection(model):return native._r13_original_prefill(model,inputs,cache)
    native.prefill_mcq_prefix=prefill
    return prefill
