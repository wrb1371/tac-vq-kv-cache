"""Numerically admitted short-query fusion; native physical GQA for other cases."""
import types
import torch
from ar20kv.attention import KernelCache,install_reference_attention as install_gqa
from .dynamic_code_flash import attend
from r14kv.cache import PackedReferenceCache


class AdaptiveCache(KernelCache):
    def receipt(self):
        r=super().receipt()
        r.update(adaptive_fused_calls=getattr(self,'adaptive_fused_calls',0),adaptive_GQA_calls=getattr(self,'adaptive_GQA_calls',0),adaptive_threshold_Q_tokens=4)
        return r


def install_reference_attention(model,backend='adaptive'):
    if backend!='adaptive':return install_gqa(model,backend)
    handles=install_gqa(model,'gqa')
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_multimodal_rotary_pos_emb
    for layer in model.thinker.model.layers:
        attn=layer.self_attn;attn._ar20_gqa=attn._r14_original_forward
        def forward(self,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,output_attentions=False,use_cache=False,cache_position=None,position_embeddings=None):
            cache=past_key_value;length=hidden_states.shape[1]
            existing=isinstance(cache,PackedReferenceCache) and self.layer_idx in cache.states
            eligible=existing and length<=4 and not getattr(cache.codec,'_hybrid_dense',False)
            if not eligible:
                if existing:cache.adaptive_GQA_calls=getattr(cache,'adaptive_GQA_calls',0)+1
                return self._ar20_gqa(hidden_states,attention_mask=attention_mask,position_ids=position_ids,past_key_value=cache,output_attentions=output_attentions,use_cache=use_cache,cache_position=cache_position,position_embeddings=position_embeddings)
            assert hidden_states.shape[0]==1 and not output_attentions and not self.training
            assert cache.codec.package_sha256=='f7a8c143037006f769cd2f021b128f06290bd10611ba77aee33f3841d43f0a8a','butterfly backend requires the verified canonical Hadamard bank'
            if self.layer_idx==0 and attention_mask is not None:assert bool(attention_mask.all())
            q=self.q_proj(hidden_states).view(1,length,16,128).transpose(1,2)
            k=self.k_proj(hidden_states).view(1,length,2,128).transpose(1,2)
            v=self.v_proj(hidden_states).view(1,length,2,128).transpose(1,2)
            q,k=apply_multimodal_rotary_pos_emb(q,k,*position_embeddings,self.rope_scaling['mrope_section'])
            state=cache.states[self.layer_idx];n=state['n']
            out=cache._time(lambda:attend(cache.codec,self.layer_idx,state,cache.positions[:,:n],cache.inv_freq,q.transpose(1,2),k,v,cache.attention_scaling),'fused_attention')
            cache.read_calls+=1;cache.incoming[self.layer_idx]=(k,v)
            if self.layer_idx==0:cache._seen_tokens+=length
            cache.adaptive_fused_calls=getattr(cache,'adaptive_fused_calls',0)+1;cache.ar20_readout_backend='adaptive_Q<=4_fused_else_physical_GQA'
            return self.o_proj(out.reshape(1,length,-1).contiguous()),None,cache
        attn._r14_original_forward=types.MethodType(forward,attn)
    return handles
