"""Native prefill; physical-GQA FA2 or compressed on-chip history at readout."""
import types
import torch
from ar18kv.attention_v2 import install_reference_attention as install_base
from r14kv.cache import PackedReferenceCache
from r16kv.cache_v2 import R16CacheLogged
from .code_flash import attend


class KernelCache(R16CacheLogged):
    def receipt(self):
        r=super().receipt()
        r.update(fused_attention_ms=sum(s.elapsed_time(e) for kind,s,e in self.events if kind=='fused_attention'),
                 readout_backend=getattr(self,'ar20_readout_backend','native'),
                 fused_attention_scope='includes tile reconstruction, mapping, online attention and split reduction; distinct from decode-only time')
        return r


def install_reference_attention(model,backend='native'):
    handles=install_base(model)
    assert backend in ['native','gqa','code_flash','code_flash_dense']
    if backend=='native':return handles
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_multimodal_rotary_pos_emb
    from flash_attn import flash_attn_func
    for layer in model.thinker.model.layers:
        a=layer.self_attn
        assert not a.config.use_sliding_window
        a._ar20_native=a._r14_original_forward
        def forward(self,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,output_attentions=False,use_cache=False,cache_position=None,position_embeddings=None):
            cache=past_key_value
            if not isinstance(cache,PackedReferenceCache) or self.layer_idx not in cache.states or (getattr(cache.codec,'_hybrid_dense',False) and backend!='gqa'):
                return self._ar20_native(hidden_states,attention_mask=attention_mask,position_ids=position_ids,past_key_value=cache,output_attentions=output_attentions,use_cache=use_cache,cache_position=cache_position,position_embeddings=position_embeddings)
            assert not output_attentions and hidden_states.shape[0]==1 and not self.training
            if self.layer_idx==0 and attention_mask is not None:assert bool(attention_mask.all()),'AR20 backend only unpadded batch1'
            batch,length,_=hidden_states.shape
            q=self.q_proj(hidden_states).view(batch,length,16,128).transpose(1,2)
            k=self.k_proj(hidden_states).view(batch,length,2,128).transpose(1,2)
            v=self.v_proj(hidden_states).view(batch,length,2,128).transpose(1,2)
            q,k=apply_multimodal_rotary_pos_emb(q,k,*position_embeddings,self.rope_scaling['mrope_section'])
            cache.ar20_readout_backend=backend
            if backend=='gqa':
                kk,vv=cache.update(k,v,self.layer_idx,{'cache_position':cache_position})
                out=flash_attn_func(q.transpose(1,2),kk.transpose(1,2),vv.transpose(1,2),causal=True)
            else:
                state=cache.states[self.layer_idx];n=state['n']
                out=cache._time(lambda:attend(cache.codec,self.layer_idx,state,cache.positions[:,:n],cache.inv_freq,q.transpose(1,2),k,v,cache.attention_scaling,dense=backend=='code_flash_dense'),'fused_attention')
                cache.read_calls+=1;cache.incoming[self.layer_idx]=(k,v)
                if self.layer_idx==0:cache._seen_tokens+=length
            return self.o_proj(out.reshape(batch,length,-1).contiguous()),None,cache
        a._r14_original_forward=types.MethodType(forward,a)
    return handles
