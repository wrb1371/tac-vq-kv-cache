"""Native dense cache with observed old/new append overlap; unbudgeted reference."""
import torch
from transformers.cache_utils import DynamicCache
from .persistent_registry import PersistentRegistry
class TrackedBF16Cache(DynamicCache):
    def __init__(self):
        super().__init__();self.registry=PersistentRegistry(2**62);self.registry.own('cache/K',self.key_cache);self.registry.own('cache/V',self.value_cache)
    def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
        if layer_idx==0:self._seen_tokens+=key_states.shape[-2]
        while len(self.key_cache)<=layer_idx:
            self.key_cache.append(torch.empty(0,device=key_states.device,dtype=key_states.dtype));self.value_cache.append(torch.empty(0,device=value_states.device,dtype=value_states.dtype))
        if not self.key_cache[layer_idx].numel():
            self.key_cache[layer_idx]=key_states;self.value_cache[layer_idx]=value_states;self.registry.observe('first_dense_layer_'+str(layer_idx))
        else:
            newk=torch.cat([self.key_cache[layer_idx],key_states],-2);self.registry.own('cache/pending_new_K',newk);self.key_cache[layer_idx]=newk;self.registry.release('cache/pending_new_K')
            newv=torch.cat([self.value_cache[layer_idx],value_states],-2);self.registry.own('cache/pending_new_V',newv);self.value_cache[layer_idx]=newv;self.registry.release('cache/pending_new_V')
        return self.key_cache[layer_idx],self.value_cache[layer_idx]
    def measured_receipt(self):
        r=self.registry.receipt();r.update(budget_pass=None,budget_reference_only=True,codec_static_bytes=0,cache_payload_bytes=r['resident_end_bytes'],cache_metadata_and_capacity_bytes=0,process_peak_allocated_bytes=torch.cuda.max_memory_allocated(),process_peak_reserved_bytes=torch.cuda.max_memory_reserved(),codec_load_ms=0,codec_workspace_peak_bytes=None,allocation_contract='unbudgeted BF16 reference; actual native per-layer old/new dense concatenation overlap observed')
        return r
