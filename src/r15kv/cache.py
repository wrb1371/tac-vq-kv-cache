"""Reference reader with bounded persistent arenas and complete allocation receipts."""
import torch
from transformers.cache_utils import DynamicCache
from r14kv.cache import PackedReferenceCache
from .persistent_registry import PersistentRegistry,storages
class BudgetCache(PackedReferenceCache):
    def __init__(self,codec,inv_freq,meta,attention_scaling=1.):
        DynamicCache.__init__(self)
        self.codec=codec;self.meta=meta;self.attention_scaling=attention_scaling;self.registry=PersistentRegistry(codec.budget_bytes)
        self.states={};self.pre={};self.incoming={};self.read_calls=0;self.materialized_peak_bytes=0;self.seen_positions=0
        self.peak_allocated=torch.cuda.max_memory_allocated();self.peak_reserved=torch.cuda.max_memory_reserved();self.workspace_peak=0;self.events=[]
        self.plan=codec.prepare_request(meta,self.registry)
        self.inv_freq=inv_freq.detach().float().clone() if codec.needs_positions else None
        self.positions=None
        if codec.needs_positions:
            dtype=torch.int16 if self.plan['position_dtype']=='torch.int16' else torch.int32
            self.positions=torch.empty(3,meta.capacity,dtype=dtype,device='cuda');self.registry.own('cache/positions',self.positions);self.registry.own('codec/inv_freq',self.inv_freq,'codec')
        self.registry.observe('request_initialized')
    def capture_positions(self,positions):
        assert positions.shape[:2]==(3,1);n=positions.shape[-1];assert self.seen_positions+n<=self.meta.capacity
        if self.positions is not None:
            x=positions[:,0];limit=torch.iinfo(self.positions.dtype);assert x.min()>=limit.min and x.max()<=limit.max
            self.positions[:,self.seen_positions:self.seen_positions+n].copy_(x)
        self.seen_positions+=n;self.registry.observe('position_append_in_place')
    def _time(self,fn,kind):
        self.peak_allocated=max(self.peak_allocated,torch.cuda.max_memory_allocated());self.peak_reserved=max(self.peak_reserved,torch.cuda.max_memory_reserved())
        torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);s.record()
        result=fn();e.record();self.events.append((kind,s,e));peak=torch.cuda.max_memory_allocated();self.workspace_peak=max(self.workspace_peak,max(0,peak-base))
        self.peak_allocated=max(self.peak_allocated,peak);self.peak_reserved=max(self.peak_reserved,torch.cuda.max_memory_reserved());return result
    def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
        if layer_idx not in self.states:return DynamicCache.update(self,key_states,value_states,layer_idx,cache_kwargs)
        state=self.states[layer_idx];pos=self.positions[:,:state['n']] if self.positions is not None else None
        k,v=self._time(lambda:self.codec.decode(layer_idx,state,pos,self.inv_freq,self.attention_scaling),'decode');self.read_calls+=1
        self.materialized_peak_bytes=max(self.materialized_peak_bytes,k.numel()*k.element_size()+v.numel()*v.element_size())
        self.incoming[layer_idx]=(key_states,value_states)
        if layer_idx==0:self._seen_tokens+=key_states.shape[-2]
        return torch.cat([k,key_states],-2),torch.cat([v,value_states],-2)
    def encode_after_attention(self,l):
        if l in self.incoming:k,v=self.incoming.pop(l)
        else:k,v=self.key_cache[l],self.value_cache[l]
        raw=self.pre.pop(l,None)
        if l not in self.states:self.states[l]=self.codec.allocate_state(l,self.registry)
        self.states[l]=self._time(lambda:self.codec.append(l,self.states[l],k,v,raw),'encode')
        self.key_cache[l]=torch.empty(0,device=k.device,dtype=k.dtype);self.value_cache[l]=torch.empty(0,device=v.device,dtype=v.dtype)
        self.registry.observe('encode_append_complete_layer_'+str(l))
    def assert_compact(self):
        assert not self.pre and not self.incoming and len(self.states)==36
        assert all(x.numel()==0 for x in self.key_cache+self.value_cache)
        assert self.read_calls>=36 and all(x['n']==self.seen_positions for x in self.states.values())
    def ledger(self):
        r=self.registry.receipt();rows=storages(self.registry.roots);r.update(cuda_bytes=r['resident_end_bytes'],static_cuda_codec_bytes=r['codec_static_bytes'],request_cuda_bytes=r['resident_end_bytes']-r['codec_static_bytes'],logical_elements=sum(2*2*x['n']*128 for x in self.states.values()),materialized_layer_KV_peak_bytes=self.materialized_peak_bytes)
        return r,rows
    def receipt(self):
        torch.cuda.synchronize();r=self.registry.receipt();payload=sum(s['arena'].payload_bytes for s in self.states.values())
        r.update(cache_payload_bytes=payload,cache_metadata_and_capacity_bytes=r['resident_end_bytes']-r['codec_static_bytes']-payload,codec_workspace_peak_bytes=self.workspace_peak,workspace_peak_bytes=self.workspace_peak,workspace_scope='maximum additional allocated bytes inside encode/decode after persistent arenas are registered; current-layer materialized KV included; allocator block rounding included',process_peak_allocated_bytes=max(self.peak_allocated,torch.cuda.max_memory_allocated()),process_peak_reserved_bytes=max(self.peak_reserved,torch.cuda.max_memory_reserved()),codec_load_ms=self.codec.load_ms,encode_ms=sum(s.elapsed_time(e) for kind,s,e in self.events if kind=='encode'),decode_ms=sum(s.elapsed_time(e) for kind,s,e in self.events if kind=='decode'),layout={k:v for k,v in self.plan.items() if k!='description'},arena_layout=self.states[0]['arena'].layout,all_36_layers_packed=len(self.states)==36,no_persistent_dense_history=all(x.numel()==0 for x in self.key_cache+self.value_cache),packed_history_read_calls=self.read_calls,initialization_resident_peak_bytes=self.codec.load_registry.peak,allocation_contract='all persistent arenas allocated once; append writes in place; FP band migration buffer permanently registered; no old/new arena replacement')
        assert r['resident_peak_bytes']>=r['initialization_resident_peak_bytes'];return r
