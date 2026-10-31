"""Real packed history; only a current layer is materialized for native FA2."""
import types,torch
from transformers.cache_utils import DynamicCache

class PackedReferenceCache(DynamicCache):
    def __init__(self,codec,inv_freq,attention_scaling=1.):
        super().__init__();self.codec=codec;self.inv_freq=inv_freq.detach().float().clone();self.attention_scaling=attention_scaling
        self.states={};self.pre={};self.incoming={};self.positions=None;self.read_calls=0;self.materialized_peak_bytes=0
    def capture_positions(self,positions):
        assert positions.shape[:2]==(3,1)
        x=positions[:,0];combined=x if self.positions is None else torch.cat([self.positions.long(),x],-1)
        assert combined.min()>=0
        self.positions=combined.to(torch.int16 if combined.max()<=32767 else torch.int32).contiguous()
    def get_seq_length(self,layer_idx=0):
        return self.states[layer_idx]['n'] if layer_idx in self.states else super().get_seq_length(layer_idx)
    def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
        if layer_idx not in self.states:return super().update(key_states,value_states,layer_idx,cache_kwargs)
        state=self.states[layer_idx]
        k,v=self.codec.decode(layer_idx,state,self.positions[:,:state['n']],self.inv_freq,self.attention_scaling)
        self.read_calls+=1;self.materialized_peak_bytes=max(self.materialized_peak_bytes,(k.numel()*k.element_size()+v.numel()*v.element_size()))
        self.incoming[layer_idx]=(key_states,value_states)
        if layer_idx==0:self._seen_tokens+=key_states.shape[-2]
        return torch.cat([k,key_states],-2),torch.cat([v,value_states],-2)
    def encode_after_attention(self,layer_idx):
        if layer_idx in self.incoming:k,v=self.incoming.pop(layer_idx)
        else:k,v=self.key_cache[layer_idx],self.value_cache[layer_idx]
        raw=self.pre.pop(layer_idx,None)
        self.states[layer_idx]=self.codec.append(layer_idx,self.states.get(layer_idx),k,v,raw)
        # No full-layer decoded copy remains in the cache after the read.
        self.key_cache[layer_idx]=torch.empty(0,device=k.device,dtype=k.dtype)
        self.value_cache[layer_idx]=torch.empty(0,device=v.device,dtype=v.dtype)
    def assert_compact(self):
        assert not self.pre and not self.incoming and len(self.states)==36
        assert all(x.numel()==0 for x in self.key_cache+self.value_cache)
        assert self.read_calls>=36 and all(x['n']==self.positions.shape[1] for x in self.states.values())
    def ledger(self):
        seen=set();rows=[]
        def walk(x,path,static=False):
            if torch.is_tensor(x):
                s=x.untyped_storage();key=(str(x.device),s.data_ptr(),s.nbytes())
                if not s.nbytes() or key in seen:return
                seen.add(key);rows.append({'path':path,'bytes':s.nbytes(),'device':str(x.device),'dtype':str(x.dtype),'shape':list(x.shape),'static':static})
            elif isinstance(x,dict):
                for k,v in x.items():walk(v,path+'/'+str(k),static)
            elif isinstance(x,(tuple,list)):
                for k,v in enumerate(x):walk(v,path+'/'+str(k),static)
        walk(self.states,'request');walk(self.positions,'request/positions');walk(self.inv_freq,'codec/inv_freq',True)
        walk(self.codec.tables,'codec/tables',True)
        gpu=sum(x['bytes'] for x in rows if x['device'].startswith('cuda'));cpu=sum(x['bytes'] for x in rows if x['device']=='cpu')
        static=sum(x['bytes'] for x in rows if x['static'] and x['device'].startswith('cuda'))
        return {'cuda_bytes':gpu,'cpu_bytes':cpu,'static_cuda_codec_bytes':static,'request_cuda_bytes':gpu-static,
            'logical_elements':sum(2*2*x['n']*128 for x in self.states.values()),'materialized_layer_KV_peak_bytes':self.materialized_peak_bytes},rows

def install_reference_attention(model):
    handles=[]
    for l,layer in enumerate(model.thinker.model.layers):
        attn=layer.self_attn
        if hasattr(attn,'_r14_original_forward'):continue
        attn._r14_original_forward=attn.forward;attn._r14_capture_owner=None
        def raw_capture(module,args,output,attn=attn,l=l):
            cache=attn._r14_capture_owner
            if isinstance(cache,PackedReferenceCache) and cache.codec.method.startswith('GSRQ'):
                cache.pre[l]=output.view(1,output.shape[1],2,128).transpose(1,2)
        handles.append(attn.k_proj.register_forward_hook(raw_capture))
        def wrapped(self,*args,**kwargs):
            cache=kwargs.get('past_key_value')
            if not isinstance(cache,PackedReferenceCache):return self._r14_original_forward(*args,**kwargs)
            if self.layer_idx==0:cache.capture_positions(kwargs['position_ids'])
            self._r14_capture_owner=cache
            try:
                result=self._r14_original_forward(*args,**kwargs)
                cache.encode_after_attention(self.layer_idx)
                return result
            finally:self._r14_capture_owner=None
        attn.forward=types.MethodType(wrapped,attn)
    return handles
