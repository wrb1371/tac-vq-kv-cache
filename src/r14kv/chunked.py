"""True causal4096 history propagation, retaining native AV embeddings/positions."""
import torch
from r13kv.codec import EvolutionCache
from .common import prefill

class OmniChunkCache(EvolutionCache):
    """Author codec plus an explicit unfused multi-token historical read adapter."""
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.chunk_history_reads=0;self.chunk_materialized_peak_bytes=0
    def update(self,k,v,layer_idx,cache_kwargs=None):
        n=k.shape[-2];old=self.get_seq_length(layer_idx)
        if n<=1 or old==0:return super().update(k,v,layer_idx,cache_kwargs)
        assert layer_idx not in self._packed_pending and layer_idx not in self._packed_decode_current
        oldk=self._unpack_layer(layer_idx,'k');oldv=self._unpack_layer(layer_idx,'v')
        self.chunk_materialized_peak_bytes=max(self.chunk_materialized_peak_bytes,sum(x.numel()*x.element_size() for x in [oldk,oldv]))
        self.chunk_history_reads+=1
        if layer_idx==0:self._seen_tokens+=n
        self._packed_seq_lens[layer_idx]=old+n;self._packed_pending[layer_idx]=(k,v,old)
        return torch.cat([oldk,k],-2),torch.cat([oldv,v],-2)

def prefill_chunked(model,inputs,cache,chunk_size=4096):
    captured={}
    class EmbeddingsReady(Exception):pass
    def capture(module,args,kwargs):
        assert not args and kwargs['past_key_values'] is cache
        assert kwargs['inputs_embeds'].shape[1]==inputs['input_ids'].shape[1]-1
        captured.update(kwargs);raise EmbeddingsReady()
    handle=model.thinker.model.register_forward_pre_hook(capture,with_kwargs=True)
    try:
        try:prefill(model,inputs,cache)
        except EmbeddingsReady:pass
    finally:handle.remove()
    assert captured and cache.get_seq_length()==0,'no full FP decoder prefill may precede chunk execution'
    emb=captured['inputs_embeds'];pos=captured['position_ids'];assert pos.shape==(3,1,emb.shape[1])
    length=emb.shape[1];attention=captured.get('attention_mask');chunks=[];positions=[]
    for begin in range(0,length,chunk_size):
        end=min(begin+chunk_size,length);p=pos[:,:,begin:end];positions.append(p)
        kwargs={'inputs_embeds':emb[:,begin:end],'position_ids':p,'past_key_values':cache,'use_cache':True,'return_dict':True,'cache_position':torch.arange(begin,end,device=emb.device),'attention_mask':attention[:,:end] if attention is not None else None}
        output=model.thinker.model(**kwargs);assert cache.get_seq_length()==end
        chunks.append({'start':begin,'end':end,'history_before':begin});del output
    assert torch.equal(torch.cat(positions,-1),pos)
    return {'protocol':'causal4096','chunk_size':chunk_size,'chunks':chunks,'native_three_axis_positions_preserved':True,'no_FP_full_decoder_prefill':True,'chunk_history_read_calls':getattr(cache,'chunk_history_reads',None),'materialized_chunk_history_peak_bytes':getattr(cache,'chunk_materialized_peak_bytes',None)}
