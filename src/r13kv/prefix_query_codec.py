"""Preparation-stage prefix-query conditioned pre-RoPE encoder.

Extends the exact existing six-bit representation; no decoder changes.
Metrics are CPU request state and all encoder copies/workspace are transient.
"""
import contextlib,torch
from .prerope_codec import PreRCache,PreRLayer,install_capture
from .prerope_packing import pack6
from model.codec.modality import TEXT
from .prefix_query_metric import prefix_query_metric,assign_position_weighted

class PrefixQueryCache(PreRCache):
    def __init__(self,*args,prefix_query_assignment=False,**kwargs):
        super().__init__(*args,**kwargs);self.prefix_query_assignment=prefix_query_assignment
        self._preR_prefix_metrics={};self._preR_prefix_query_counts={}
    def observe_prefix_queries(self,query,layer):
        if not self.prefix_query_assignment or layer in self._preR_prefix_metrics:return
        tags=self._chunk_tags(0,query.shape[2]);ids=(tags==TEXT).nonzero(as_tuple=True)[0][-16:]
        assert ids.numel()>0,'native MCQ prefix contains text queries'
        selected=query.index_select(2,ids.to(query.device))
        self._preR_prefix_metrics[layer]=prefix_query_metric(selected)
        self._preR_prefix_query_counts[layer]=8*len(ids)
    def _pack_append(self,x,offset,layer_idx,which):
        if which!='k' or not self.prefix_query_assignment:return super()._pack_append(x,offset,layer_idx,which)
        raw=self._preR_raw.pop(layer_idx);assert raw.shape==x.shape
        metric=self._preR_prefix_metrics[layer_idx];assert not metric.is_cuda
        b,h,t,d=raw.shape;assert (b,h,d)==(1,2,128)
        key=('preR',layer_idx)
        if key not in self._packed_codebooks:self._packed_codebooks[key]=self.preR_books[layer_idx].to(raw.device)
        book=self._packed_codebooks[key];source=raw.float();norm=source.norm(dim=-1).clamp_min(1e-12);unit=source/norm.unsqueeze(-1)
        if layer_idx not in self._packed['k']:self._packed['k'][layer_idx]=PreRLayer(b,h,d,raw.dtype,raw.device)
        tags=self._chunk_tags(offset,t)
        for mod in tags.unique().tolist():
            ids=(tags==mod).nonzero(as_tuple=True)[0];dev=ids.to(raw.device)
            coordinates=self.preR_coordinates.index_select(1,dev+offset)
            labels=assign_position_weighted(unit.index_select(2,dev),book,metric,coordinates,self.preR_cos,self.preR_sin)
            self._packed['k'][layer_idx].group(mod).append(pack6(labels),norm.index_select(2,dev).half(),ids+offset)
            elements=b*h*len(ids)*d;self._elems['k'][mod]=self._elems['k'].get(mod,0)+elements;self._bits['k'][mod]=self._bits['k'].get(mod,0)+elements*(1.5+16/d)
    def summary(self):
        d=super().summary()
        if self.prefix_query_assignment:d.update(prefix_query_conditioned_assignment=True,
            prefix_query_metric_residency='CPU per-request; no resident GPU query metric',
            query_access='last<=16 available text queries in unmodified native prefill, no answer queries required',
            assignment_proxy='perkey-position4x4 quadratic, FP32 linear RoPE; ignores global norm/crossgroup/BF16 nonlinearities')
        return d

@contextlib.contextmanager
def observe_prefix_queries(get_cache):
    import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as hf
    original=hf.apply_multimodal_rotary_pos_emb;count=0
    def wrapped(q,k,cos,sin,mrope_section,unsqueeze_dim=1):
        nonlocal count
        qe,ke=original(q,k,cos,sin,mrope_section,unsqueeze_dim)
        cache=get_cache()
        if isinstance(cache,PrefixQueryCache):
            matches=[layer for layer,raw in cache._preR_raw.items()
                if raw.data_ptr()==k.data_ptr() and raw.shape==k.shape and raw.stride()==k.stride()]
            assert len(matches)==1,('native K tensor must identify exactly one captured layer',matches)
            layer=matches[0];assert layer==count%36,('observed call order disagrees with actual K owner',layer,count)
            cache.observe_prefix_queries(qe,layer)
        count+=1;return qe,ke
    hf.apply_multimodal_rotary_pos_emb=wrapped
    try:
        yield
        assert count in [36,72],('expected one prefill optionally plus one answer-producing current step',count)
    finally:hf.apply_multimodal_rotary_pos_emb=original
