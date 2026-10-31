"""Bounded codec mutations around the unchanged complete OmniKVQuant parent.

The first optional component restores a completed local K block's direction
norm through the existing norm slot. Partial groups retain original semantics
because the author's incremental encoder reuses their original norms.
"""
import torch
import triton
import triton.language as tl

from model.codec.kv_cache import OmniKVQuantCache
from model.codec.packed_cache import CODEC_LOCAL_LM,unpack_unsigned,_decode_mxfp4
from model.codec.turboquant import lloyd_max
from model.codec.triton_packed_attention import _decode_mxfp4_metadata
from .metadata import decode_unsigned4,decode_unsigned4_triton
from .unsigned_span import UnsignedSpanGroup,UnsignedSpanLayer
from .video_order import pack_reordered


@triton.jit(do_not_specialize=('START','CAP','GCAP'))
def _restore_closed_direction_norm(index_ptr,norm_ptr,min_ptr,span_ptr,min_exp_ptr,span_exp_ptr,levels_ptr,
                                   START,CAP,GCAP,D:tl.constexpr,UNSIGNED_SPAN:tl.constexpr):
    bh=tl.program_id(0);token=START+tl.program_id(1);d=tl.arange(0,D)
    packed=tl.load(index_ptr+(bh*CAP+token)*(D//4)+d//4)
    index=(packed>>((d%4)*2))&3
    g=token//32
    min_packed=tl.load(min_ptr+(bh*GCAP+g)*(D//2)+d//2)
    span_packed=tl.load(span_ptr+(bh*GCAP+g)*(D//2)+d//2)
    min_code=(min_packed>>((d%2)*4))&15
    span_code=(span_packed>>((d%2)*4))&15
    min_exp=tl.load(min_exp_ptr+(bh*GCAP+g)*(D//32)+d//32)
    span_exp=tl.load(span_exp_ptr+(bh*GCAP+g)*(D//32)+d//32)
    minimum=_decode_mxfp4_metadata(min_code,min_exp)
    if UNSIGNED_SPAN:span=decode_unsigned4_triton(span_code,span_exp)
    else:span=_decode_mxfp4_metadata(span_code,span_exp)
    level=tl.load(levels_ptr+index)
    direction=minimum+span*level
    scale=tl.sqrt(tl.sum(direction*direction,axis=0))
    original=tl.load(norm_ptr+bh*CAP+token).to(tl.float32)
    corrected=tl.where(scale>1.e-12,original/scale,original)
    tl.store(norm_ptr+bh*CAP+token,corrected)


def correct_closed_norm(group,start,end):
    assert start%32==end%32==0 and 0<=start<=end<=group.length
    if start==end:return
    levels=lloyd_max(group.dim)[0].to(group.device)
    levels=(levels-levels[0])/(levels[-1]-levels[0])
    if group.norm.is_cuda:
        _restore_closed_direction_norm[(group.batch*group.heads,end-start)](
            group.index,group.norm,group.local_min,group.local_span,group.local_min_scale,group.local_span_scale,levels,
            START=start,CAP=group.capacity,GCAP=group.local_min.shape[2],D=group.dim,
            UNSIGNED_SPAN=isinstance(group,UnsignedSpanGroup),num_warps=4)
    else:
        code=unpack_unsigned(group.index[:,:,start:end]).long()
        positions=torch.arange(start,end,device=group.device)//32
        minimum=_decode_mxfp4(group.local_min,group.local_min_scale,group.dim).index_select(2,positions)
        decoder=decode_unsigned4 if isinstance(group,UnsignedSpanGroup) else _decode_mxfp4
        span=decoder(group.local_span,group.local_span_scale,group.dim).index_select(2,positions)
        unit=minimum+span*levels[code]
        scale=unit.norm(dim=-1)
        before=group.norm[:,:,start:end].float()
        group.norm[:,:,start:end].copy_(torch.where(scale>1e-12,before/scale,before))


class EvolutionCache(OmniKVQuantCache):
    def __init__(self,cfg,*,closed_key_norm=False,unsigned_span=False,reorder_video=False,video_grid=None,key_vq_books=None,optimal_value_radius=False,shared_value_basis=False,key_rotation=None,global_scalar_key=False,assignment_metric=None,share_key_rotation=False):
        super().__init__(cfg)
        self.closed_key_norm=closed_key_norm
        self.unsigned_span=unsigned_span
        self.reorder_video=reorder_video
        self.video_grid=video_grid
        self.key_vq_books=key_vq_books
        self.optimal_value_radius=optimal_value_radius
        self.shared_value_basis=shared_value_basis
        self.key_rotation=key_rotation
        self.global_scalar_key=global_scalar_key
        self.assignment_metric=assignment_metric
        self.share_key_rotation=share_key_rotation
        assert not global_scalar_key or not (closed_key_norm or unsigned_span or reorder_video or key_vq_books is not None),'complete global K replacement: incompatible local/VQ flags'
        if assignment_metric is not None:
            assert key_vq_books is not None and assignment_metric.shape==(36,2,32,4,4) and assignment_metric.dtype==torch.float32 and not assignment_metric.is_cuda
        assert self.key_rotation is None or self.key_vq_books is None,'K books must match their fitted coordinates'
        self._closed_norm_corrected_vectors=0

    def _shared_rot(self,layer_idx,heads,dim,device,which='k'):
        if self.share_key_rotation and which=='k':
            # All layers use exactly the same original H or fixed custom RHT.
            # Share only across layers; retain both physical heads and shape.
            key=('exact_shared_K_rotation',which,str(device),heads,dim)
            if key not in self._rots:
                if self.key_rotation is None:
                    from model.codec.turboquant import hadamard
                    self._rots[key]=hadamard(dim).expand(heads,dim,dim).contiguous().to(device)
                else:
                    assert self.key_rotation.shape==(heads,dim,dim)
                    self._rots[key]=self.key_rotation.to(device)
            return self._rots[key]
        if self.key_rotation is None or which!='k':return super()._shared_rot(layer_idx,heads,dim,device,which)
        assert self.key_rotation.shape==(heads,dim,dim)
        key=('fixed_key_rotation',layer_idx,which,str(device))
        if key not in self._rots:self._rots[key]=self.key_rotation.to(device)
        return self._rots[key]

    def _rot(self,layer_idx,mod,heads,dim,device,which='k'):
        if which!='v' or not self.shared_value_basis:
            return super()._rot(layer_idx,mod,heads,dim,device,which)
        assert self.cfg.method=='omnikvquant'
        key=('v_shared',layer_idx,mod,str(device))
        if key not in self._rots:
            first=self.cfg.rotations_v[layer_idx,0,mod]
            assert all(torch.equal(first,self.cfg.rotations_v[layer_idx,h,mod]) for h in range(heads)),'shared artifact must retain the same fitted basis for all heads'
            self._rots[key]=first.float().to(device).unsqueeze(0).expand(heads,dim,dim)
        return self._rots[key]

    def _pack_append(self,x,offset,layer_idx,which):
        if self.global_scalar_key and which=='k':
            from .global_key import append_global_key
            return append_global_key(self,x,offset,layer_idx)
        if self.assignment_metric is not None and which=='k':
            from .query_metric import append_query_metric
            return append_query_metric(self,x,offset,layer_idx)
        if self.optimal_value_radius and which=='v':
            from .value_radius import append_optimal_radius
            def original_value():
                if self.reorder_video:return pack_reordered(self,x,offset,layer_idx,'v')
                return super(EvolutionCache,self)._pack_append(x,offset,layer_idx,'v')
            return append_optimal_radius(self,x,offset,layer_idx,original_value)
        if self.key_vq_books is not None and which=='k':
            from .key_vq import append_key_vq
            return append_key_vq(self,x,offset,layer_idx)
        if self.unsigned_span and which=='k' and self.cfg.method=='omnikvquant' and layer_idx not in self._packed['k']:
            batch,heads,_,dim=x.shape
            self._packed['k'][layer_idx]=UnsignedSpanLayer(batch,heads,dim,x.dtype,x.device)
        def pack():
            if self.reorder_video:return pack_reordered(self,x,offset,layer_idx,which)
            return super(EvolutionCache,self)._pack_append(x,offset,layer_idx,which)
        if not self.closed_key_norm or which!='k' or self.cfg.method!='omnikvquant':return pack()
        old_layer=self._packed['k'].get(layer_idx)
        old={m:g.length//32*32 for m,g in old_layer.groups.items()} if old_layer else {}
        pack()
        for mod,group in self._packed['k'][layer_idx].groups.items():
            if group.codec!=CODEC_LOCAL_LM:continue
            first=old.get(mod,0);last=group.length//32*32
            correct_closed_norm(group,first,last)
            self._closed_norm_corrected_vectors+=(last-first)*group.batch*group.heads

    def summary(self):
        result=super().summary()
        result['closed_key_norm']=self.closed_key_norm
        result['unsigned_span']=self.unsigned_span
        result['reorder_video']=self.reorder_video
        result['optimal_value_radius']=self.optimal_value_radius
        result['shared_value_basis']=self.shared_value_basis
        result['fixed_custom_K_rotation']=self.key_rotation is not None
        if self.share_key_rotation:result['identical_K_rotation_storage_shared_across_layers']=True
        if self.global_scalar_key:
            result.update(key_codec='author_global_scalar_K_parent_modality_V',k_local_group_size=0,k_local_metadata=None)
        if self.assignment_metric is not None:
            result.update(query_metric_assignment=True,
                assignment_metric_residency='CPU; bounded transient encoder copies',
                assignment_objective='pergroup4x4 postH native query second moment; no crossgroup/global renormalization coupling')
        result['closed_norm_corrected_vectors']=self._closed_norm_corrected_vectors
        if self.key_vq_books is not None:
            result.update(key_codec='byte_index_VQ_g4_C256',k_local_group_size=0,
                          k_local_metadata=None,packed_fused_store=0,
                          encoder='bounded Torch reference assignment on deployed FP16 books')
        return result

    def _unpack_layer(self,layer_idx,which):
        if self.key_vq_books is not None and which=='k':
            from .key_vq import unpack_key_vq
            return unpack_key_vq(self,layer_idx)
        return super()._unpack_layer(layer_idx,which)


def install_component_reader():
    import model.codec.triton_packed_attention as native
    if hasattr(native,'_r13_original_reader'):return
    native._r13_original_reader=native.packed_decode_attention
    def read(query,cache,layer_idx,**kwargs):
        if getattr(cache,'preR_books',None) is not None:
            from .prerope_reader import packed_decode_attention
            return packed_decode_attention(query,cache,layer_idx,**kwargs)
        if getattr(cache,'key_vq_books',None) is not None:
            from .triton_vq_reader import packed_decode_attention
            return packed_decode_attention(query,cache,layer_idx,**kwargs)
        if getattr(cache,'unsigned_span',False):
            from .triton_unsigned_reader import packed_decode_attention
            return packed_decode_attention(query,cache,layer_idx,**kwargs)
        return native._r13_original_reader(query,cache,layer_idx,**kwargs)
    native.packed_decode_attention=read
