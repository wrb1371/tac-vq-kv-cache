"""Preparation-stage private6-bit pre-MRoPE K codec; no task result yet."""
import math,torch
from r13kv.codec import EvolutionCache
from model.codec.packed_cache import PackedLayer
from .prerope_packing import pack6,unpack6,paired_groups,undo_paired_groups


def apply_native_rope(pre,positions,cos_lut,sin_lut):
    # positions are genuine3-axis coordinates in original stream order.
    n=pre.shape[2];assert positions.shape==(3,n)
    axis=torch.tensor([0]*16+[1]*24+[2]*24,device=pre.device)
    pos=positions.long()[axis].T;freq=torch.arange(64,device=pre.device)
    cos=cos_lut[pos,freq[None]];sin=sin_lut[pos,freq[None]]
    cos=torch.cat((cos,cos),-1)[None,None];sin=torch.cat((sin,sin),-1)[None,None]
    x=pre.to(torch.bfloat16);other=torch.cat((-x[...,64:],x[...,:64]),-1)
    return x*cos+other*sin


class PreRGroup:
    def __init__(self,mod,batch,heads,dim,device):
        self.mod,self.batch,self.heads,self.dim,self.device=mod,batch,heads,dim,device
        self.codec='preR_private_g4_C64';self.length=self.capacity=0
        self.index=torch.empty(0,device=device,dtype=torch.uint8)
        self.norm=torch.empty(0,device=device,dtype=torch.float16)
        self.positions=torch.empty(0,device=device,dtype=torch.int32)
    @property
    def index_width(self):return 24
    def append(self,index,norm,positions):
        n=positions.numel();needed=self.length+n
        if needed>self.capacity:
            cap=needed if not self.capacity else max(needed,2*self.capacity)
            ix=torch.empty(self.batch,self.heads,cap,24,device=self.device,dtype=torch.uint8)
            no=torch.empty(self.batch,self.heads,cap,device=self.device,dtype=torch.float16)
            ps=torch.empty(cap,device=self.device,dtype=torch.int32)
            if self.length:
                ix[:,:,:self.length]=self.index[:,:,:self.length];no[:,:,:self.length]=self.norm[:,:,:self.length];ps[:self.length]=self.positions[:self.length]
            self.index,self.norm,self.positions,self.capacity=ix,no,ps,cap
        self.index[:,:,self.length:needed]=index;self.norm[:,:,self.length:needed]=norm
        self.positions[self.length:needed]=positions.to(self.device,dtype=torch.int32);self.length=needed
    def storage_bytes(self,*,live=False):
        return self.batch*self.heads*self.length*(24+2)+self.length*4 if live else sum(t.untyped_storage().nbytes() for t in [self.index,self.norm,self.positions])


class PreRLayer(PackedLayer):
    def group(self,mod,codec=None):
        if mod not in self.groups:self.groups[mod]=PreRGroup(mod,self.batch,self.heads,self.dim,self.device)
        return self.groups[mod]


def assign(unit,book,weights):
    assert unit.shape[0]==1 and book.shape==(2,32,64,4) and weights.shape==(2,64)
    grouped=paired_groups(unit);indices=torch.empty(1,2,unit.shape[2],32,device=unit.device,dtype=torch.uint8)
    for head in range(2):
        x=grouped[0,head].permute(1,0,2)
        w=weights[head].float().reshape(32,2).repeat(1,2).sqrt()
        c=book[head].float()*w[:,None,:]
        for start in range(0,x.shape[1],512):
            sub=x[:,start:start+512].float()*w[:,None,:]
            distance=sub.square().sum(-1,keepdim=True)+c.square().sum(-1)[:,None,:]-2*torch.bmm(sub,c.transpose(1,2))
            indices[0,head,start:start+len(sub[0])]=distance.argmin(-1).T.to(torch.uint8)
    return indices


def decode_pre(group,book):
    codes=unpack6(group.index[:,:,:group.length]).long();g=torch.arange(32,device=book.device)
    values=torch.stack([book[h][g[None,None,:],codes[:,h]] for h in range(2)],dim=1)
    x=undo_paired_groups(values.float());unit=x/x.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    return unit*group.norm[:,:,:group.length].float().unsqueeze(-1)


class PreRCache(EvolutionCache):
    def __init__(self,cfg,*,books,query_pair_weights,inv_freq,attention_scaling=1.,**kwargs):
        super().__init__(cfg,**kwargs)
        assert not self.closed_key_norm and not self.unsigned_span and not self.reorder_video and self.key_vq_books is None and self.key_rotation is None and not self.global_scalar_key and self.assignment_metric is None and not self.share_key_rotation,'complete K replacement must explicitly remove inapplicable old K components'
        assert books.shape==(36,2,32,64,4) and books.dtype==torch.float16
        self.preR_books=books;self.preR_query_pair_weights=query_pair_weights
        self.preR_inv_freq=inv_freq.detach().float().cpu().clone();self.preR_attention_scaling=float(attention_scaling)
        self.preR_coordinates=None;self.preR_cos=None;self.preR_sin=None;self.preR_lut_capacity=0
        self._preR_raw={}
    def capture_positions(self,positions,position_embeddings=None):
        assert positions.shape[0:2]==(3,1) and int(positions.min())>=0
        x=positions[:,0];maximum=int(x.max())
        if self.preR_coordinates is None:combined=x
        else:
            assert x.shape[-1]==1 and self.preR_coordinates.shape[-1]==self.get_seq_length(0)
            combined=torch.cat((self.preR_coordinates.long(),x),-1)
        dtype=torch.int16 if int(combined.max())<=32767 else torch.int32
        self.preR_coordinates=combined.to(dtype).contiguous()
        if maximum>=self.preR_lut_capacity:
            capacity=max(256,math.ceil((maximum+1)/256)*256)
            phase=torch.arange(capacity,device=x.device,dtype=torch.float32)[:,None]*self.preR_inv_freq.to(x.device)[None,:]
            self.preR_cos=(phase.cos()*self.preR_attention_scaling).bfloat16()
            self.preR_sin=(phase.sin()*self.preR_attention_scaling).bfloat16()
            self.preR_lut_capacity=capacity
        if position_embeddings is not None:
            cos,sin=position_embeddings
            for axis in range(3):
                assert torch.equal(self.preR_cos[x[axis].long()],cos[axis,0,:,:64]),'native LUT mismatch'
                assert torch.equal(self.preR_sin[x[axis].long()],sin[axis,0,:,:64]),'native LUT mismatch'
    def _pack_append(self,x,offset,layer_idx,which):
        if which!='k':return super()._pack_append(x,offset,layer_idx,which)
        raw=self._preR_raw.pop(layer_idx);assert raw.shape==x.shape and self.preR_coordinates is not None
        b,h,t,d=raw.shape;assert (b,h,d)==(1,2,128)
        key=('preR',layer_idx)
        if key not in self._packed_codebooks:self._packed_codebooks[key]=self.preR_books[layer_idx].to(raw.device)
        book=self._packed_codebooks[key];metric=self.preR_query_pair_weights[layer_idx].to(raw.device)
        source=raw.float();norm=source.norm(dim=-1).clamp_min(1e-12);unit=source/norm.unsqueeze(-1)
        if layer_idx not in self._packed['k']:self._packed['k'][layer_idx]=PreRLayer(b,h,d,raw.dtype,raw.device)
        tags=self._chunk_tags(offset,t)
        for mod in tags.unique().tolist():
            ids=(tags==mod).nonzero(as_tuple=True)[0];dev=ids.to(raw.device)
            codes=assign(unit.index_select(2,dev),book,metric);packed=pack6(codes)
            self._packed['k'][layer_idx].group(mod).append(packed,norm.index_select(2,dev).half(),ids+offset)
            elements=b*h*len(ids)*d;self._elems['k'][mod]=self._elems['k'].get(mod,0)+elements
            self._bits['k'][mod]=self._bits['k'].get(mod,0)+elements*(1.5+16/d)
    def _unpack_layer(self,layer_idx,which):
        if which!='k':return super()._unpack_layer(layer_idx,which)
        layer=self._packed['k'][layer_idx];total=self.get_seq_length(layer_idx)
        out=torch.empty(layer.batch,layer.heads,total,128,device=layer.device,dtype=torch.bfloat16)
        filled=torch.zeros(total,device=layer.device,dtype=torch.bool);book=self._packed_codebooks[('preR',layer_idx)]
        for group in layer.groups.values():
            positions=group.positions[:group.length].long();pre=decode_pre(group,book)
            value=apply_native_rope(pre,self.preR_coordinates.index_select(1,positions),self.preR_cos,self.preR_sin)
            out.index_copy_(2,positions,value);filled[positions]=True
        key,_,positions=self.current_token(layer_idx)
        if key is not None:out.index_copy_(2,positions.long(),key);filled[positions.long()]=True
        assert filled.all();return out
    def summary(self):
        result=super().summary();result.update(bits_k=1.5,payload_avg=1.75,k_local_group_size=0,k_local_metadata=None,
            key_codec='preR_private_two_RoPE_pairs_g4_C64_6bit',packed_fused_store=0,
            actual_mrope_coordinates=True,paid_LUT_capacity=self.preR_lut_capacity,encoder='bounded Torch reference VQ')
        return result


def install_capture(model):
    import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as hf
    hooks=[]
    for module in model.modules():
        if not isinstance(module,hf.Qwen2_5OmniAttention):continue
        def before(attn,args,kwargs):
            cache=kwargs.get('past_key_value');attn._r13proto_pre_owner=cache if isinstance(cache,PreRCache) else None
            if isinstance(cache,PreRCache) and attn.layer_idx==0:cache.capture_positions(kwargs['position_ids'],kwargs['position_embeddings'])
        def capture(proj,args,output,attn=module):
            cache=attn._r13proto_pre_owner;attn._r13proto_pre_owner=None # no retained cache/hidden history on model modules
            if cache is not None:
                b,t,_=output.shape;cache._preR_raw[attn.layer_idx]=output.view(b,t,2,128).transpose(1,2)
        hooks.extend([module.register_forward_pre_hook(before,with_kwargs=True),module.k_proj.register_forward_hook(capture)])
    assert len(hooks)==72;return hooks
