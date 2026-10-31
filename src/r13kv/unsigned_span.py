"""Omni local K group with a same-width unsigned span codec.

Derived incremental layout/packing semantics: kaistmm/OmniKVQuant,
Apache-2.0, source commit e8d880faae1a35275c8abe41003c8d8d2a8c8b51.
Only the positive span's FP4 level set differs. All indices are still 2-bit.
"""
import math
import torch
from model.codec.packed_cache import PackedGroup,PackedLayer,CODEC_LOCAL_LM,pack_unsigned,unpack_unsigned,_encode_mxfp4,_decode_mxfp4
from model.codec.turboquant import lloyd_max
from .metadata import encode_unsigned4,decode_unsigned4


class UnsignedSpanGroup(PackedGroup):
    def append_local_lm(self,values,norms,positions):
        assert self.codec==CODEC_LOCAL_LM
        assert values.ndim==4 and values.shape[:2]==(self.batch,self.heads)
        assert values.shape[-1]==self.dim and norms.shape==values.shape[:-1]
        n=values.shape[2];assert n==positions.numel()
        if not n:return
        tail=self.length%32;prefix=self.length-tail
        source=values.to(torch.float16);new_norms=norms.to(torch.float16)
        new_positions=positions.to(self.device,dtype=torch.int32)
        if tail:
            assert self.local_pending.shape==(self.batch,self.heads,tail,self.dim)
            source=torch.cat((self.local_pending,source),2)
            new_norms=torch.cat((self.norm[:,:,prefix:self.length].clone(),new_norms),2)
            new_positions=torch.cat((self.positions[prefix:self.length].clone(),new_positions))
        updated=source.shape[2];self._reserve(prefix+updated)
        centroids=lloyd_max(self.dim)[0].to(self.device)
        levels=(centroids-centroids[0])/(centroids[-1]-centroids[0]);boundaries=(levels[:-1]+levels[1:])/2
        groups=math.ceil(updated/32);padded_tokens=groups*32
        padded=torch.zeros(self.batch,self.heads,padded_tokens,self.dim,dtype=source.dtype,device=self.device)
        padded[:,:,:updated].copy_(source)
        grouped=padded.reshape(self.batch,self.heads,groups,32,self.dim).float()
        valid=(torch.arange(padded_tokens,device=self.device)<updated).reshape(1,1,groups,32,1)
        minimum=torch.where(valid,grouped,torch.full_like(grouped,float('inf'))).amin(3)
        maximum=torch.where(valid,grouped,torch.full_like(grouped,float('-inf'))).amax(3)
        raw_span=(maximum-minimum).clamp_min_(1e-12)
        min_code,min_scale,reconstructed_min=_encode_mxfp4(minimum)
        span_code,span_scale,reconstructed_span=encode_unsigned4(raw_span)
        target=slice(prefix//32,prefix//32+groups)
        self.local_min[:,:,target].copy_(min_code);self.local_min_scale[:,:,target].copy_(min_scale)
        self.local_span[:,:,target].copy_(span_code);self.local_span_scale[:,:,target].copy_(span_scale)
        if source.is_cuda:
            from model.codec.triton_store import pack_local_lm_into
            pack_local_lm_into(source,new_norms,boundaries,self.index,self.norm,token_offset=prefix,
                metadata_minimum=reconstructed_min.to(torch.float16).contiguous(),
                metadata_span=reconstructed_span.to(torch.float16).contiguous())
        else:
            group_ids=torch.arange(updated,device=self.device)//32
            minimum=reconstructed_min.index_select(2,group_ids);span=reconstructed_span.index_select(2,group_ids)
            safe=torch.where(span==0,torch.ones_like(span),span)
            unit=((source.float()-minimum)/safe).clamp_(0,1)
            index=torch.bucketize(unit.contiguous(),boundaries,right=True)
            self.index[:,:,prefix:prefix+updated].copy_(pack_unsigned(index))
            self.norm[:,:,prefix:prefix+updated].copy_(new_norms)
        self.positions[prefix:prefix+updated].copy_(new_positions)
        self.length=prefix+updated;pending=self.length%32
        self.local_pending=source[:,:,-pending:].clone() if pending else torch.empty(0,dtype=torch.float16,device=self.device)

    def unpack_rotated(self):
        n=self.length;code=unpack_unsigned(self.index[:,:,:n]).long()
        centroids=lloyd_max(self.dim)[0].to(self.device);levels=(centroids-centroids[0])/(centroids[-1]-centroids[0])
        group_ids=torch.arange(n,device=self.device)//32
        minimum=_decode_mxfp4(self.local_min,self.local_min_scale,self.dim).index_select(2,group_ids)
        span=decode_unsigned4(self.local_span,self.local_span_scale,self.dim).index_select(2,group_ids)
        values=minimum+span*levels[code]
        return values*self.norm[:,:,:n].float().unsqueeze(-1),self.positions[:n].long()


class UnsignedSpanLayer(PackedLayer):
    def group(self,mod,codec):
        if mod not in self.groups:
            cls=UnsignedSpanGroup if codec==CODEC_LOCAL_LM else PackedGroup
            self.groups[mod]=cls(mod,codec,self.batch,self.heads,self.dim,self.device)
        assert self.groups[mod].codec==codec
        return self.groups[mod]
