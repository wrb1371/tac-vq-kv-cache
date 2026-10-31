"""Common dense-if-feasible policy, always paying the entire frozen codec bank."""
import torch
from r15kv.codecs import PackageCodec,NOVABudget
from r16kv.nova import NOVAAllocated
from r16kv.nsn64 import NSN64Reference
from ar17kv.factorized import CompactPrivate
from ar17kv.text_protection import ProductTextProtected
from ar18kv.codecs import ProtectedAffine,ProtectedAllocated
from ar18kv.packed import ProtectedPacked
from r15kv.persistent_registry import Arena
class DenseFirst:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._base_needs_positions=self.needs_positions;self._hybrid_dense=False
    def prepare_request(self,meta,registry):
        self._hybrid_dense=self.static_bytes+36864*meta.capacity<=self.budget_bytes
        self.needs_positions=False if self._hybrid_dense else self._base_needs_positions
        if self._hybrid_dense:
            self._text_gpu=None;self._text_list=[]
            return PackageCodec.prepare_request(self,meta,registry)
        return super().prepare_request(meta,registry)
    def make_plan(self,meta):
        if self._hybrid_dense:
            desc=[(side+'_BF16',(2,meta.capacity,128),torch.bfloat16,'payload') for side in ['K','V']];total=self.static_bytes+36*Arena.bytes(desc);assert total<=self.budget_bytes
            return {'description':desc,'capacity':meta.capacity,'hybrid_dense':True,'dense_K_representation':'native post-MRoPE BF16','fixed_codec_still_fully_loaded_and_paid':True,'estimate':{'codec_static_bytes':self.static_bytes,'global_request_bytes':0,'layer_arena_bytes':Arena.bytes(desc),'resident_capacity_bytes':total,'kind':'INTEGER_ESTIMATE_NOT_MEASUREMENT'}}
        plan=super().make_plan(meta);plan.update(hybrid_dense=False,fixed_codec_still_fully_loaded_and_paid=True);return plan
    def allocate_state(self,l,registry):
        if not self._hybrid_dense:return super().allocate_state(l,registry)
        arena=Arena(self.plan['description'],'cuda',registry,f'cache/layer{l:02}/arena');return {'arena':arena,'n':0,**arena.views}
    def append(self,l,state,kpost,v,pre=None):
        if not self._hybrid_dense:return super().append(l,state,kpost,v,pre)
        start=state['n'];n=v.shape[-2];state['K_BF16'][:,start:start+n].copy_(kpost[0]);state['V_BF16'][:,start:start+n].copy_(v[0]);state['n']+=n;return state
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        if not self._hybrid_dense:return super().decode(l,state,positions,inv_freq,attention_scaling)
        return state['K_BF16'][:,:state['n']][None],state['V_BF16'][:,:state['n']][None]
    def finish_request(self):
        super().finish_request();self.needs_positions=self._base_needs_positions;self._hybrid_dense=False
class HybridParent(DenseFirst,ProductTextProtected):pass
class HybridAffine(DenseFirst,ProtectedAffine):pass
class HybridAllocated(DenseFirst,ProtectedAllocated):pass
class HybridPacked(DenseFirst,ProtectedPacked):pass
class HybridNOVA(DenseFirst,NOVABudget):pass
class HybridNplus(DenseFirst,NOVAAllocated):pass
class HybridPrivate(DenseFirst,CompactPrivate):pass
class HybridNSN(DenseFirst,NSN64Reference):pass
