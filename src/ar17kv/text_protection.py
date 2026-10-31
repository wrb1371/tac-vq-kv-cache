"""Paid exact native text/special KV overrides in the media prefix; no teacher copy."""
import bisect,torch
from r15kv.persistent_registry import Arena
from ar17kv.psrvq import ProductRVQ
from ar17kv.factorized import CompactPrivate
class PaidPrefixText:
    def prepare_request(self,meta,registry):
        tags=self.context['tags'];media=(tags!=0).nonzero().flatten();assert media.numel()>0;end=int(media[-1])+1;self._text_list=(tags[:end]==0).nonzero().flatten().tolist();self._text_gpu=None
        plan=super().prepare_request(meta,registry);self._text_gpu=torch.tensor(self._text_list,device='cuda',dtype=torch.int32);registry.own('cache/shared_prefix_text_positions',self._text_gpu);return plan
    def make_plan(self,meta):
        n=len(self._text_list);extra_desc=[(k,(2,n,128),torch.bfloat16,'payload') for k in ['text_FP_K','text_FP_V']];per_layer=Arena.bytes(extra_desc);extra=36*per_layer+4*n;original=self.budget_bytes;self.budget_bytes-=extra
        try:plan=super().make_plan(meta)
        finally:self.budget_bytes=original
        before=Arena.bytes(plan['description']);assert Arena.bytes(plan['description']+extra_desc)-before==per_layer;plan['description']=plan['description']+extra_desc
        if 'descriptions' in plan:plan['descriptions']=[d+extra_desc for d in plan['descriptions']]
        plan['estimate']['resident_capacity_bytes']+=extra;plan['estimate']['global_request_bytes']+=4*n
        if 'layer_arena_bytes' in plan['estimate']:plan['estimate']['layer_arena_bytes']+=per_layer
        plan.update(protected_prefix_text_tokens=n,protection_extra_bytes=extra,protection_rule='native text/special positions before last media feature; query text not protected',index_slots_retained_and_paid=True);return plan
    def allocate_state(self,l,registry):
        state=super().allocate_state(l,registry);state['text_filled']=0;return state
    def append(self,l,state,kpost,v,pre=None):
        start=state['n'];n=kpost.shape[-2];lo=bisect.bisect_left(self._text_list,start);hi=bisect.bisect_left(self._text_list,start+n);assert state['text_filled']==lo;state=super().append(l,state,kpost,v,pre)
        if hi>lo:
            local=self._text_gpu[lo:hi].long()-start;state['text_FP_K'][:,lo:hi].copy_(kpost[0].index_select(1,local));state['text_FP_V'][:,lo:hi].copy_(v[0].index_select(1,local))
        state['text_filled']=hi;return state
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        k,v=super().decode(l,state,positions,inv_freq,attention_scaling);m=bisect.bisect_left(self._text_list,state['n']);assert m==state['text_filled']
        if m:
            ix=self._text_gpu[:m].long();k[0].index_copy_(1,ix,state['text_FP_K'][:,:m]);v[0].index_copy_(1,ix,state['text_FP_V'][:,:m])
        return k,v
    def finish_request(self):
        self._text_gpu=None;self._text_list=[];super().finish_request()
class ProductTextProtected(PaidPrefixText,ProductRVQ):
    method='PSRVQ-paid-native-prefix-text';needs_pre_rope_K=True
class PrivateTextProtected(PaidPrefixText,CompactPrivate):
    method='GSRQ-paid-native-prefix-text'
