"""One paid codec bank; select representation by exact admitted rate, not QA."""
from .packed_sparse import PackedSparse
from ar17kv.text_protection import PaidPrefixText
from ar18kv.hybrid import DenseFirst

class RateGated(PackedSparse):
    method='AR19-rate-aware-parent-or-packed-temporal'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._route=None
    def prepare_request(self,meta,registry):
        tags=self.context['tags'];self._video_list=(tags==2).nonzero().flatten().tolist();end=self._video_list[-1]+1
        self._text_list=(tags[:end]==0).nonzero().flatten().tolist();grid=self.context['video_grid_thw'];assert grid.shape==(1,3)
        self._nt,self._h,self._w=map(int,grid[0].tolist());self._h//=2;self._w//=2;assert len(self._video_list)==self._nt*self._h*self._w
        self._vid_gpu=None;parent=None;temporal=None
        def probe(fn):
            try:return fn(self,meta)
            except RuntimeError as e:
                if 'INFEASIBLE_BY_INTEGER_ORACLE' in str(e):return None
                raise
        parent=probe(PaidPrefixText.make_plan);temporal=probe(PackedSparse.make_plan)
        self._parent_sum=None if parent is None else parent['R_K']+parent['R_V'];self._temporal_sum=None if temporal is None else temporal['R_K']+temporal['R_V']
        full=parent is not None and parent['R_K']==self.package['Rmax_K'] and parent['R_V']==self.package['Rmax_V']
        use_parent=parent is not None and (temporal is None or full or (self.policy.get('gate_rule','rate_gain')=='rate_gain' and self._parent_sum>=self._temporal_sum))
        self._route='parent' if use_parent else 'packed_temporal'
        if use_parent:plan=PaidPrefixText.prepare_request(self,meta,registry)
        else:plan=super().prepare_request(meta,registry)
        plan.update(selected_route=self._route,parent_stage_sum=self._parent_sum,temporal_stage_sum=self._temporal_sum,gate_rule=self.policy.get('gate_rule','rate_gain'),no_QA_used_to_select_representation=True)
        return plan
    def make_plan(self,meta):
        if self._route=='parent':return PaidPrefixText.make_plan(self,meta)
        return PackedSparse.make_plan(self,meta)
    def allocate_state(self,l,registry):
        if self._route=='parent':return PaidPrefixText.allocate_state(self,l,registry)
        return super().allocate_state(l,registry)
    def append(self,l,state,kpost,v,pre=None):
        if self._route=='parent':return PaidPrefixText.append(self,l,state,kpost,v,pre)
        return super().append(l,state,kpost,v,pre)
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        if self._route=='parent':return PaidPrefixText.decode(self,l,state,positions,inv_freq,attention_scaling)
        return super().decode(l,state,positions,inv_freq,attention_scaling)
    def finish_request(self):
        super().finish_request();self._route=None

class HybridRateGated(DenseFirst,RateGated):pass
