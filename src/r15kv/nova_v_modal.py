"""N2 paid donor geometry and a matched pooled-statistic/recipe control."""
import torch
from .codecs import NOVABudget
from .cal_diagnostics import hash_modalities
class NOVAConditionedV(NOVABudget):
    method='NOVA-R15-N2-paid-modality-V-donor'
    def __init__(self,package,budget_mib,**policy):
        super().__init__(package,budget_mib,**policy);assert self.tables[0]['V_rotation'].shape==(2,3,128,128)
    def set_context(self,**kwargs):
        super().set_context(**kwargs);tags=kwargs['tags'].cpu();self.source_tags=tags.clone()
        if self.policy.get('routing')=='hash':
            prefix=hash_modalities(tags[:-1],kwargs['input_sha256'] if 'input_sha256' in kwargs else __import__('hashlib').sha256(kwargs['input_ids'][0].cpu().numpy().tobytes()).hexdigest());self.source_tags[:-1]=prefix
    def extra_global_bytes(self,meta,description):return meta.capacity
    def prepare_request(self,meta,registry):
        plan=super().prepare_request(meta,registry);self.route_tags=torch.zeros(meta.capacity,dtype=torch.uint8,device='cuda');self.route_tags[:len(self.source_tags)].copy_(self.source_tags);registry.own('cache/N2_modality_or_hash_tags',self.route_tags);return plan
    @torch.no_grad()
    def rotate_v(self,l,v,start=0):
        tags=self.route_tags[self.plan['sink']+start:self.plan['sink']+start+v.shape[1]];out=torch.empty(v.shape,device=v.device,dtype=torch.float32)
        for mod in [0,1,2]:
            ids=(tags==mod).nonzero(as_tuple=True)[0]
            if ids.numel():out[:,ids]=torch.bmm(v.index_select(1,ids),self.tables[l]['V_rotation'][:,mod]).float()
        return out
    @torch.no_grad()
    def inverse_v(self,l,vr,start=0):
        tags=self.route_tags[self.plan['sink']+start:self.plan['sink']+start+vr.shape[1]];out=torch.empty_like(vr)
        for mod in [0,1,2]:
            ids=(tags==mod).nonzero(as_tuple=True)[0]
            if ids.numel():out[:,ids]=torch.bmm(vr.index_select(1,ids),self.tables[l]['V_rotation'][:,mod].float().transpose(1,2))
        return out
    def finish_request(self):
        self.route_tags=None;self.source_tags=None;super().finish_request()
class NOVAMatchedPooledDonor(NOVABudget):
    method='NOVA-R15-N2-matched-pooled-donor-control'
    def make_plan(self,meta):
        # Reserve only for selecting the SAME layout as three-basis controls.
        # This is not allocated padding and is excluded from measured/actual estimates.
        reserve=int(4.5*2**20)+meta.capacity;actual=self.static_bytes;self.static_bytes+=reserve
        try:plan=super().make_plan(meta)
        finally:self.static_bytes=actual
        plan['estimate']={**plan['estimate'],'codec_static_bytes':actual,'resident_capacity_bytes':plan['estimate']['resident_capacity_bytes']-reserve};plan['matching_layout_only_unallocated_reserve_bytes']=reserve;return plan
