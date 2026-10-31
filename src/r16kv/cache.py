"""Inherited accounting plus heterogeneous-layout reporting and shared positions."""
from r15kv.cache import BudgetCache
class R16Cache(BudgetCache):
    def __init__(self,codec,inv_freq,meta,attention_scaling=1.):
        super().__init__(codec,inv_freq,meta,attention_scaling)
        if codec.package['family']=='NSN':
            codec.position_storage=self.positions;codec.position_inv_freq=self.inv_freq;codec.position_attention_scaling=attention_scaling
    def receipt(self):
        plan=self.codec.plan;filtered={k:v for k,v in plan.items() if k not in ['modes','descriptions']};self.codec.plan=filtered
        try:r=super().receipt()
        finally:self.codec.plan=plan
        r.update(system_cap_verified=False,subsystem_simultaneous_peak_bytes=None,
                 prompt_tokens=self.meta.prompt_tokens,prefill_cached_tokens=self.meta.prompt_tokens-1,
                 actual_cached_tokens=self.seen_positions,capacity_tokens=self.meta.capacity,generation_cap=self.meta.generation_cap,
                 codec_unique_bytes=r['codec_static_bytes'],resident_high_water_bytes=r['resident_peak_bytes'],
                 codec_workspace_high_water_bytes=r['codec_workspace_peak_bytes'],
                 budget_resident_bytes=self.codec.budget_bytes,
                 plan_action_counts={'V2_tokens_across_layers':plan.get('V2_tokens_across_layers'),'low_tokens_across_layers':plan.get('low_tokens_across_layers'),'R_K':plan.get('R_K'),'R_V':plan.get('R_V')})
        return r
