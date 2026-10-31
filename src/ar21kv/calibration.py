"""Label-free rate curves from finite, source-disjoint native video prefixes."""
import torch
from ar20kv.exact_gemm import ExactPacked
from r14kv.fitting import nearest

class RateCurveCapture(ExactPacked):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.query_moments={};self.output_diagonals={};self.curve_records=[]
    @torch.no_grad()
    def _innovation(self,l,state,raw,slots,kind):
        n=raw.shape[1]
        if n:
            take=min(128,n);indices=torch.linspace(0,n-1,take,device=raw.device).long();x=raw[:,indices].float();t=self.tables[0];scale=t[kind+'_scale'][l].float()[:,None];u=x/scale;rho=u.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6).bfloat16();z=((u/rho.float())@t['H']).reshape(1,2*take*8,16)
            metric=self.query_moments[l]/128 if kind=='K' else self.output_diagonals[l]
            def error():
                raw_error=(z.reshape(2,take,128)@t['H'].T)*rho.float()*scale
                return float((raw_error.square()*metric[:,None]).sum(-1).mean())
            losses=[error()]
            for stage in range(6):
                cb=t[kind+'_books'][stage][None].float();assignment=nearest(z,cb);z-=torch.gather(cb,1,assignment[...,None].expand(-1,-1,16));losses.append(error())
            role='anchor' if len(slots)==self.plan['anchor_tokens'] and int(slots[0])==0 else 'innovation'
            factor=1+(int(self.policy['gop'])-1)*(1-float(self.policy['keep_fraction'])) if role=='anchor' else 1.
            self.curve_records.append({'layer':l,'kind':kind,'role':role,'population_slots_per_head':n,'sampled_slots_per_head':take,'plain_loss':losses,'fanout_factor':factor})
        return super()._innovation(l,state,raw,slots,kind)
