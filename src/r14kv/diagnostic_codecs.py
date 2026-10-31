"""K/V localization only. Full parent is kept, one side replaced with exact FP.

These deliberately retain the parent payload plus the FP intervention. Their
measured storage is charged, and is not an operating point or research child.
"""
import torch
from .codecs import NOVA,GSRQ

class _SideOverride:
    def setup(self,fp_side):
        assert fp_side in ['K','V'];self.fp_side=fp_side
        self.metadata={**self.metadata,'diagnostic_only':True,'FP_side_intervention':fp_side,'storage_policy':'full parent payload/static retained plus intervention FP side; no memory efficiency claim'}
    @torch.no_grad()
    def append(self,l,state,kpost,v,pre=None):
        old=state.get('FP_intervention') if state is not None else None
        result=super().append(l,state,kpost,v,pre)
        x=kpost if self.fp_side=='K' else v
        result['FP_intervention']=x.clone() if old is None else torch.cat([old,x],-2)
        return result
    @torch.no_grad()
    def decode(self,l,state,positions,inv_freq,attention_scaling=1.):
        k,v=super().decode(l,state,positions,inv_freq,attention_scaling)
        if self.fp_side=='K':k=state['FP_intervention']
        else:v=state['FP_intervention']
        return k,v

class NOVASide(_SideOverride,NOVA):
    def __init__(self,directory,fp_side,device='cuda'):
        NOVA.__init__(self,directory,device);self.setup(fp_side)

class GSRQSide(_SideOverride,GSRQ):
    def __init__(self,directory,fp_side,device='cuda',stages=32):
        GSRQ.__init__(self,directory,stages,device);self.setup(fp_side)
