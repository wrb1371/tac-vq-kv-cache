"""One fixed counterfactual: same G1 package and parent V, uniform feasible K."""
import json,pathlib,torch
from .codecs import GSRQBudget
from .budget import gs_split,gsrq_description,estimate_resident_bytes
class GSRQUniformVMatched(GSRQBudget):
    method='GSRQ-R15-G1-same-package-uniform-V-matched-control'
    def __init__(self,package,budget_mib,**policy):
        super().__init__(package,budget_mib,**policy);self.parent=json.loads((pathlib.Path(policy['parent_package'])/'completion.json').read_text())
    def make_plan(self,meta):
        dtype=torch.int16 if max(meta.position_max)+max(0,meta.generation_cap-1)<=32767 else torch.int32;posbytes=3*meta.capacity*torch.empty((),dtype=dtype).element_size()+256
        total=min(self.parent['Rmax_K']+self.parent['Rmax_V'],(self.budget_bytes-self.parent['static_tensor_bytes']-posbytes)//(72*meta.capacity));parent_k,rv=gs_split(total,self.parent['Rmax_K'],self.parent['Rmax_V'],0)
        rk=min(self.package['Rmax_K'],(self.budget_bytes-self.static_bytes-posbytes)//(72*meta.capacity)-rv)
        if rk<1:raise RuntimeError('NOT_FEASIBLE_BY_INTEGER_ORACLE uniform samepackage Vmatched')
        desc=gsrq_description(meta.capacity,rk,rv);est=estimate_resident_bytes(self.static_bytes,posbytes,desc)
        return {'R_K':rk,'R_V':rv,'parent_R_K':parent_k,'parent_R_V':rv,'loaded_Rmax_K':self.package['Rmax_K'],'loaded_Rmax_V':self.package['Rmax_V'],'description':desc,'estimate':est,'position_dtype':str(dtype),'capacity':meta.capacity,'causal4096_supported':True,'control_identity':'exact same loaded G1 K14V12 package and directparent V count; strongest feasible uniformK; no fake routepadding or extraQ; one predetermined negativecontrol, not a K/V grid'}
