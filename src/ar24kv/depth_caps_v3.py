"""Same-paid-arena precision interventions, for causal illustrations only.

The full original arenas are still registered and paid. Prefix stage views
control effective encoding/reading depth without claiming any byte saving.
Both anchor and innovation residuals are re-encoded at the intervened depth.
Raw-feature references and innovation ranking remain depth independent.
"""
import json
from ar22kv.joint import JointTemporal
from r14kv.common import ROOT,sha
from .attribution import CheckedDecode


class DepthCapControl(CheckedDecode,JointTemporal):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        path=ROOT/self.policy['depth_cap_file']
        self._caps=json.loads(path.read_text())['depths']
        self.metadata.update(diagnostic_depth_cap=True,depth_cap_file_SHA256=sha(path),
            capacity_rule='entire original full-depth arenas remain charged; effective prefix depths only; no byte saving claimed')

    def prepare_request(self,meta,registry):
        self._cap_item_key=self.context['item_id'].rsplit('-',1)[0]
        return super().prepare_request(meta,registry)

    def _cap(self,layer,state):
        if self._hybrid_dense:return state
        band=self.policy['cap_band']
        low,high={'early':(0,12),'middle':(12,24),'late':(24,36),'all':(0,36)}[band]
        if not low<=layer<high:return state
        key=self._cap_item_key
        desired=self._caps[key]
        result=dict(state)
        for side in ['K','V']:
            if self.policy['cap_side'] not in [side,'both']:continue
            name=side+'_index';depth=min(state[name].shape[2],desired[side][layer])
            result[name]=state[name][:,:,:depth,:]
        return result

    def append(self,layer,state,kpost,value,pre=None):
        return super().append(layer,self._cap(layer,state),kpost,value,pre)

    def decode(self,layer,state,positions,inv_freq,attention_scaling=1.):
        return super().decode(layer,self._cap(layer,state),positions,inv_freq,attention_scaling)
