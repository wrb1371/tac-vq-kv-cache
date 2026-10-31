"""Explicit uncompressed diagnostic oracle; never presented as a quantizer."""
import torch
from r15kv.codecs import PackageCodec
from r15kv.budget import estimate_resident_bytes
class IdentityOracle(PackageCodec):
    method='uncompressed-identity-API-oracle';needs_positions=False
    def make_plan(self,meta):
        desc=[(k,(2,meta.capacity,128),torch.bfloat16,'payload') for k in ['K_dense','V_dense']];return {'description':desc,'estimate':estimate_resident_bytes(0,0,desc),'capacity':meta.capacity,'diagnostic_uncompressed':True}
    def append(self,l,state,kpost,v,pre=None):
        n=kpost.shape[-2];start=state['n'];state['K_dense'][:,start:start+n].copy_(kpost[0]);state['V_dense'][:,start:start+n].copy_(v[0]);state['n']+=n;return state
    def decode(self,l,state,positions=None,inv_freq=None,attention_scaling=1.):
        n=state['n'];return state['K_dense'][:,:n][None],state['V_dense'][:,:n][None]
