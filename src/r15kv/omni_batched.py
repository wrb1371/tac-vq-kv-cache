"""Identical independent Omni window arithmetic, 128 groups per kernel batch."""
import collections,torch
from .omni_budget import OmniBudget
from .budget import pack_indices
from model.codec.packed_cache import _encode_mxfp4
class OmniBudgetBatched(OmniBudget):
    @torch.no_grad()
    def write_key_groups(self,l,state,k,start):
        n=k.shape[1];norm=k.float().norm(dim=-1,keepdim=True).clamp_min(1e-12);state['K_norm'][:,start:start+n].copy_(norm.half());direction=torch.matmul(k.float()/norm,self.aux['H'].float().T).half()
        if start==0:
            assert n==self.plan['initial_prefix_tokens'];groups=range(self._nbase)
        else:
            assert n==1;groups=[self._nbase+start-self.plan['initial_prefix_tokens']]
        by=collections.defaultdict(list)
        for j in groups:by[len(self._windows[j])].append(j)
        for length,js in by.items():
            for b in range(0,len(js),128):
                batch=js[b:b+128];ids=torch.tensor(batch,device='cuda');beg=self.routes['offset'].index_select(0,ids).long();op=(beg[:,None]+torch.arange(length,device='cuda')[None,:]).flatten();native=self.routes['order'].index_select(0,op).long();z=direction.index_select(1,native-start).reshape(2,len(batch),length,128).float();minimum=z.amin(2);span=(z.amax(2)-minimum).clamp_min(1e-12);mp,me,mr=_encode_mxfp4(minimum);sp,se,sr=_encode_mxfp4(span)
                state['K_min'][:,ids]=mp;state['K_min_exp'][:,ids]=me;state['K_span'][:,ids]=sp;state['K_span_exp'][:,ids]=se
                mr=mr.half().float();sr=sr.half().float();unit=((z-mr[:,:,None])/torch.where(sr==0,torch.ones_like(sr),sr)[:,:,None]).clamp(0,1);q=torch.bucketize(unit.contiguous(),self.aux['local_boundaries'+str(self.plan['K_bits'])],right=True);state['K_index'][:,native]=pack_indices(q,self.plan['K_bits']).reshape(2,-1,128*self.plan['K_bits']//8)
