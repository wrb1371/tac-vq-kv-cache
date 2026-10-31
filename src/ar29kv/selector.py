"""Same CoRDS mixed trim, caching minima and retaining argmax on the GPU."""
import torch
@torch.no_grad()
def async_mixed(K,V,budget,alpha):
    n=len(K);b=min(int(budget),n);device=K.device
    if b<=0 or n==0:return torch.zeros(0,dtype=torch.long,device=device)
    if b>=n:return torch.arange(n,device=device)
    k=K.float();v=V.float();assert torch.isfinite(k).all()and torch.isfinite(v).all()
    a=min(1.,max(0.,float(alpha)));mask=torch.zeros(n,dtype=torch.bool,device=device)
    dk=(k-k.mean(0,keepdim=True)).pow(2).sum(-1);dv=(v-v.mean(0,keepdim=True)).pow(2).sum(-1)
    selected=torch.empty(b,device=device,dtype=torch.long)
    for step in range(b):
        if step:
            nk=(k-torch.index_select(k,0,j)).pow(2).sum(-1)
            nv=(v-torch.index_select(v,0,j)).pow(2).sum(-1)
            dk=nk if step==1 else torch.minimum(dk,nk)
            dv=nv if step==1 else torch.minimum(dv,nv)
        dm=(a*dk+(1.-a)*dv).masked_fill(mask,-1.)
        j=torch.argmax(dm).reshape(1)
        selected[step:step+1]=j;mask.scatter_(0,j,True)
    # Finite squared distances leave at least one nonnegative unselected
    # candidate at every step; original early-exit/safety-fill cannot activate.
    return selected.sort().values

from ar26kv.codec import Codec
from ar19kv.video_prune import reference_module
import types
class BudgetSelector(Codec):
    def select(self,k,v):
        assert self.name=='CoRDS'
        nt,h,w=self.grid;p=h*w;keep=self.plan['media_kept_per_head']
        if keep==self.M:return torch.arange(self.M,device='cuda')[None].expand(4,-1)
        recent=min(int(nt*.25),(keep-round(keep*.5))//p)
        qr=(recent+1e-8)/nt if recent else 0.;tar=.5 if recent else 0.
        dummy=types.SimpleNamespace(config=types.SimpleNamespace(height_width=h),model=types.SimpleNamespace(language_model=types.SimpleNamespace(layers=[None])))
        official=reference_module('cords');official.select_bicriteria_farthest_mixed=async_mixed
        ix=official._cords_select_indices(k[None],v[None],4,p,nt,keep/p,tar,qr,-1,1,dummy,8,'cords',.5,1.,'maxcos',.25,0.,1.,'none',False,1)
        assert ix.shape==(4,keep);return ix
