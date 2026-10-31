"""Same CoRDS trim recurrence, incremental nearest-distance minima."""
import torch
from .video_prune import VideoPrune

def incremental_mixed(K,V,budget,alpha):
    n=len(K);b=min(int(budget),n);device=K.device
    if b<=0:return torch.empty(0,dtype=torch.long,device=device)
    if b>=n:return torch.arange(n,device=device)
    a=min(1.,max(0.,float(alpha)));k=K.float();v=V.float()
    dk=(k-k.mean(0,keepdim=True)).square().sum(-1);dv=(v-v.mean(0,keepdim=True)).square().sum(-1)
    mask=torch.zeros(n,dtype=torch.bool,device=device);chosen=[]
    for step in range(b):
        score=(a*dk+(1-a)*dv).masked_fill(mask,-1);j=int(score.argmax());chosen.append(j);mask[j]=True
        nk=(k-k[j]).square().sum(-1);nv=(v-v[j]).square().sum(-1)
        # The empty-set distance is only the first seed; it must NOT remain
        # in the min recurrence after the first selected representative.
        if step==0:dk,dv=nk,nv
        else:dk=torch.minimum(dk,nk);dv=torch.minimum(dv,nv)
    return torch.tensor(sorted(chosen),dtype=torch.long,device=device)

class FastVideoPrune(VideoPrune):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.selector=='cords':
            self.official.select_bicriteria_farthest_mixed=incremental_mixed
            self.metadata['implementation_optimization']='only expanded-frame trim: exact incremental min-distance recurrence; same seed/score/argmax/order, verified against official on CPU and CUDA'
