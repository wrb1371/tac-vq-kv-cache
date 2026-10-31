"""Paid C128 residual pool:127 weighted GSKM centroids plus an explicit null code."""
import torch
from r14kv.fitting import gskm,nearest
@torch.no_grad()
def fit_residual_pool(x,weights,seeds,stages=7,iterations=40):
    residual=x.float().clone();books=[];trace=[]
    for stage in range(stages):
        masters=[];counts=[]
        for h in range(2):
            selected=(residual[h].norm(dim=-1)>1e-8)&(weights[h]>0);n=int(selected.sum());c=min(127,n);master=torch.zeros(128,128,device=x.device)
            if c:
                fitted,_=gskm(residual[h:h+1,selected],weights[h:h+1,selected],[seeds[h]+stage],iterations=iterations,clusters=c);master[1:c+1]=fitted[0]
            masters.append(master);counts.append({'nonzero_supervised_samples':n,'learned_centroids':c,'null_or_unused_centroids':128-c})
        deployed=torch.stack(masters).to(torch.float8_e4m3fn);assert torch.isfinite(deployed.float()).all() and not deployed[:,0].float().any();ix=nearest(residual,deployed.float());residual-=torch.gather(deployed.float(),1,ix[...,None].expand(-1,-1,128));books.append(deployed.cpu())
        trace.append({'stage':stage,'head_fit_counts':counts,'weighted_residual_MSE':((residual.square().sum(-1)*weights).sum(-1)/weights.sum(-1).clamp_min(1e-30)).cpu().tolist(),'null_code_fraction':(ix==0).float().mean(-1).cpu().tolist()})
    return torch.stack(books,1),trace
