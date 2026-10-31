"""One ordinary FP8-aware raw weighted GSRQ refit; unchanged GSKM primitive."""
import torch
from r14kv.fitting import gskm,logweights,nearest
@torch.no_grad()
def fit_fp8_rq(x,norms,seeds,stages=16,iterations=40,callback=None):
    w,diagnostics=logweights(norms);residual=x.float().clone();books=[];history=[]
    for stage in range(stages):
        master,hist=gskm(residual,w,[s+stage for s in seeds],iterations)
        deployed=master.to(torch.float8_e4m3fn)
        assert torch.isfinite(deployed.float()).all()
        ix=nearest(residual,deployed.float());chosen=torch.gather(deployed.float(),1,ix[...,None].expand(-1,-1,x.shape[-1]));residual-=chosen
        books.append(deployed.cpu());row={'stage':stage,'training':hist,'residual_MSE':residual.square().sum(-1).mean(-1).cpu().tolist()};history.append(row)
        if callback:callback(row)
    return torch.stack(books,1),{'weighting':diagnostics,'stages':history,'actual_FP8_snap_before_next_residual':True}
