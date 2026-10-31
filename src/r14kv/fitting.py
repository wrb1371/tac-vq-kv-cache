"""Full NOVA transform/group fitting and batched weighted raw GSKM/RQ."""
import math,torch
import torch.nn.functional as nnf

@torch.no_grad()
def nearest(x,c,chunk=8192):
    # B,N,D and B,C,D. No sample weights in fixed-codebook assignments.
    result=[];cn=c.square().sum(-1)
    for start in range(0,x.shape[1],chunk):
        z=x[:,start:start+chunk]
        d=z.square().sum(-1,keepdim=True)+cn[:,None,:]-2*torch.bmm(z,c.transpose(1,2))
        result.append(d.argmin(-1))
    return torch.cat(result,1)

@torch.no_grad()
def _update(x,unit,w,assignment,shapes,gains,eps=1e-8):
    b,_,d=x.shape;c=shapes.shape[1]
    ss=torch.zeros_like(shapes);rs=torch.zeros_like(shapes);ws=torch.zeros(b,c,device=x.device)
    ix=assignment[...,None].expand(-1,-1,d)
    ss.scatter_add_(1,ix,unit*w[...,None]);rs.scatter_add_(1,ix,x*w[...,None]);ws.scatter_add_(1,assignment,w)
    norm=ss.norm(dim=-1);valid=(ws>eps)&(norm>eps)
    newshape=torch.where(valid[...,None],ss/norm.clamp_min(eps)[...,None],shapes)
    newgain=torch.where(valid,((rs*newshape).sum(-1)/ws.clamp_min(eps)).clamp_min(0),gains)
    return newshape,newgain,(~valid).sum(-1)

@torch.no_grad()
def gskm(x,w,seeds,iterations=40,clusters=256,chunk=8192):
    assert x.ndim==3 and w.shape==x.shape[:2] and x.shape[1]>=clusters
    unit=nnf.normalize(x,dim=-1,eps=1e-8);b,n,d=x.shape
    shapes=[]
    for j,seed in enumerate(seeds):
        gen=torch.Generator(device=x.device).manual_seed(int(seed));ix=torch.randperm(n,device=x.device,generator=gen)[:clusters]
        s=unit[j,ix].clone();zero=s.norm(dim=-1)<=1e-8
        if zero.any():
            replacement=torch.zeros(int(zero.sum()),d,device=x.device);ii=torch.arange(len(replacement),device=x.device);replacement[ii,ii%d]=1;s[zero]=replacement
        shapes.append(s)
    shapes=torch.stack(shapes);gains=torch.ones(b,clusters,device=x.device)
    def assign():
        ans=[]
        for start in range(0,n,chunk):
            score=2*torch.bmm(x[:,start:start+chunk],shapes.transpose(1,2))*gains[:,None,:]-gains[:,None,:].square()
            ans.append(score.argmax(-1))
        return torch.cat(ans,1)
    shapes,gains,_=_update(x,unit,w,assign(),shapes,gains)
    history=[]
    for iteration in range(iterations):
        ix=assign();shapes,gains,empty=_update(x,unit,w,ix,shapes,gains)
        if iteration in [0,iterations-1]:
            code=shapes*gains[...,None];rec=torch.gather(code,1,ix[...,None].expand(-1,-1,d))
            mse=((x-rec).square().sum(-1)*w).sum(-1)/w.sum(-1)
            history.append({'iteration':iteration+1,'weighted_MSE':mse.cpu().tolist(),'empty_or_degenerate':empty.cpu().tolist()})
    return shapes*gains[...,None],history

@torch.no_grad()
def logweights(norms):
    assert torch.isfinite(norms).all() and (norms>=0).all()
    median=norms.median(dim=-1).values;w=torch.log1p(norms/(median[:,None]+1e-12))
    assert torch.isfinite(w).all() and (w.sum(-1)>0).all()
    return w,{'median':median.cpu().tolist(),'zero_fraction':(norms==0).float().mean(-1).cpu().tolist(),'mean_weight':w.mean(-1).cpu().tolist(),'formula':'log1p(u/(per-layer-physical-head-kind median+1e-12)); fixed all stages'}

@torch.no_grad()
def fit_rq(x,norms,seeds,stages=32,iterations=40,callback=None):
    w,diagnostics=logweights(norms);residual=x.float().clone();books=[];history=[]
    for stage in range(stages):
        master,hist=gskm(residual,w,[s+stage for s in seeds],iterations)
        # Train next residual against actual deployed FP16 centroids.
        deployed=master.half();ix=nearest(residual,deployed.float())
        chosen=torch.gather(deployed.float(),1,ix[...,None].expand(-1,-1,x.shape[-1]));residual-=chosen
        books.append(deployed.cpu())
        row={'stage':stage,'iterations':iterations,'training':hist,'residual_MSE':residual.square().sum(-1).mean(-1).cpu().tolist()}
        history.append(row)
        if callback:callback(row)
    return torch.stack(books,1),{'weighting':diagnostics,'stages':history,'deployed_FP16_snap_before_next_residual':True}

@torch.no_grad()
def qpca(q2,k2,ksum,count):
    sq=q2.double()/count;mu=ksum.double()/count;sk=k2.double()/count-mu[:,None]*mu[None,:]
    sq=(sq+sq.T)/2;sk=(sk+sk.T)/2
    ev,U=torch.linalg.eigh(sq);clipped=ev.clamp_min(1e-30)
    root=(U*clipped.sqrt()[None,:])@U.T;iroot=(U*clipped.rsqrt()[None,:])@U.T
    mat=root@sk@root;lam,E=torch.linalg.eigh((mat+mat.T)/2);order=lam.argsort(descending=True);E=E[:,order]
    perm=torch.tensor([g+j*32 for g in range(32) for j in range(4)],device=E.device)
    forward=(root@E)[:,perm];inverse=(E.T@iroot)[perm]
    return forward,inverse.T,mu,{'minimum_raw_Q_eigenvalue':float(ev.min()),'maximum_Q_eigenvalue':float(ev.max()),'floor':1e-30,'additive_ridge':0.,
        'FP64_identity_max_abs':float((forward@inverse-torch.eye(128,device=forward.device)).abs().max()),
        'BF16_identity_max_abs':float((forward.bfloat16().double()@inverse.bfloat16().double()-torch.eye(128,device=forward.device)).abs().max()),'grouping':'descending eigenvalue ranks g,g+32,g+64,g+96'}

@torch.no_grad()
def lloyd_group(x,seed=0,iterations=25,clusters=256):
    # x: independent groups,N,4; canonical same random-distinct rows per group.
    b,n,d=x.shape;gen=torch.Generator(device=x.device).manual_seed(seed);ix=torch.randperm(n,device=x.device,generator=gen)[:clusters]
    cent=x[:,ix].clone()
    for _ in range(iterations):
        assignment=nearest(x,cent);sums=torch.zeros_like(cent);counts=torch.zeros(b,clusters,device=x.device)
        sums.scatter_add_(1,assignment[...,None].expand(-1,-1,d),x)
        counts.scatter_add_(1,assignment,torch.ones_like(assignment,dtype=torch.float32))
        updated=sums/counts.clamp_min(1)[...,None]
        empty=counts==0
        if empty.any():
            selected=torch.gather(cent,1,assignment[...,None].expand(-1,-1,d));err=(x-selected).square().sum(-1)
            for g in range(b):
                m=int(empty[g].sum())
                if m:updated[g,empty[g]]=x[g,err[g].argsort(descending=True)[:m]]
        cent=updated
    return cent

def hadamard(d):
    h=torch.ones(1,1,dtype=torch.float64)
    while len(h)<d:h=torch.cat([torch.cat([h,h],1),torch.cat([h,-h],1)],0)/math.sqrt(2)
    return h

@torch.no_grad()
def value_basis(o2):
    ev,U=torch.linalg.eigh((o2.double()+o2.double().T)/2);d=len(ev)
    order=ev.argsort(descending=True);br=torch.tensor([int(bin(i)[2:].zfill(7)[::-1],2) for i in range(d)])
    perm=torch.empty(d,dtype=torch.long);perm[br]=order.cpu()
    P=torch.eye(d,dtype=torch.float64)[:,perm].to(U.device)
    return U@hadamard(d).to(U.device)@P

@torch.no_grad()
def fit_nova_head(post_k,q2,k2,ksum,o2,count,seed,limit=131072):
    f,a,mu,diag=qpca(q2,k2,ksum,count)
    gen=torch.Generator(device=post_k.device).manual_seed(seed+140000)
    ix=torch.randperm(len(post_k),device=post_k.device,generator=gen)[:min(limit,len(post_k))]
    r=(post_k[ix].double()-mu)@f;r=r/r.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
    groups=r.reshape(-1,32,4).permute(1,0,2).float().contiguous()
    cent=lloyd_group(groups,seed=seed,iterations=25)
    # Explicitly preserve repository default trainer + runtime snap sequence.
    scale=cent.abs().amax((-2,-1),keepdim=True).clamp_min(1e-12)/57344.
    trainer=((cent/scale).to(torch.float8_e5m2).float()*scale).half()
    deployed=trainer.to(torch.float8_e5m2)
    diag.update(actual_fit_samples=len(ix),iterations=25,centroids_per_group=256,groups=32,
        deployment_snap='author default scaled-e5m2 roundtrip then loader direct-e5m2 resnap; assignment uses deployed bytes',
        runtime_resnap_relative_RMSE=float(((deployed.float()-trainer.float()).square().sum()/trainer.float().square().sum().clamp_min(1e-30)).sqrt()))
    bundle={'F':f.bfloat16().cpu(),'A':a.bfloat16().cpu(),'mean':mu.bfloat16().cpu(),'V_rotation':value_basis(o2).bfloat16().cpu(),'centroids':deployed.cpu()}
    masters={'F':f.cpu(),'A':a.cpu(),'mean':mu.cpu()}
    return bundle,masters,diag
