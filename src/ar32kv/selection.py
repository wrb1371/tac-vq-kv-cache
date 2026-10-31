"""Question-independent SnapKV and AnchorKV (Approx) physical-head selectors.

Published criteria; video adaptation protects all nonvisual prefix tokens. The
AnchorKV implementation follows paper Algorithm1, not unreleased author code.
"""
import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def causal_importance(q,k,rows,groups,end,mode):
    """Bound scratch by query tiles; normalize each original causal query row."""
    h,n,d=k.shape;assert n==end and q.shape[0]==h*groups
    total=torch.zeros((h,n),device=k.device,dtype=torch.float32)
    keys=k[:,None].expand(h,groups,n,d).reshape(h*groups,n,d)
    if mode=='SnapKV':take=rows>=max(0,end-64)
    else:take=torch.zeros(len(rows),device=rows.device,dtype=torch.bool);take[:]=True
    q=q[:,take];rows=rows[take]
    cols=torch.arange(n,device=k.device)
    for a in range(0,len(rows),32):
        jj=rows[a:a+32];scores=torch.matmul(q[:,a:a+32],keys.transpose(1,2)).float()/math.sqrt(d)
        scores.masked_fill_(cols[None,None,:]>jj[None,:,None],float('-inf'))
        p=torch.softmax(scores,-1)
        # Official SnapKV casts softmax to model dtype before window sum.
        if mode=='SnapKV':p=p.to(q.dtype).float()
        total+=p.view(h,groups,len(jj),n).mean(1).sum(1)
    if mode=='AnchorKV':
        observed=(rows[:,None]>=cols[None]).sum(0).clamp_min(1)
        total/=observed[None]
    return total

def quotas(lengths,total):
    n=sum(lengths);base=[total*x//n for x in lengths]
    for j in sorted(range(len(lengths)),key=lambda i:(-(total*lengths[i]%n),i))[:total-sum(base)]:base[j]+=1
    assert sum(base)==total and all(0<=b<=n for b,n in zip(base,lengths))
    return base

@torch.no_grad()
def select(method,k,q,rows,vi,keep,groups,end,anchor_rows=None):
    h,n,d=k.shape;device=k.device;vi=vi.long();M=len(vi)
    if keep==M:return vi[None].expand(h,-1)
    if method=='SnapKV':
        weights=causal_importance(q,k,rows,groups,end,method)
        old=max(0,end-64);scores=F.avg_pool1d(weights[:,:old],5,stride=1,padding=2)
        forced=vi[vi>=old];candidates=vi[vi<old]
        assert keep>=len(forced)
        chosen=candidates[scores[:,candidates].topk(keep-len(forced),dim=1).indices]
        return torch.cat([chosen,forced[None].expand(h,-1)],1).sort(1).values
    assert method=='AnchorKV'and anchor_rows is not None
    # q contains union of Approx rows and SnapKV window; Approx uses its own
    # exact declared rows rather than silently paying for a different estimator.
    mask=torch.isin(rows,anchor_rows);weight=causal_importance(q[:,mask],k,rows[mask],groups,end,method)
    forced=vi[(vi<10)|(vi>=max(10,end-64))];middle=vi[(vi>=10)&(vi<max(10,end-64))]
    assert keep>=len(forced);blocks=[middle[(middle>=b)&(middle<min(b+4096,end-64))]for b in range(10,max(10,end-64),4096)]
    blocks=[b for b in blocks if len(b)];qq=quotas([len(b)for b in blocks],keep-len(forced))
    # rho is the complete retained-prefix ratio, excluding prepaid query slots.
    rho=(keep+end-M)/end;alpha=1-rho;pieces=[forced[None].expand(h,-1)]
    for b,count in zip(blocks,qq):
        if not count:continue
        z=F.normalize(k[:,b].float(),dim=-1);mu=F.normalize(z.mean(1,keepdim=True),dim=-1)
        diversity=(1-(z*mu).sum(-1)).clamp_min(0)
        score=diversity*weight[:,b].clamp_min(0).pow(alpha)
        pieces.append(b[score.topk(count,dim=1).indices])
    selected=torch.cat(pieces,1).sort(1).values
    assert selected.shape==(h,keep)
    return selected
