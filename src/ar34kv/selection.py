import math
import torch
import torch.nn.functional as F

def layer_quotas(total,L,M,recent):
    """PyramidKV beta20 ramp, integerized without overspending a global cap.

    Protected text/query slots are first paid uniformly. The published ramp is
    applied to the remaining visual history. Largest remainders avoid the
    original floor(step) implementation's extra unbudgeted slots.
    """
    total=min(total,L*M);assert total>=L*recent
    history=total-L*recent;avg=history/L;lo=math.floor(avg/20)
    hi=min(M-recent,2*avg-lo);lo=2*avg-hi
    raw=[hi+(lo-hi)*l/(L-1) for l in range(L)]
    base=[math.floor(x+1e-10) for x in raw]
    for i in sorted(range(L),key=lambda j:(-(raw[j]-base[j]),j))[:history-sum(base)]:base[i]+=1
    result=[v+recent for v in base]
    assert sum(result)==total and all(recent<=v<=M for v in result)
    return result

@torch.no_grad()
def pyramid_scores(k,q,groups,end):
    H,n,D=k.shape;assert n==end and q.shape[1]==min(64,end)
    kk=k[:,None].expand(H,groups,n,D).reshape(H*groups,n,D)
    # Preserve the author's model-dtype matmul/scaling/softmax cast and pooling.
    z=torch.matmul(q,kk.transpose(1,2))/math.sqrt(D)
    rows=torch.arange(end-q.shape[1],end,device=k.device)
    z.masked_fill_(torch.arange(end,device=k.device)[None,None]>rows[None,:,None],torch.finfo(z.dtype).min)
    p=torch.softmax(z,dim=-1,dtype=torch.float32).to(q.dtype)
    weight=p[:,:,:max(0,end-64)].sum(1).view(H,groups,-1).mean(1)
    return F.avg_pool1d(weight,5,stride=1,padding=2)

@torch.no_grad()
def expected_scores(k,v,mu,cov,groups,n_sink=4,epsilon=.02):
    H,n,D=k.shape;keys=k[:,n_sink:].float();values=v[:,n_sink:].float()
    kk=keys[:,None].expand(H,groups,n-n_sink,D).reshape(H*groups,n-n_sink,D)
    logits=torch.einsum('hnd,hd->hn',kk,mu)/math.sqrt(D)
    logits+=torch.einsum('hnd,hde,hne->hn',kk,cov,kk)/(2*D)
    p=logits.softmax(-1).view(H,groups,n-n_sink).mean(1)
    scores=(p+epsilon)*values.norm(dim=-1)
    scores=F.pad(scores,(n_sink,0),value=scores.max().item()+1)
    assert torch.isfinite(scores).all()
    return scores

@torch.no_grad()
def select(method,k,v,q,stats,vi,keep,groups,end):
    H,n,D=k.shape;M=len(vi)
    if keep==M:return vi[None].expand(H,-1)
    if method=='PyramidKV':
        forced=vi[vi>=max(0,end-64)];candidate=vi[vi<max(0,end-64)]
        score=pyramid_scores(k,q,groups,end)
    else:
        assert method=='ExpectedAttention'
        forced=vi[vi<4];candidate=vi[vi>=4]
        score=expected_scores(k,v,stats['mean'].cuda(),stats['cov'].cuda(),groups)
    assert keep>=len(forced)
    chosen=candidate[score[:,candidate].topk(keep-len(forced),dim=1).indices]
    return torch.cat([chosen,forced[None].expand(H,-1)],1).sort(1).values
