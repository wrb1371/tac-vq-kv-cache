"""Exact finite prefix-depth optimization in actual stage-byte units."""
import numpy as np

def allocate(curves,capacity,lower=None,mode='dp'):
    costs=np.asarray(curves,dtype=np.float64);n,levels=costs.shape;maximum=levels-1
    assert costs.ndim==2 and np.isfinite(costs).all() and maximum>=1
    low=np.ones(n,dtype=np.int32) if lower is None else np.asarray(lower,dtype=np.int32)
    assert low.shape==(n,) and np.all((low>=1)&(low<=maximum))
    cap=min(int(capacity),n*maximum);assert int(low.sum())<=cap
    if mode=='uniform':
        rates=low.copy()
        while int(rates.sum())<cap:
            choices=[i for i in range(n) if rates[i]<maximum]
            i=min(choices,key=lambda i:(rates[i],i));rates[i]+=1
        return rates.tolist(),float(sum(costs[i,r] for i,r in enumerate(rates)))
    assert mode=='dp'
    dp=np.full(cap+1,np.inf);dp[0]=0;parents=np.full((n,cap+1),-1,np.int16)
    for i in range(n):
        nxt=np.full_like(dp,np.inf)
        for level in range(int(low[i]),min(maximum,cap)+1):
            candidate=dp[:cap+1-level]+costs[i,level];current=nxt[level:];take=candidate<current
            current[take]=candidate[take];parents[i,level:][take]=level
        dp=nxt
    # More bytes are permitted, never required: a nonmonotone curve may prefer
    # an earlier prefix. Deterministic tie break chooses greater spent units.
    end=min(range(int(low.sum()),cap+1),key=lambda j:(dp[j],-j));objective=float(dp[end]);assert np.isfinite(objective)
    rates=[]
    for i in reversed(range(n)):
        level=int(parents[i,end]);assert level>=low[i];rates.append(level);end-=level
    assert end==0;return rates[::-1],objective
