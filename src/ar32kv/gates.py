"""Independent small-array scoring and real packed-reader checks."""
import importlib.util,math,pathlib
import torch
import torch.nn.functional as F
from .selection import select,quotas

@torch.no_grad()
def selection_gate(root):
    torch.manual_seed(72);n,h,g,d=241,2,3,16;k=torch.randn(h,n,d,device='cuda',dtype=torch.bfloat16);q=torch.randn(h*g,n,d,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k);rows=torch.arange(n,device='cuda');vi=torch.arange(14,n,device='cuda');keep=121
    # Reference explicitly constructs full attention once, including GQA heads.
    kr=k.repeat_interleave(g,0);p=torch.softmax((q@kr.transpose(1,2)).float()/math.sqrt(d)+torch.full((n,n),float('-inf'),device='cuda').triu(1),-1)
    snap=p[:,-64:].bfloat16().float().view(h,g,64,n).mean(1).sum(1);pooled=F.avg_pool1d(snap[:,:n-64],5,1,2);early=vi[vi<n-64];late=vi[vi>=n-64]
    expected=torch.cat([early[pooled[:,early].topk(keep-len(late),dim=1).indices],late[None].expand(h,-1)],1).sort(1).values
    assert torch.equal(expected,select('SnapKV',k,q,rows,vi,keep,g,n))
    # Official SnapKV equivalence on one physical query head per key head.
    path=root/'third_party/ar32_candidates/SnapKV/snapkv/monkeypatch/snapkv_utils.py';spec=importlib.util.spec_from_file_location('official_snapkv',path);official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    single=q.view(h,g,n,d)[:,0];ref=official.SnapKVCluster(window_size=64,max_capacity_prompt=keep,kernel_size=5,pooling='avgpool');rk,rv=ref.update_kv(k[None],single[None],v[None],None,1)
    fullvi=torch.arange(n,device='cuda');ix=select('SnapKV',k,single,rows,fullvi,keep,1,n)
    # Official order is topk then recent, ours is sorted for native source order.
    for j in range(h):
        assert torch.equal(rk[0,j].float().sum(-1).sort().values,k[j,ix[j]].float().sum(-1).sort().values)
    ar=torch.linspace(0,n-1,int(.1*n),device='cuda').long();wm=(p[:,ar].view(h,g,len(ar),n).mean(1).sum(1)/((ar[:,None]>=rows[None]).sum(0).clamp_min(1)))
    mid=vi[(vi>=10)&(vi<n-64)];forced=vi[(vi<10)|(vi>=n-64)];z=F.normalize(k[:,mid].float(),dim=-1);center=F.normalize(z.mean(1,keepdim=True),dim=-1);div=(1-(z*center).sum(-1)).clamp_min(0);rho=(keep+n-len(vi))/n;score=div*wm[:,mid].pow(1-rho)
    ai=torch.cat([mid[score.topk(keep-len(forced),dim=1).indices],forced[None].expand(h,-1)],1).sort(1).values
    assert torch.equal(ai,select('AnchorKV',k,q,rows,vi,keep,g,n,ar))
    assert quotas([4090,4096,52],137)==[68,68,1]
    return {'independent_causal_GQA_and_topk_exact':True,'official_SnapKV_default_criteria_exact':True,'AnchorKV_Algorithm1_small_array_exact':True,'largest_remainder_quota_exact':True}

@torch.no_grad()
def packed_gate(codec,teacher):
    for l in range(codec.L):
        ix=codec.selected[l].cuda();a,b=codec.decode(l)
        for name,t in [('K_post',a),('V',b)]:
            expected=teacher.states[l][name].cuda().gather(1,ix[...,None].expand(-1,-1,codec.D));assert torch.equal(t,expected)
        assert codec.states[l]['K_BF16'].shape[1]-a.shape[1]>=codec.q
    assert codec.registry.receipt()['resident_bytes']==codec.plan['independent_byte_formula']<=codec.budget
    return {'all_layers_KV_exact_gather':True,'all_query_reserved_capacity_paid':True,'independent_bytes_exact':True}
