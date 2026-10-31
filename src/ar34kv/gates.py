"""Independent published-code comparisons and real sufficient-statistic checks."""
import ast,math,types
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .selection import pyramid_scores,expected_scores,layer_quotas

def official_class(path,name,extra=None):
    tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==name]
    @dataclass
    class ScorerPress:compression_ratio:float=0.
    env={'torch':torch,'nn':torch.nn,'F':F,'math':math,'dataclass':dataclass,'ScorerPress':ScorerPress,'repeat_kv':lambda x,g:x.repeat_interleave(g,1)}
    if extra:env.update(extra)
    exec(compile(tree,str(path),'exec'),env);return env[name]

@torch.no_grad()
def statistics_gate(z,mu,cov,R,rotated):
    # Independent centered einsum on all rows, then explicit linear transform.
    m=z.sum(1)/z.shape[1];c=torch.einsum('hni,hnj->hij',z-m[:,None],z-m[:,None])/z.shape[1]
    assert torch.allclose(m,mu,atol=2e-5,rtol=2e-5)
    assert torch.allclose(c,cov,atol=2e-4,rtol=2e-4)
    assert torch.allclose(m@R.T,rotated['mean'].cuda(),atol=2e-5,rtol=2e-5)
    assert torch.allclose(R@c@R.T,rotated['cov'].cuda(),atol=2e-4,rtol=2e-4)
    return {'all_prefix_prerope_mean_covariance_and_rotation_checked':True,'rows':z.shape[1]}

@torch.no_grad()
def selection_gate(root):
    torch.manual_seed(340);H,g,n,D=2,3,241,16
    k=torch.randn(H,n,D,device='cuda');v=torch.randn_like(k);q=torch.randn(H*g,n,D,device='cuda')
    pyramid=official_class(root/'literature/ar34/pyramidkv_utils_reference.py','PyramidKVCluster')
    # Integer-exact published ramps (4 layers, protected64, avgold60,beta20).
    qq=layer_quotas(4*124,4,n,64);assert qq==[181,143,105,67]
    for l in range(4):
        single=q.view(H,g,n,D)[:,0];oracle=pyramid(num_hidden_layers=4,window_size=64,max_capacity_prompt=124,kernel_size=5,beta=20,layer_idx=l)
        rk,rv=oracle.update_kv(k[None],single[None],v[None],None,1)
        scores=pyramid_scores(k,single[:,-64:],1,n);ix=torch.cat([scores.topk(qq[l]-64,dim=1).indices,torch.arange(n-64,n,device='cuda')[None].expand(H,-1)],1)
        assert torch.equal(rk[0],k.gather(1,ix[...,None].expand(-1,-1,D)))
    ea=official_class(root/'third_party/ar34_candidates/kvpress/kvpress/presses/expected_attention_press.py','ExpectedAttentionPress')(epsilon=.02)
    z=q[:,4:];mu=z.mean(1);center=z-mu[:,None];cov=center.transpose(1,2)@center/z.shape[1]
    ea.get_query_statistics=lambda module,hidden:(mu[None],cov[None])
    module=types.SimpleNamespace(config=types.SimpleNamespace(num_attention_heads=H*g))
    ref=ea.score(module,None,k[None],v[None],None,None)[0]
    own=expected_scores(k,v,mu,cov,g)
    assert torch.allclose(own,ref,atol=1e-7,rtol=1e-5)
    assert torch.equal(own.topk(99,dim=1).indices,ref.topk(99,dim=1).indices)
    # Independent direct GQA probability reduction before pooling.
    scores=q[:,-64:]@k.repeat_interleave(g,0).transpose(1,2)/math.sqrt(D)
    scores.masked_fill_(torch.arange(n,device='cuda')[None,None]>torch.arange(n-64,n,device='cuda')[None,:,None],torch.finfo(scores.dtype).min)
    refp=F.avg_pool1d(scores.softmax(-1)[:,:,:n-64].sum(1).view(H,g,-1).mean(1),5,1,2)
    assert torch.allclose(refp,pyramid_scores(k,q[:,-64:],g,n),atol=1e-6,rtol=1e-5)
    for total in [4*64,497,899,4*n]:
        alloc=layer_quotas(total,4,n,64);assert sum(alloc)==total and all(64<=x<=n for x in alloc)
    return {'official_PyramidKV_integer_exact_ramps_and_selected_KV':True,'official_ExpectedAttention_full_covariance_vnorm_GQA_scores_topk':True,'independent_GQA_pooled_attention':True,'global_budget_remainder_and_saturation':True}
