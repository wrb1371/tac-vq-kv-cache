"""Independent frozen two-head Triton reader oracle, pairwise physical-head audit."""
import types
import torch
from ar21kv.dynamic_kernels import read_slots,compose

def reader_gate(codec,l):
    from ar26kv.codec import H,D
    assert codec.name=='STI-RVQ' and not codec.plan['hybrid_dense']
    st=codec.states[l];t=codec.table;plan=codec.plan;g=plan['gop'];nt,h,w=codec.grid;a=plan['anchors'];nr=plan['layer_choices'][l]['innovations'];vi=codec.global_maps['video'];ti=codec.global_maps['text'];pos=codec.global_maps['positions'][:,:codec.end];freq=codec.global_maps['freq'];ours=codec.decode(l);checks=[]
    for begin in [0,2]:
        s={k:v[begin:begin+2] for k,v in st.items() if torch.is_tensor(v)};s['n']=codec.end
        tables={k:(v[:,begin:begin+2] if k.endswith('_mean') or k.endswith('_scale') else v) for k,v in t.items()}
        arrays=[]
        for start,n,indep in [(0,a,True),(a,nr,False),(a+nr,0,True)]:
            slots=torch.arange(start,start+n,device='cuda')[None].expand(2,-1).contiguous()
            for side in ['K','V']:
                if n:decoded=read_slots(l,s,side,slots,indep,tables)
                else:decoded=torch.empty((2,0,D),device='cuda',dtype=torch.bfloat16)
                arrays.append(decoded)
        dummy=types.SimpleNamespace(_video_list=list(range(len(vi))),_text_list=list(range(len(ti))),_vid_gpu=vi,_text_gpu=ti,_end=codec.end,_h=h,_w=w,policy={'gop':g},plan={'anchor_tokens':a,'innovation_slots':nr})
        ref=compose(dummy,s,pos,freq,codec.scale,arrays)
        for side,own,expected in zip(['K','V'],ours,ref):
            x=own[begin:begin+2].float();y=expected[0].float();err=float((x-y).abs().max());rmse=float((x-y).square().sum().sqrt()/y.square().sum().sqrt().clamp_min(1e-20))
            assert torch.isfinite(y).all() and rmse<1e-3,(side,rmse,err)
            checks.append({'layer':l,'head_pair':begin,'side':side,'max_abs_error':err,'relative_L2_error':rmse,'frozen_two_head_Triton_oracle':True})
    return checks
