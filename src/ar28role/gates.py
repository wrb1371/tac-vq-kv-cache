"""Independent bytes and stored-role reference with native HF MRoPE."""
import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
def bytes_expected(c):
    if c.plan['hybrid_dense']:return 278528+4*28*4*128*c.capacity
    b=c.global_maps['positions'].element_size()
    total=278528+3*c.capacity*b+256+4*c.M+4*c.T+28*(4*c.capacity+4*4*128*c.T)+3140
    a=c.plan['anchors']
    for ll in c.plan['layer_head_roles']:
        for z in ll:
            total+=a*(8*(z['rAK']+z['rAV'])+4)+z['innovations']*(8*(z['rIK']+z['rIV'])+4)+c.q*(8*(z['qK']+z['qV'])+4)
    return total

@torch.no_grad()
def stored_gate(c,l):
    if c.plan['hybrid_dense']:return []
    st=c.states[l];vi=c.global_maps['video'].long();ti=c.global_maps['text'].long();nt,gh,gw=c.grid
    p=gh*gw;g=c.plan['gop'];o=torch.arange(c.M,device='cuda');ao=o[(o//p)%g==0];to=o[(o//p)%g!=0]
    result={s:torch.empty((4,c.end,128),dtype=torch.bfloat16,device='cuda')for s in ['K','V']}
    for h,z in enumerate(c.plan['layer_head_roles'][l]):
        code=st['T_code'][h,vi[to]];lo=code.long()&127;mask=(code&128)!=0
        yy=(to//gw%gh+lo//3-1).clamp(0,gh-1);xx=(to%gw+lo%3-1).clamp(0,gw-1)
        slots=(to//p//g)*p+yy*gw+xx;assert int(mask.sum())==z['innovations']
        for s in ['K','V']:
            decoded={}
            for role in ['A','I']:
                ix=st[f'{s}_{role}_index_h{h}'];n=len(ix)
                if not n:decoded[role]=None;continue
                words=torch.stack([c.table[s+'_books'][r].float()[ix[:,r].long()]for r in range(ix.shape[1])])
                x=(words.sum(0).reshape(n,128)@c.table['H'].T)*st[f'{s}_{role}_norm_h{h}'].float()*c.table[s+'_scale'][l,h].float()
                if role=='A':x+=c.table[s+'_mean'][l,h].float()
                decoded[role]=x.bfloat16()
            a=decoded['A'];pred=a[slots].clone()
            if z['innovations']:pred[mask]=(pred[mask].float()+decoded['I'].float()).bfloat16()
            result[s][h,vi[ao]]=a;result[s][h,vi[to]]=pred
    pos=c.global_maps['positions'][:,:c.end].long()[:,None];dummy=torch.empty((1,c.end,1),dtype=torch.bfloat16,device='cuda')
    cos,sin=c.model.model.language_model.rotary_emb(dummy,pos)
    _,rot=apply_multimodal_rotary_pos_emb(result['K'][None],result['K'][None],cos,sin,[16,24,24]);result['K']=rot[0]
    result['K'][:,ti]=st['text_FP_K'];result['V'][:,ti]=st['text_FP_V'];actual=dict(zip(['K','V'],c.decode(l)))
    records=[]
    for s in ['K','V']:
        err=float((actual[s].float()-result[s].float()).norm()/result[s].float().norm().clamp_min(1e-12))
        assert err<.002,(l,s,err)
        records.append({'layer':l,'side':s,'relative_L2':err,'BF16_different_elements':int((actual[s]!=result[s]).sum()),'actual_stored_indices':True})
    return records
