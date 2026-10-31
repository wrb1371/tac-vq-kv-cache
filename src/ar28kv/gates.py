"""Independent physical-byte and native-MRoPE checks for mixed-head layouts."""
import torch
from ar26kv.calibration import prefixes
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

def bytes_expected(codec):
    p=codec.capacity;plan=codec.plan
    if plan['hybrid_dense']:return 278528+4*28*4*128*p
    b=codec.global_maps['positions'].element_size()
    total=278528+3*p*b+256+4*codec.M+4*codec.T+28*(4*p+4*4*128*codec.T)
    if plan.get('headwise'):
        total+=plan['head_metadata_bytes']
        total+=sum(z['slots']*(8*(z['rK']+z['rV'])+4) for ll in plan['layer_head_choices'] for z in ll)
    else:total+=sum(4*z['slots']*(8*(z['rK']+z['rV'])+4) for z in plan['layer_choices'])
    return total

@torch.no_grad()
def head_reader_gate(codec,layer,teacher):
    if not codec.plan.get('headwise') or codec.plan['hybrid_dense']:return []
    c=codec;st=c.states[layer];vi=c.global_maps['video'].long();ti=c.global_maps['text'].long()
    nt,h,w=c.grid;p=h*w;g=c.plan['gop'];ordinal=torch.arange(c.M,device='cuda')
    ao=ordinal[(ordinal//p)%g==0];to=ordinal[(ordinal//p)%g!=0];a=len(ao)
    raw={'K':teacher.states[layer]['K_pre'].cuda().float(),'V':teacher.states[layer]['V'].cuda().float()}
    expected={s:torch.empty((4,c.end,128),dtype=torch.bfloat16,device='cuda') for s in ['K','V']}
    checks=[]
    for side in ['K','V']:
        all_anchors=prefixes(raw[side][:,vi[ao]],c.table,side,layer)
        for head,z in enumerate(c.plan['layer_head_choices'][layer]):
            anchor=all_anchors[z['r'+side]-1][head]
            code=st['T_code'][head,vi[to]];kind=code.long()&127
            ry=(to//w%h+kind//3-1).clamp(0,h-1);rx=(to%w+kind%3-1).clamp(0,w-1)
            slots=((to//p//g)*p+ry*w+rx).long();pred=anchor[slots].clone();mask=(code&128)!=0
            assert int(mask.sum())==z['innovations']
            if int(mask.sum()):
                residual=raw[side][head:head+1,vi[to[mask]]]-pred[mask][None].float()
                table={**c.table,side+'_mean':c.table[side+'_mean'][:,head:head+1],side+'_scale':c.table[side+'_scale'][:,head:head+1]}
                corrected=prefixes(residual,table,side,layer,independent=False,levels=z['r'+side])[-1][0]
                pred[mask]=(pred[mask].float()+corrected.float()).bfloat16()
            expected[side][head,vi[ao]]=anchor;expected[side][head,vi[to]]=pred
        expected[side][:,ti]=raw[side][:,ti].bfloat16()
    rotary=c.model.model.language_model.rotary_emb
    pos=c.global_maps['positions'][:,:c.end].long()[:,None]
    dummy=torch.empty((1,c.end,1),dtype=torch.bfloat16,device='cuda');cos,sin=rotary(dummy,pos)
    _,native_rotated=apply_multimodal_rotary_pos_emb(expected['K'][None],expected['K'][None],cos,sin,[16,24,24])
    expected['K']=native_rotated[0]
    expected['K'][:,ti]=teacher.states[layer]['K_post'].cuda()[:,ti]
    actual=dict(zip(['K','V'],c.decode(layer)))
    for side in ['K','V']:
        err=(actual[side].float()-expected[side].float()).norm()/expected[side].float().norm().clamp_min(1e-12)
        assert float(err)<2e-3,(layer,side,float(err))
        checks.append({'layer':layer,'side':side,'relative_L2':float(err),
                       'BF16_differences':int((actual[side]!=expected[side]).sum()),'native_HF_MRoPE_reference':True})
    return checks
