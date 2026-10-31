"""Independent physical formulas and actual-index/native-position reference."""
import torch
from ar28kv.gates import bytes_expected as sti_bytes
from ar28role.gates import bytes_expected as role_bytes
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
def bytes_expected(c):
    if c.name.startswith('GSRQ'):
        static=28*4*2*c.plan['loaded_stages']*256*128
        if c.plan['hybrid_dense']:return static+28*4*4*128*c.capacity
        b=c.global_maps['positions'].element_size()
        return static+3*c.capacity*b+256+4*c.M+4*c.T+28*4*4*128*c.T+28*4*c.capacity*(c.plan['rK']+c.plan['rV'])
    if c.name in ['InfiniPot-V','CoRDS']:return 4*(c.M+c.T)+28*4*4*128*c.plan['physical_slots']
    return role_bytes(c)if c.plan.get('role_separated')else sti_bytes(c)

@torch.no_grad()
def gs_reader_gate(c,l):
    if c.plan['hybrid_dense']:return []
    st=c.states[l];expected={}
    for s in ['K','V']:
        ix=st[s+'_index'][:,:c.end];parts=[]
        for r in range(ix.shape[-1]):parts.append(torch.gather(c.books[l][s][:,r].float(),1,ix[:,:,r].long()[...,None].expand(-1,-1,128)))
        expected[s]=torch.stack(parts).sum(0).bfloat16()
    pos=c.global_maps['positions'][:,:c.end].long()[:,None];dummy=torch.empty((1,c.end,1),dtype=torch.bfloat16,device='cuda')
    cos,sin=c.model.model.language_model.rotary_emb(dummy,pos)
    _,rot=apply_multimodal_rotary_pos_emb(expected['K'][None],expected['K'][None],cos,sin,[16,24,24]);expected['K']=rot[0]
    ti=c.global_maps['text'].long();expected['K'][:,ti]=st['text_FP_K'];expected['V'][:,ti]=st['text_FP_V']
    actual=dict(zip(['K','V'],c.decode(l)));records=[]
    for s in ['K','V']:
        error=float((actual[s].float()-expected[s].float()).norm()/expected[s].float().norm().clamp_min(1e-12))
        assert error<.002,(l,s,error)
        records.append({'layer':l,'side':s,'actual_stored_indices':True,'native_HF_MRoPE_reference':True,'relative_L2':error})
    return records
