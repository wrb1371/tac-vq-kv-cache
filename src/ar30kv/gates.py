"""Closed-form bytes, actual-index reconstruction, and old-pair encoder checks."""
import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
from r15kv.codecs import NOVABudget
from r16kv.nsn64 import NSN64Reference
from ar30kv.planning import L,H,D

def bytes_expected(c):
    P=c.capacity;t=c.T;s=c.plan['static_bytes']
    if c.name.startswith('NOVA'):
        a=min(c.plan['sink'],P);r=min(c.plan['recent'],P-a);n=P-a-r
        ib=(D//c.plan['group_size']*c.plan['K_index_bits']+7)//8
        unit=ib+2+D*c.plan['v_bits']//8+2+(0 if c.plan['v_mode']=='sign1'else 2)
        return s+4*t+4*H*D*r+L*(4*H*D*(a+r)+H*n*unit+4*H*D*t)
    nb=P//64;n=nb*64;b=c.global_maps['positions'].element_size()
    layer=H*n*(32*(1+(c.plan['bits']==2))+4)+H*nb*232+4*H*64*D+4*H*D*t
    return s+3*P*b+256+4*t+L*layer

def unpack(p,bits,n):
    shifts=torch.arange(0,8,bits,device=p.device)
    return ((p[...,None].long()>>shifts)&((1<<bits)-1)).reshape(*p.shape[:-1],-1)[...,:n]
def q4restore(p,scale,offset,width,group):
    q=unpack(p,4,width).half()
    return (q.reshape(-1,group)*scale.reshape(-1,1)+offset.reshape(-1,1)).reshape(*p.shape[:-1],width)
def native_rope(c,x,start=0):
    pos=c.global_maps['positions'][:,start:start+x.shape[1]].long()[:,None]
    dummy=torch.empty((1,x.shape[1],1),device=x.device,dtype=x.dtype)
    cos,sin=c.model.model.language_model.rotary_emb(dummy,pos)
    _,out=apply_multimodal_rotary_pos_emb(x[None],x[None],cos,sin,c.model.config.text_config.rope_scaling['mrope_section'])
    return out[0]

@torch.no_grad()
def stored_reader_gate(c,l):
    st=c.states[l];expected={}
    if c.name.startswith('NOVA'):
        t=c.tables[l];n=st['nlow'];g=c.plan['group_size'];ng=D//g
        ix=unpack(st['K_index'][:,:n],c.plan['K_index_bits'],ng).permute(0,2,1)
        words=torch.gather(t['centroids'].float(),2,ix[...,None].expand(-1,-1,-1,g)).permute(0,2,1,3).reshape(H,n,D)
        k=(torch.matmul(words*st['rho'][:,:n].float(),t['A'].float().transpose(1,2))+t['mean'].float()[:,None]).bfloat16()
        q=unpack(st['V_index'][:,:n],c.plan['v_bits'],D).float()
        v=(q*2-1)*st['V_scale'][:,:n].float()if c.plan['v_mode']=='sign1'else(q-st['V_zero'][:,:n].float())*st['V_scale'][:,:n].float()
        v=torch.matmul(v,t['V_rotation'].float().transpose(1,2)).bfloat16()
        expected={'K':torch.cat([st['firstK'][:,:st['nf']],k,st['recentK'][:,:st['nr']]],1),
                  'V':torch.cat([st['firstV'][:,:st['nf']],v,st['recentV'][:,:st['nr']]],1)}
    else:
        n=st['quantized'];nb=n//64;book=c.tables[0]['book'+str(c.plan['bits'])];had=c.tables[0]['H']
        for side in ['K','V']:
            z=book[st[side+'_index'][:,:n].long()].reshape(H,n,D)
            if c.plan['bits']==2:z=torch.where(unpack(st[side+'_sign'][:,:n],1,D).bool(),z,-z)
            z=(z*st[side+'_norm2'][:,:n]).half()
            mean=q4restore(st[side+'_mean_index'][:,:nb],st[side+'_mean_scale'][:,:nb],st[side+'_mean_offset'][:,:nb],128,32)
            mean=mean[:,:,None].expand(-1,-1,64,-1).reshape(H,n,D)
            if side=='K':z=(torch.matmul(z,had)+native_rope(c,mean)).half()
            else:z=(z+mean).half()
            norm=q4restore(st[side+'_norm1_index'][:,:nb],st[side+'_norm1_scale'][:,:nb],st[side+'_norm1_offset'][:,:nb],64,64).reshape(H,n,1).clamp_min(1e-5)
            z=(z*norm).half()
            if side=='V':z=torch.matmul(z,had)
            pending=st['pending']
            if pending:
                tail=st[side+'_pending'][:,:pending]
                z=torch.cat([z,native_rope(c,tail,n)if side=='K'else tail.half()],1)
            expected[side]=z.bfloat16()
    ti=c.global_maps['text'].long();actual=dict(zip(['K','V'],c.decode(l)));records=[]
    for side in ['K','V']:
        expected[side][:,ti]=st['text_FP_'+side]
        err=float((actual[side].float()-expected[side].float()).norm()/expected[side].float().norm().clamp_min(1e-12))
        assert err<.002,(c.name,l,side,err)
        records.append({'layer':l,'side':side,'relative_L2':err,'actual_stored_indices':True,'all4_KV_heads':True,'native_MRoPE_or_post_MRoPE_keys':True})
    return records

@torch.no_grad()
def pair_encoder_gate(c,l,teacher):
    st=c.states[l];records=[]
    for h in [0,2]:
        reference=object.__new__(NOVABudget if c.name.startswith('NOVA')else NSN64Reference)
        reference.plan=c.plan
        dst={k:torch.zeros_like(v[h:h+2])for k,v in st.items()if torch.is_tensor(v)and k!='arena'}
        if c.name.startswith('NOVA'):
            reference.tables=[{k:v[h:h+2]for k,v in c.tables[l].items()}];reference.groups=D//c.plan['group_size'];reference.gdim=c.plan['group_size'];reference.index_bits=c.plan['K_index_bits']
            a=st['nf'];n=st['nlow'];k=teacher.states[l]['K_post'][h:h+2,a:a+n].cuda();v=teacher.states[l]['V'][h:h+2,a:a+n].cuda()
            reference.write_keys(0,dst,k,0);reference.write_values(0,dst,v,0)
            fields=['K_index','rho','V_index','V_scale']+(['V_zero']if c.plan['v_mode']!='sign1'else[])
        else:
            reference.tables=c.tables;reference.position_storage=c.position_storage;reference.position_inv_freq=c.position_inv_freq;reference.position_attention_scaling=c.position_attention_scaling
            dst.update(n=0,pending=0,quantized=0)
            reference.append(0,dst,teacher.states[l]['K_post'][h:h+2][None].cuda(),teacher.states[l]['V'][h:h+2][None].cuda(),teacher.states[l]['K_pre'][h:h+2][None].cuda())
            fields=[k for k in dst if k not in ['n','pending','quantized','text_FP_K','text_FP_V']]
            assert(dst['pending'],dst['quantized'])==(st['pending'],st['quantized'])
        for key in fields:assert torch.equal(dst[key],st[key][h:h+2]),(c.name,l,h,key)
        records.append({'layer':l,'head_pair':[h,h+1],'encoder_matches_original_2head_reference':True,'matched_fields':fields})
    return records
