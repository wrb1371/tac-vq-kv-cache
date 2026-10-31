"""Explicit budget controls; neither alone is a multimodal algorithm novelty."""
import torch
from .codecs import NOVA,GSRQ
from .fitting import nearest

def pack4(x):return x[...,0::2].to(torch.uint8)|(x[...,1::2].to(torch.uint8)<<4)
def unpack4(x):return torch.stack([x&15,x>>4],-1).flatten(-2)

class NOVAK1V2(NOVA):
    def __init__(self,directory,device='cuda'):
        super().__init__(directory,device);assert all(t['centroids'].shape==(2,32,16,4) for t in self.tables)
        check=torch.arange(16,dtype=torch.uint8).repeat(2);assert torch.equal(unpack4(pack4(check)),check)
        self.method='NOVA-Omni-BM-K1V2'
        self.metadata={**self.metadata,'budget_control_only':True,'K_bits_per_coordinate':1,'V_bits_per_coordinate':2,'K_centroids_per_group':16,'K_index_layout':'32 real4bit group indices, packed two per byte','source_legal_configuration':'author trainer bpc1,G4; independent exact4bit reference storage','not_TaSQ_nova1b':True,'otherwise_frozen_complete_M0':True}
    @torch.no_grad()
    def encode_low(self,l,k,v):
        t=self.tables[l];h,n,d=k.shape;r=torch.bmm((k-t['mean'][:,None]).bfloat16(),t['F']).float()
        rho=r.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8);z=(r/rho).reshape(h,n,32,4)
        groups=z.permute(0,2,1,3).reshape(h*32,n,4);cb=t['centroids'].float().reshape(h*32,16,4)
        ix=nearest(groups,cb).reshape(h,32,n).transpose(1,2).contiguous().to(torch.uint8)
        vr=torch.bmm(v,t['V_rotation']).float();threshold=vr.abs().sort(-1).values[...,int(.92*128):int(.92*128)+1]
        clipped=torch.maximum(torch.minimum(vr,threshold),-threshold);lo=clipped.amin(-1,keepdim=True);hi=clipped.amax(-1,keepdim=True)
        scale=(hi-lo).clamp_min(1e-8)/3;zero=-lo/scale;q=(clipped/scale+zero+.5).to(torch.int32).clamp(0,3).to(torch.uint8)
        from .codecs import pack2
        return {'K_index':pack4(ix),'rho':rho,'V_index':pack2(q),'V_scale':scale,'V_zero':zero}
    @torch.no_grad()
    def synthesize_low(self,l,low):return super().synthesize_low(l,{**low,'K_index':unpack4(low['K_index'])})

class NOVAMetadataBF16(NOVA):
    def __init__(self,directory,device='cuda'):
        super().__init__(directory,device)
        self.method='NOVA-Omni-BM-metadataBF16'
        self.metadata={**self.metadata,'engineering_control_only':True,'rho_and_V_scale_zero_dtype':'bfloat16','change':'encode assignment uses parent RMS; persist BF16 rho, V scale/zero instead of FP32; same deployed books, transform and64/256band'}
    @torch.no_grad()
    def encode_low(self,l,k,v):
        payload=super().encode_low(l,k,v)
        for name in ['rho','V_scale','V_zero']:payload[name]=payload[name].bfloat16()
        return payload

class GSRQCompactFP8(GSRQ):
    def __init__(self,directory,stages=16,device='cuda'):
        super().__init__(directory,stages,device)
        errors=[]
        for l,tables in enumerate(self.tables):
            for kind in ['K','V']:
                source=tables[kind];book=source.to(torch.float8_e4m3fn);assert torch.isfinite(book.float()).all()
                errors.append({'layer':l,'kind':kind,'relative_RMSE':float(((book.float()-source.float()).square().sum()/source.float().square().sum().clamp_min(1e-30)).sqrt()),'maximum_centroid_abs':float(source.float().abs().max())})
                tables[kind]=book
        self.method=f'GSRQ-Omni-BM-R{stages}-FP8'
        self.metadata={**self.metadata,'engineering_control_only':True,'codebook_dtype':'float8_e4m3fn','persistent_FP16_mirror':False,'nearest_and_decode':'actual deployed FP8 centroid bytes, transient FP32 per-stage dequantization','offline_retraining':False,'parent_FP16_to_FP8_errors':errors,'static_tables_count_all_loaded_stages':True}
