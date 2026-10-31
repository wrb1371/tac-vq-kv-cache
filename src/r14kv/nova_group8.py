"""Legal NOVA g8/C256 K1V2 strong budget control, with unchanged Q/K statistics."""
import torch
from .codecs import NOVA,pack2
from .fitting import nearest

class NOVAG8K1V2(NOVA):
    def __init__(self,directory,device='cuda'):
        super().__init__(directory,device);assert all(t['centroids'].shape==(2,16,256,8) for t in self.tables)
        self.method='NOVA-Omni-BM-g8C256-K1V2'
        self.metadata={**self.metadata,'budget_control_only':True,'K_bits_per_coordinate':1,'V_bits_per_coordinate':2,'K_group_dimension':8,'K_groups':16,'K_centroids_per_group':256,
            'K_index_layout':'16 uint8 indices per head/token, real1indexBPA; no4bit unpack',
            'statistics':'same full Q/K covariance, mean and V basis as complete M0; only stratified grouping and necessary K book refit changed',
            'grouping':'descending eigenvalue ranks g,g+16,...,g+112','source_legal_configuration':'author trainer G8,bpc1; not a TaSQ component or a new multimodal mechanism',
            'static_vs_g4C16':'larger actual FP8 K books paid; no equal-byte claim'}
    @torch.no_grad()
    def encode_low(self,l,k,v):
        t=self.tables[l];h,n,d=k.shape;r=torch.bmm((k-t['mean'][:,None]).bfloat16(),t['F']).float();rho=r.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
        z=(r/rho).reshape(h,n,16,8);groups=z.permute(0,2,1,3).reshape(h*16,n,8);cb=t['centroids'].float().reshape(h*16,256,8)
        ix=nearest(groups,cb).reshape(h,16,n).transpose(1,2).contiguous().to(torch.uint8)
        vr=torch.bmm(v,t['V_rotation']).float();threshold=vr.abs().sort(-1).values[...,int(.92*128):int(.92*128)+1]
        clipped=torch.maximum(torch.minimum(vr,threshold),-threshold);lo=clipped.amin(-1,keepdim=True);hi=clipped.amax(-1,keepdim=True)
        scale=(hi-lo).clamp_min(1e-8)/3;zero=-lo/scale;q=(clipped/scale+zero+.5).to(torch.int32).clamp(0,3).to(torch.uint8)
        return {'K_index':ix,'rho':rho,'V_index':pack2(q),'V_scale':scale,'V_zero':zero}
