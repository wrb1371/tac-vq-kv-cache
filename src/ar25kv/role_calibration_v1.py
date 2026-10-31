"""New conditional targets for A=min(r+1,6), I=r; never splice old losses."""
import math
import torch
from ar22kv.calibration import prefixes,FRACTIONS,GOPS
from ar22kv.matching import fixed_reference

@torch.no_grad()
def measure_role_layer(layer,native,query_moments,value_metric,table,video,text,grid,importance,sample_count=96):
 raw={'K':native['K_pre'][0].cuda().float(),'V':native['V'][0].cuda().float()}
 metrics={'K':query_moments.cuda()/128.,'V':value_metric.cuda()};nt,h,w=grid;p=h*w;m=len(video);assert m==nt*p
 ordinal=torch.arange(m,device=video.device);qs=text[torch.linspace(0,len(text)-1,min(sample_count,len(text)),device=text.device).long()]
 query={s:torch.stack([((raw[s][:,qs]-y.float()).square()*metrics[s][:,None]).sum(-1).mean() for y in prefixes(raw[s][:,qs],table,s,layer)]) for s in ['K','V']}
 output={'norm':{}}
 for gop in GOPS:
  anchors=ordinal[(ordinal//p)%gop==0];targets=ordinal[(ordinal//p)%gop!=0]
  a=anchors[torch.linspace(0,len(anchors)-1,min(sample_count,len(anchors)),device=video.device).long()]
  t=targets[torch.linspace(0,len(targets)-1,min(sample_count,len(targets)),device=video.device).long()]
  codes,scores=fixed_reference(raw,video,t,[table[s+'_scale'][layer] for s in ['K','V']],h,w,gop,None)
  ry=(t[None]//w%h+codes.long()//3-1).clamp(0,h-1);rx=(t[None]%w+codes.long()%3-1).clamp(0,w-1)
  refs=video[t[None]//p//gop*gop*p+ry*w+rx].long();order=torch.argsort(scores,dim=1,descending=True,stable=True);n=len(t);record={}
  for side in ['K','V']:
   anchor_values=raw[side][:,video[a].long()];ap=prefixes(anchor_values,table,side,layer)
   ae=torch.stack([((anchor_values-y.float()).square()*metrics[side][:,None]).sum(-1).mean() for y in ap])
   reference=raw[side].gather(1,refs[...,None].expand(-1,-1,128));rp=prefixes(reference,table,side,layer);target=raw[side][:,video[t].long()];loss=torch.empty((7,6),device=video.device)
   for ri in range(1,7):
    ra=min(6,ri+1);prediction=rp[ra-1].float();residual=target-prediction
    corrected=(prediction+prefixes(residual,table,side,layer,independent=False,levels=ri)[-1].float()).bfloat16().float()
    copy=(residual.square()*metrics[side][:,None]).sum(-1).gather(1,order)
    repair=((target-corrected).square()*metrics[side][:,None]).sum(-1).gather(1,order)
    for fi,f in enumerate(FRACTIONS):
     count=math.ceil(n*f);loss[fi,ri-1]=(repair[:,:count].sum()+copy[:,count:].sum())/(2*n)
   assert torch.isfinite(loss).all() and (loss>=0).all()
   record[side]={'anchor':ae[[1,2,3,4,5,5]].cpu().tolist(),'target':loss.cpu().tolist(),'query':query[side].cpu().tolist()}
  output['norm'][str(gop)]=record
 return output
