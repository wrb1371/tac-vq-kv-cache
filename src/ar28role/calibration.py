"""Role-conditioned distortion against the actual quantized reference."""
import math
import torch
from ar28kv.calibration import prefixes,GOPS
from ar26kv.matching import fixed_reference,references
FRACTIONS=[0.,1/32,1/16,1/8,3/16,1/4,3/8,1/2,5/8,3/4,7/8,1.]
PATTERNS={'41':(4,4,1,1),'42':(4,4,2,2),'51':(5,5,1,1),
          '61':(6,6,1,1),'62':(6,6,2,2),'Kprotect':(5,4,2,1),'Vprotect':(4,5,1,2)}

@torch.no_grad()
def measure(layer,native,qmoment,vmetric,table,vi,grid,count=96):
    raw={s:native[k].cuda().float()for s,k in [('K','K_pre'),('V','V')]}
    metric={'K':qmoment.cuda()/128.,'V':vmetric.cuda()};nt,gh,gw=grid;p=gh*gw
    ordinal=torch.arange(len(vi),device='cuda');result={}
    for g in GOPS:
        targets=ordinal[(ordinal//p)%g!=0]
        sample=targets[torch.linspace(0,len(targets)-1,min(count,len(targets)),device='cuda').long()]
        codes,scores=fixed_reference(raw,vi.to(torch.int32),sample,[table[s+'_scale'][layer]for s in ['K','V']],gh,gw,g)
        refs=vi[references(sample,codes,gh,gw,g)].long();order=torch.argsort(scores,dim=1,descending=True,stable=True)
        result[str(g)]={}
        for s in ['K','V']:
            anchor=raw[s].gather(1,refs[...,None].expand(-1,-1,128));target=raw[s][:,vi[sample]]
            decoded=prefixes(anchor,table,s,layer);records={};n=len(sample)
            for ra in [4,5,6]:
                pred=decoded[ra-1].float();residual=target-pred
                corrected=prefixes(residual,table,s,layer,independent=False,levels=4 if ra==4 else 2)
                copied=(residual.square()*metric[s][:,None]).sum(-1).gather(1,order)
                for ri in ([1,2,4]if ra==4 else[1,2]):
                    reconstruction=(pred+corrected[ri-1].float()).bfloat16().float()
                    error=((target-reconstruction).square()*metric[s][:,None]).sum(-1).gather(1,order)
                    losses=[]
                    for fraction in FRACTIONS:
                        k=math.ceil(n*fraction);losses.append((error[:,:k].sum(1)+copied[:,k:].sum(1))/n)
                    z=torch.stack(losses,1);assert torch.isfinite(z).all();records[f'{ra}_{ri}']=z.cpu().tolist()
            result[str(g)][s]=records
    return result
