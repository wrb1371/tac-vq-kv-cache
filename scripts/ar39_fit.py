"""Three independent shared RVQ, GSRQ, NOVA and CQ fits on common WorldSense pools."""
import argparse,gc,json,pathlib,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np,torch
from ar39cal.common import *
from r14kv.fitting import nearest,hadamard,fit_rq,qpca,value_basis,lloyd_group
from ar37cq.fitting import kmeans,validate
def tensor_record(path,t,l=0):return {'layer':l,'file':path.name,'SHA256':sha(path),'tensors':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'storage_bytes':v.untyped_storage().nbytes()}for k,v in t.items()}}
def export(folder,t,l=None):
 folder.mkdir(parents=True,exist_ok=True);path=folder/('global.pt'if l is None else f'layer{l:02}.pt');assert not path.exists();t={k:v.cpu().contiguous().clone()for k,v in t.items()};torch.save(t,path);return tensor_record(path,t,l or 0)
def finish(folder,rows,**kw):
 save(folder/'completion.json',{'completed':True,'CPU_export':True,'records':rows,'static_tensor_bytes':sum(v['storage_bytes']for r in rows for v in r['tensors'].values()),'calibration':'WorldSense64, video-text only','labels_used':kw.get('family')!='NOVA','no_benchmark_media_or_QA_used':True,**kw})
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['Omni','VL'],required=True);ap.add_argument('--replicate',type=int,choices=[0,1,2],required=True);a=ap.parse_args();device(16);start=time.time();out=ROOT/f'data/runs/AR39_FIT_{a.model}_r{a.replicate}_v1';out.mkdir(parents=True,exist_ok=False);cal=receipt(a.model);pools=[]
 for r in cal['rows']:assert sha(ROOT/r['path'])==r['SHA256'];pools.append(torch.load(ROOT/r['path'],weights_only=True,map_location='cpu'))
 L,H,N,width=geometry(a.model);rep=a.replicate;offset=rep*1000003;root=base(rep);shared=root/('artifacts/ar17/packages/PSRVQ_d16'if a.model=='Omni'else'artifacts/ar26/packages/STI_RVQ');mask=torch.arange(256)%4!=0
 save(out/'run_manifest.json',{'model':a.model,'replicate':rep,'seed_offset':offset,'protocol_SHA256':sha(ROOT/'configs/ar39/protocol.json'),'calibration_receipt_SHA256':sha(capture_run(a.model)/'completion.json'),'code_SHA256':sha(__file__),'all64_sources_used':True,'holdout':'every fourth sampled vector; fitted normalization uses training vectors','WorldSense_only':True})
 try:
  # Independent per-model normalization; no old fitted centroid is inherited.
  table={'H':hadamard(128).float()};raw={};weights={};traces={}
  for si,s in enumerate(['K','V']):
   x=torch.stack([torch.cat([p[l][s][:,mask]for p in pools],1)for l in range(L)]).float();g=torch.stack([torch.cat([p[l][s+'_grad'][:,mask].norm(dim=-1)for p in pools],1)for l in range(L)]);mean=x.mean(2).half();scale=x.std(2,unbiased=False).clamp_min(1e-4).half();table[s+'_mean']=mean;table[s+'_scale']=scale;raw[s]=x;weights[s]=g
   u=(x.cuda()-mean.cuda().float()[:,:,None])/scale.cuda().float()[:,:,None];gain=u.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6).bfloat16();z=((u/gain.float())@table['H'].cuda()).reshape(1,-1,16);w=torch.log1p(g.cuda()/(g.cuda().median(-1,keepdim=True).values+1e-12)).clamp_min(1e-8);ww=w[...,None].expand(-1,-1,-1,8).reshape(1,-1);rng=np.random.default_rng(17261023+16+si*1000+offset);idx=torch.tensor(rng.choice(z.shape[1],262144,replace=False),device='cuda');res=z[:,idx].clone();wt=ww[:,idx].clone();books=[];hist=[]
   for stage in range(6):
    gen=torch.Generator(device='cuda').manual_seed(17261023+1600+1000*si+stage+offset);c=res[:,torch.randperm(res.shape[1],generator=gen,device='cuda')[:256]].clone()
    for it in range(20):
     ids=nearest(res,c);ss=torch.zeros_like(c);cnt=torch.zeros(1,256,device='cuda');ss.scatter_add_(1,ids[...,None].expand(-1,-1,16),res*wt[...,None]);cnt.scatter_add_(1,ids,wt);c=torch.where((cnt>0)[...,None],ss/cnt.clamp_min(1e-20)[...,None],c)
    dep=c.half();ix=nearest(res,dep.float());res-=torch.gather(dep.float(),1,ix[...,None].expand(-1,-1,16));books.append(dep[0].cpu());hist.append({'stage':stage+1,'weighted_MSE':float((res.square().sum(-1)*wt).sum()/wt.sum())});print(json.dumps({'family':'shared','side':s,'stage':stage+1,'seconds':time.time()-start}),flush=True)
   table[s+'_books']=torch.stack(books);traces[s]=hist;del u,gain,z,w,ww,idx,res,wt,c;gc.collect();torch.cuda.empty_cache()
  sr=export(shared,table);finish(shared,[sr],family='PSRVQ'if a.model=='Omni'else'STI-RVQ',group_dim=16,stages=6,Rmax_K=6,Rmax_V=6,model=a.model,replicate=rep,trace=traces,global_SHA256=sr['SHA256'],adapters_inherited_exact=False)
  save(out/'shared_complete.json',{'completed':True,'path':str(shared.relative_to(ROOT)),'SHA256':sha(shared/'completion.json')});mom=torch.load(pool(a.model)/'moments.pt',weights_only=True,map_location='cpu');count=mom['prefix_token_count'];cq_pairs=sorted({(v['C'],v['c'])for k,v in read(ROOT/'configs/ar37/selected.json')['groups'].items()if k.startswith(a.model+'/')});gs=[2,4,8]if a.model=='Omni'else[1,3,6,7];gs_records={r:[]for r in gs};nova_records={c:[]for c in([16]if a.model=='Omni'else[16,256])};cq_records={pair:[]for pair in cq_pairs}
  for l in range(L):
   # Retain original FP16-fit/FP8-export residual update semantics.
   gb={}
   for si,s in enumerate(['K','V']):
    seeds=[14261021+(l*4+si*2+h if a.model=='Omni'else 10000*si+l*10+h)+offset for h in range(H)];b,tr=fit_rq(raw[s][l].cuda(),weights[s][l].cuda(),seeds,stages=max(gs),iterations=40);gb[s]=b.to(torch.float8_e4m3fn);save(out/f'gsrq_{l:02}_{s}.json',tr)
   for r in gs:
    rel=f'artifacts/ar22/private_tiers/R{r}'if a.model=='Omni'else f'artifacts/{"ar26"if r==1 else"ar31"}/packages/GSRQExport/R{r}';gs_records[r].append(export(root/rel,{s:v[:,:r].clone()for s,v in gb.items()},l))
   del gb;torch.cuda.empty_cache()
   # NOVA uses real native attention-product covariance, no task-label weights.
   nv={C:{k:[]for k in ['F','A','mean','V_rotation','centroids']}for C in nova_records}
   for h in range(H):
    f0,a0,mean,diag=qpca(mom['q2'][l,h].cuda(),mom['k2'][l,h].cuda(),mom['ksum'][l,h].cuda(),count);assert diag['FP64_identity_max_abs']<1e-6,diag;old=[i+j*32 for i in range(32)for j in range(4)];inv={r:i for i,r in enumerate(old)};perm=torch.tensor([inv[i+j*16]for i in range(16)for j in range(8)],device='cuda');f=f0[:,perm];a_inv=a0[:,perm];x=torch.cat([p[l]['K_post'][h,mask]for p in pools]).cuda().double();z=(x-mean)@f;z/=z.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8);grouped=z.reshape(-1,16,8).permute(1,0,2).float().contiguous();seed=14261021+l*H+h+offset;vr=value_basis(mom['o2'][l,h].cuda()/count).bfloat16()
    for C,t in nv.items():
     cent=lloyd_group(grouped,seed=seed,iterations=25,clusters=C);sc=cent.abs().amax((-2,-1),keepdim=True).clamp_min(1e-12)/57344.;trained=((cent/sc).to(torch.float8_e5m2).float()*sc).half();bundle={'F':f.bfloat16(),'A':a_inv.bfloat16(),'mean':mean.bfloat16(),'V_rotation':vr,'centroids':trained.to(torch.float8_e5m2)}
     for k,v in bundle.items():assert torch.isfinite(v.float()).all();t[k].append(v.cpu().clone())
    del x,z,grouped;torch.cuda.empty_cache()
   for C,t in nv.items():
    rel='artifacts/round15/packages/nova_g8c16'if a.model=='Omni'else f'artifacts/ar30/packages/NOVA_g8_C{C}';nova_records[C].append(export(root/rel,{k:torch.stack(v)for k,v in t.items()},l))
   for C,c in cq_pairs:
    bank={};trace={};J=128//c
    for si,s in enumerate(['K','V']):
     xx=raw[s][l].reshape(H,-1,J,c).permute(0,2,1,3).reshape(H*J,-1,c).cuda();grad=torch.cat([p[l][s+'_grad'][:,mask]for p in pools],1).reshape(H,-1,J,c).permute(0,2,1,3).reshape(H*J,-1,c).cuda();wt=grad.square().sum(-1);seed=20261031+100000*l+10000*si+C+c+offset;cent,tr=kmeans(xx,wt,C,seed);bank[s]=cent.reshape(H,J,C,c).cpu();trace[s]=tr;del xx,grad,wt,cent;torch.cuda.empty_cache()
    cq_records[C,c].append(export(root/f'artifacts/ar37/packages/{a.model}/C{C}_c{c}',bank,l));save(out/f'cq_{l:02}_C{C}_c{c}.json',trace)
   save(out/'progress.json',{'completed_layers':l+1,'total_layers':L,'elapsed_s':time.time()-start});print(json.dumps({'model':a.model,'replicate':rep,'layer':l,'seconds':time.time()-start}),flush=True);gc.collect();torch.cuda.empty_cache()
  for r,rows in gs_records.items():
   rel=f'artifacts/ar22/private_tiers/R{r}'if a.model=='Omni'else f'artifacts/{"ar26"if r==1 else"ar31"}/packages/GSRQExport/R{r}';finish(root/rel,rows,family='GSRQ',Rmax=r,Rmax_K=r,Rmax_V=r,private_stages=r,replicate=rep)
  for C,rows in nova_records.items():
   rel='artifacts/round15/packages/nova_g8c16'if a.model=='Omni'else f'artifacts/ar30/packages/NOVA_g8_C{C}';finish(root/rel,rows,family='NOVA',replicate=rep)
  for(C,c),rows in cq_records.items():finish(root/f'artifacts/ar37/packages/{a.model}/C{C}_c{c}',rows,family='CQ',C=C,c=c,replicate=rep)
  save(out/'completion.json',{'completed':True,'model':a.model,'replicate':rep,'elapsed_s':time.time()-start,'all64_sources_used':True,'families':['STI-RVQ','GSRQ-export','CQ','NOVA-KV-g8'],'GPU_peak_MiB':torch.cuda.max_memory_allocated()/2**20,'GPU1_used':False})
 except BaseException as e:save(out/'failure.json',{'completed':False,'error':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
