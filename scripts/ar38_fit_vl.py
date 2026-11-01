"""Unchanged VL shared-RVQ and FP16-fit/FP8-export GSRQ recipes, two fresh RNGs."""
import argparse,hashlib,json,pathlib,sys,time,gc
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ar26kv.runtime import save

def main():
 import torch
 from r14kv.fitting import nearest,hadamard,fit_rq
 from safetensors import safe_open
 ap=argparse.ArgumentParser();ap.add_argument('--replicate',type=int,choices=[1,2],required=True);a=ap.parse_args();rep=a.replicate;offset=rep*1000003;start=time.time();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(26*2**30/torch.cuda.get_device_properties(0).total_memory)
 out=ROOT/f'data/runs/AR38_VL_FIT_r{rep}_v2';out.mkdir(parents=True,exist_ok=False);base=ROOT/f'artifacts/ar38/replicas/r{rep}';cal=ROOT/'data/runs/AR26_VL_MEDIA_CAL24_v1/completion.json';done=json.loads(cal.read_text());assert done['completed'];pools=[torch.load(ROOT/r['sample_file'],map_location='cpu',weights_only=True)for r in done['rows']];model=ROOT/'artifacts/ar26/models/Qwen2.5-VL-7B-Instruct';index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];metrics={'K':[],'V':[]}
 for l in range(28):
  key=f'model.layers.{l}.self_attn.o_proj.weight'
  if key not in index:key=f'model.language_model.layers.{l}.self_attn.o_proj.weight'
  with safe_open(model/index[key],framework='pt')as f:w=f.get_tensor(key).float()
  metrics['V'].append(w.reshape(3584,4,7,128).square().sum((0,2)));metrics['K'].append(torch.stack([p[l]['q_moment']for p in pools]).mean(0)/128.)
 raw={s:torch.stack([torch.cat([p[l][s]for p in pools],1)for l in range(28)])for s in ['K','V']};old=torch.load(ROOT/'artifacts/ar26/packages/STI_RVQ/global.pt',map_location='cpu',weights_only=True);table={k:v.clone()for k,v in old.items()if not k.endswith('_books')};h=table['H'].cuda();traces={}
 save(out/'run_manifest.json',{'replicate':rep,'seed_offset':offset,'calibration_SHA256':hashlib.sha256(cal.read_bytes()).hexdigest(),'families':['STI-RVQ','GSRQExport'],'hyperparameters_unchanged':True,'started_epoch':start})
 for si,s in enumerate(['K','V']):
  x=raw[s].cuda().float();metric=torch.stack(metrics[s]).cuda();assert torch.equal(raw[s].float().mean(2).half(),old[s+'_mean'])and torch.equal(raw[s].float().std(2,unbiased=False).clamp_min(1e-4).half(),old[s+'_scale']);mag=(x.square()*metric[:,:,None]).sum(-1).sqrt();weights=torch.log1p(mag/mag.median(2,keepdim=True).values.clamp_min(1e-12)).clamp_min(1e-8);u=(x-table[s+'_mean'].cuda().float()[:,:,None])/table[s+'_scale'].cuda().float()[:,:,None];gain=u.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6).bfloat16();z=((u/gain.float())@h).reshape(1,-1,16);bw=weights[:,:,:,None].expand(-1,-1,-1,8).reshape(1,-1);pick=np.random.default_rng(17261023+16+si*1000+offset).choice(z.shape[1],262144,replace=False);ix=torch.tensor(pick,device='cuda');res=z[:,ix].clone();wt=bw[:,ix].clone();books=[];hist=[]
  for r in range(6):
   gen=torch.Generator(device='cuda').manual_seed(17261023+1600+1000*si+r+offset);c=res[:,torch.randperm(res.shape[1],generator=gen,device='cuda')[:256]].clone()
   for it in range(20):
    ids=nearest(res,c);ss=torch.zeros_like(c);cnt=torch.zeros(1,256,device='cuda');ss.scatter_add_(1,ids[...,None].expand(-1,-1,16),res*wt[...,None]);cnt.scatter_add_(1,ids,wt);c=torch.where((cnt>0)[...,None],ss/cnt.clamp_min(1e-20)[...,None],c)
   dep=c.half();ids=nearest(res,dep.float());res-=torch.gather(dep.float(),1,ids[...,None].expand(-1,-1,16));books.append(dep[0].cpu());hist.append({'stage':r+1,'weighted_MSE':float((res.square().sum(-1)*wt).sum()/wt.sum())})
  table[s+'_books']=torch.stack(books);traces[s]=hist;del x,metric,mag,weights,u,gain,z,bw,res,wt;torch.cuda.empty_cache()
 package=base/'artifacts/ar26/packages/STI_RVQ';package.mkdir(parents=True,exist_ok=False);torch.save(table,package/'global.pt');save(package/'completion.json',{'completed':True,'replicate':rep,'model':'Qwen2.5-VL-7B-Instruct','static_tensor_bytes':sum(v.untyped_storage().nbytes()for v in table.values()),'global_SHA256':hashlib.sha256((package/'global.pt').read_bytes()).hexdigest(),'trace':traces,'seed_offset':offset,'records':[{'file':'global.pt','SHA256':hashlib.sha256((package/'global.pt').read_bytes()).hexdigest()}]})
 for l in range(28):
  bank={}
  for si,s in enumerate(['K','V']):
   x=raw[s][l].cuda().float();norm=(x.square()*metrics[s][l].cuda()[:,None]).sum(-1).sqrt();books,trace=fit_rq(x,norm,[14261021+10000*si+l*10+h+offset for h in range(4)],stages=7,iterations=40);bank[s]=books.to(torch.float8_e4m3fn);save(out/f'GSRQ_layer{l:02}_{s}.json',trace)
  for r in[1,3,6,7]:
   stage='ar26'if r==1 else'ar31';folder=base/f'artifacts/{stage}/packages/GSRQExport/R{r}';folder.mkdir(parents=True,exist_ok=True);torch.save({s:v[:,:r].clone()for s,v in bank.items()},folder/f'layer{l:02}.pt')
  print(json.dumps({'layer':l,'replicate':rep,'elapsed_s':time.time()-start}),flush=True);gc.collect();torch.cuda.empty_cache()
 for r in[1,3,6,7]:
  stage='ar26'if r==1 else'ar31';folder=base/f'artifacts/{stage}/packages/GSRQExport/R{r}';save(folder/'completion.json',{'completed':True,'replicate':rep,'family':'GSRQExport','Rmax':r,'static_tensor_bytes':28*4*2*r*256*128,'seed_offset':offset,'FP8_aware_fit':False,'records':[{'file':f'layer{l:02}.pt','SHA256':hashlib.sha256((folder/f'layer{l:02}.pt').read_bytes()).hexdigest()}for l in range(28)]})
 save(out/'completion.json',{'completed':True,'replicate':rep,'elapsed_s':time.time()-start,'model':'VL','families':['STI-RVQ','GSRQExport']})
if __name__=='__main__':main()
