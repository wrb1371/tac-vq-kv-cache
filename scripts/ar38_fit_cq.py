"""Independent seeds for the six fixed CQ operating formats; no QA selection."""
import argparse,gc,hashlib,json,os,pathlib,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from ar37cq.fitting import kmeans,validate
from ar26kv.runtime import save

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);a=ap.parse_args();start=time.time();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(26*2**30/torch.cuda.get_device_properties(0).total_memory);run=f'AR38_CQ_FIT_{a.model}_v1';out=ROOT/'data/runs'/run;out.mkdir(parents=True,exist_ok=False)
 selected=json.loads((ROOT/'configs/ar37/selected.json').read_text())['groups'];pairs=sorted({(p['C'],p['c'])for key,p in selected.items()if key.startswith(a.model+'/')});cal=json.loads((ROOT/f'data/runs/AR38_CQ_CAL_{a.model}_v1/completion.json').read_text());assert cal['completed']and len(cal['rows'])==24;pools=[torch.load(ROOT/r['path'],weights_only=True)for r in cal['rows']];L,H=(36,2)if a.model=='Omni'else(28,4);done=[]
 save(out/'run_manifest.json',{'pairs':pairs,'replicates':[1,2],'original_seed_base':20261031,'offsets':[1000003,2000006],'calibration_SHA256':hashlib.sha256((ROOT/f'data/runs/AR38_CQ_CAL_{a.model}_v1/completion.json').read_bytes()).hexdigest(),'GPU':os.environ['CUDA_VISIBLE_DEVICES'],'fixed_capacity_without_QA_selection':True})
 try:
  for rep in[1,2]:
   for C,c in pairs:
    folder=ROOT/f'artifacts/ar38/replicas/r{rep}/artifacts/ar37/packages/{a.model}/C{C}_c{c}';folder.mkdir(parents=True,exist_ok=False);records=[];J=128//c
    for l in range(L):
     bank={};r={'layer':l,'sides':{}}
     for si,side in enumerate(['K','V']):
      xx=torch.cat([p[l][side][:,torch.arange(512)%4!=0]for p in pools],1).float().reshape(H,-1,J,c).permute(0,2,1,3).reshape(H*J,-1,c).cuda();gg=torch.cat([p[l][side+'_grad'][:,torch.arange(512)%4!=0]for p in pools],1).float().reshape(H,-1,J,c).permute(0,2,1,3).reshape(H*J,-1,c).cuda();w=gg.square().sum(-1);del gg
      seed=20261031+100000*l+10000*si+C+c+rep*1000003;z,trace=kmeans(xx,w,C,seed);del xx,w;z=z.reshape(H,J,C,c).cpu();bank[side]=z;r['sides'][side]={'seed':seed,'training':trace,'heldout':[validate(p[l][side][:,::4],p[l][side+'_grad'][:,::4],z,c)for p in pools]}
     f=folder/f'layer{l:02}.pt';torch.save(bank,f);r['SHA256']=hashlib.sha256(f.read_bytes()).hexdigest();records.append(r);save(folder/'progress.json',{'layers':records});print(json.dumps({'replicate':rep,'model':a.model,'C':C,'c':c,'layer':l,'elapsed_s':time.time()-start}),flush=True);gc.collect();torch.cuda.empty_cache()
    save(folder/'completion.json',{'completed':True,'replicate':rep,'model':a.model,'C':C,'c':c,'layers':records,'codec_bytes':4*L*H*128*C,'no_QA_used':True,'capacity_frozen_from_run0':True});done.append(str(folder.relative_to(ROOT)))
  save(out/'completion.json',{'completed':True,'packages':done,'elapsed_s':time.time()-start,'GPU_peak_MiB':torch.cuda.max_memory_allocated()/2**20})
 except BaseException as e:save(out/'failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
