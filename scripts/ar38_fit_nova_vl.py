"""Native28x4 NOVA fits from full attention moments and bounded real keys."""
import argparse,hashlib,json,os,pathlib,sys,time,traceback
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from ar26kv.runtime import save
from r14kv.fitting import qpca,value_basis,lloyd_group,nearest
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);ap.add_argument('--replicate',type=int,choices=[1,2],required=True);ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,default=6);a=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')in['0','2','3','4','6','7'];torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(26*2**30/torch.cuda.get_device_properties(0).total_memory)
    started=time.time();out=ROOT/'data/runs'/a.run_id;out.mkdir(exist_ok=False,parents=True)
    cfg=json.loads((ROOT/'configs/ar30/protocol.json').read_text());cal=json.loads((ROOT/'data/runs/AR30_NOVA_CAL24_v1/completion.json').read_text());assert cal['completed']and cal['media']==24 and not cal['calibration_labels_used']
    folder=ROOT/cal['folder'];assert sha(folder/'moments.pt')==cal['artifacts']['moments.pt']['SHA256'];mom=torch.load(folder/'moments.pt',map_location='cpu',weights_only=True)
    cfg['NOVA_seed_base']+=a.replicate*1000003;pairs=[(8,16),(8,256)];layers=list(range(a.shard,28,a.shards));count=mom['prefix_token_count'];records=[]
    protected=[pathlib.Path(__file__),ROOT/'src/r14kv/fitting.py',ROOT/'configs/ar30/protocol.json',ROOT/'data/runs/AR30_NOVA_CAL24_v1/completion.json',folder/'moments.pt']
    protected +=[folder/f'layer{l:02}_Kpost.u16'for l in layers]
    hashes={str(p.relative_to(ROOT)):sha(p)for p in protected};save(out/'run_manifest.json',{'GPU':os.environ['CUDA_VISIBLE_DEVICES'],'source_SHA256':hashes,'layers':layers,'labels_used':False,'pairs':pairs})
    try:
        for l in layers:
            pool=np.memmap(folder/f'layer{l:02}_Kpost.u16',mode='r',dtype=np.uint16,shape=(4,cal['capacity'],128))
            books={p:{k:[]for k in ['F','A','mean','V_rotation','centroids']}for p in pairs}
            for h in range(4):
                f0,a0,mean,diag=qpca(mom['q2'][l,h],mom['k2'][l,h],mom['ksum'][l,h],count)
                assert diag['FP64_identity_max_abs']<1e-6,diag
                vr=value_basis(mom['o2'][l,h]/count).bfloat16();seed=cfg['NOVA_seed_base']+4*l+h
                x=torch.from_numpy(pool[h,:cal['actual_reservoir_positions']].copy()).view(torch.bfloat16).cuda().float()
                gen=torch.Generator(device='cuda').manual_seed(seed+140000);ix=torch.randperm(len(x),generator=gen,device='cuda')[:cfg['NOVA_fit_samples_cap']];x=x[ix]
                for g in sorted({p[0]for p in pairs}):
                    groups=128//g;old=[i+j*32 for i in range(32)for j in range(4)];inv={r:i for i,r in enumerate(old)}
                    permutation=torch.tensor([inv[i+j*groups]for i in range(groups)for j in range(g)])
                    f=f0[:,permutation];a_inv=a0[:,permutation]
                    z=(x.double()-mean.cuda())@f.cuda();z/=z.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
                    grouped=z.reshape(-1,groups,g).permute(1,0,2).float().contiguous();del z
                    for gg,nc in pairs:
                        if gg!=g:continue
                        cent=lloyd_group(grouped,seed=seed,iterations=cfg['NOVA_kmeans_iterations'],clusters=nc)
                        scale=cent.abs().amax((-2,-1),keepdim=True).clamp_min(1e-12)/57344.
                        trainer=((cent/scale).to(torch.float8_e5m2).float()*scale).half();deployed=trainer.to(torch.float8_e5m2)
                        assert torch.isfinite(deployed.float()).all()
                        chosen=nearest(grouped,deployed.float());recon=torch.gather(deployed.float(),1,chosen[...,None].expand(-1,-1,g))
                        mse=float((grouped-recon).square().mean())
                        target=books[(g,nc)]
                        for k,v in {'F':f.bfloat16(),'A':a_inv.bfloat16(),'mean':mean.bfloat16(),'V_rotation':vr,'centroids':deployed}.items():target[k].append(v.cpu())
                        row={'layer':l,'head':h,'group':g,'centroids':nc,'samples':len(x),'iterations':cfg['NOVA_kmeans_iterations'],
                             'deployed_group_MSE':mse,'QPCA':diag,'elapsed_s':time.time()-started};records.append(row);print(json.dumps(row),flush=True)
                        del cent,trainer,deployed,chosen,recon;torch.cuda.empty_cache()
                    del grouped
                del x;torch.cuda.empty_cache();save(out/'progress.json',{'completed_units':len(records),'records':records,'elapsed_s':time.time()-started})
            for(g,nc),tables in books.items():
                package=ROOT/f'artifacts/ar38/replicas/r{a.replicate}/artifacts/ar30/packages/NOVA_g{g}_C{nc}';package.mkdir(exist_ok=True,parents=True)
                path=package/f'layer{l:02}.pt';assert not path.exists();torch.save({k:torch.stack(v).contiguous().clone()for k,v in tables.items()},path)
        for g,nc in pairs:
            dest=ROOT/f'artifacts/ar38/replicas/r{a.replicate}/artifacts/ar30/packages/NOVA_g{g}_C{nc}'
            meta=json.loads((ROOT/f'artifacts/ar30/packages/NOVA_g{g}_C{nc}/completion.json').read_text());meta.update(replicate=a.replicate,seed_offset=a.replicate*1000003,records=[{'layer':l,'file':f'layer{l:02}.pt','SHA256':sha(dest/f'layer{l:02}.pt')}for l in layers]);save(dest/'completion.json',meta)
        assert all(sha(ROOT/p)==v for p,v in hashes.items())
        save(out/'completion.json',{'completed':True,'GPU':os.environ['CUDA_VISIBLE_DEVICES'],'layers':layers,'pairs':pairs,
             'source_SHA256':hashes,'labels_used':False,'records':records,'package_root':f'artifacts/ar30/fit_parts/{a.run_id}',
             'elapsed_s':time.time()-started,'GPU_peak_MiB':torch.cuda.max_memory_allocated()/2**20})
    except BaseException as e:
        save(out/'failure.json',{'completed':False,'error':repr(e),'traceback':traceback.format_exc(),'units':len(records),'elapsed_s':time.time()-started});raise
if __name__=='__main__':main()
