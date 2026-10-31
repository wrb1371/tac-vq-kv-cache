"""Legal NOVA g8/C256 K1/V2 budget point; same moments with stratified column regrouping."""
import argparse,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from r14kv.common import sha,json_save
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    import torch
    from r14kv.fitting import lloyd_group
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    parent=ROOT/'artifacts/round14/fitted/round14_NOVA_M0_fit_v1';pm=json.loads((parent/'completion.json').read_text());assert pm['completed'] and pm['full36_layer_artifact']
    cal=ROOT/pm['calibration'];cm=json.loads((cal/'completion.json').read_text());assert sha(cal/'completion.json')==pm['calibration_completion_SHA256']
    for n,r in cm['artifacts'].items():assert sha(cal/n)==r['SHA256']
    dest=ROOT/'artifacts/round14/fitted'/a.run_id;dest.mkdir(parents=True,exist_ok=False)
    out=ROOT/'runs'/a.run_id;out.mkdir(exist_ok=True,parents=True);begin=time.time();records=[]
    code={str(p.relative_to(ROOT)):sha(p) for p in [pathlib.Path(__file__),ROOT/'src/r14kv/fitting.py',ROOT/'src/r14kv/common.py']}
    for name,h in code.items():
        path=ROOT/'artifacts/round14/source_snapshots'/(h+'.py');path.parent.mkdir(exist_ok=True,parents=True)
        if not path.exists():path.write_bytes((ROOT/name).read_bytes())
    manifest={'method':'nova','completed':False,'calibration':pm['calibration'],'calibration_completion_SHA256':sha(cal/'completion.json'),'parent_completion_SHA256':sha(parent/'completion.json'),'full36_layer_artifact':True,'pilot_calibration':False,'budget_control_only':True,'K_bits_per_coordinate':1,'V_bits_per_coordinate':2,'centroids_per_group':256,'K_group_dimension':8,'metadata':'FP32 as M0, band64/256 unchanged','sample_policy':'identical131072 sample IDs and random-distinct seed/opportunities as M0; g8C256 uses the same131072 original sample IDs and25 iterations; larger paid books initialized from256 random distinct rows','code_SHA256':code,'started_epoch':begin}
    json_save(out/'run_manifest.json',manifest)
    for l in range(36):
        pool=np.memmap(cal/f'layer{l:02}_vectors.u16',mode='r',dtype=np.uint16,shape=(3,2,cm['capacity'],128))
        for h in range(2):
            seed=14261021+l*2+h;f=parent/f'layer{l:02}_head{h}.pt';rec=next(x for x in pm['records'] if x['layer']==l and x['head']==h);assert sha(f)==rec['SHA256']
            table=torch.load(f,map_location='cpu',weights_only=True);master=torch.load(parent/'offline_masters'/f.name,map_location='cpu',weights_only=True)
            assert all(torch.equal(master[key].bfloat16(),table[key]) for key in ['F','A','mean'])
            oldperm=[g+j*32 for g in range(32) for j in range(4)];newperm=[g+j*16 for g in range(16) for j in range(8)]
            inverse={rank:i for i,rank in enumerate(oldperm)};relative=torch.tensor([inverse[i] for i in newperm])
            forward=master['F'][:,relative];query_inverse=master['A'][:,relative]
            assert torch.allclose(forward@query_inverse.T,torch.eye(128,dtype=torch.float64),atol=1e-8,rtol=1e-8)
            table['F']=forward.bfloat16();table['A']=query_inverse.bfloat16()
            x=torch.from_numpy(pool[1,h,:cm['actual_reservoir_positions']].copy()).view(torch.bfloat16).cuda().float()
            gen=torch.Generator(device='cuda').manual_seed(seed+140000);ix=torch.randperm(len(x),device='cuda',generator=gen)[:131072]
            r=(x[ix].double()-master['mean'].cuda())@forward.cuda();r=r/r.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
            groups=r.reshape(-1,16,8).permute(1,0,2).float().contiguous();cent=lloyd_group(groups,seed=seed,iterations=25,clusters=256)
            scale=cent.abs().amax((-2,-1),keepdim=True).clamp_min(1e-12)/57344;trained=((cent/scale).to(torch.float8_e5m2).float()*scale).half();table['centroids']=trained.to(torch.float8_e5m2).cpu()
            path=dest/f.name;torch.save(table,path);records.append({'layer':l,'head':h,'SHA256':sha(path),'parent_SHA256':sha(f),'parent_offline_master_SHA256':sha(parent/'offline_masters'/f.name),'fit_samples':len(ix),'centroids_per_group':256,'K_group_dimension':8,'iterations':25})
            json_save(dest/'progress.json',records);print(json.dumps({'fit_units':len(records),'seconds':time.time()-begin}),flush=True)
            del x,r,groups,cent;torch.cuda.empty_cache()
    assert all(sha(ROOT/n)==h for n,h in code.items())
    complete={**manifest,'completed':True,'records':records,'finished_epoch':time.time(),'elapsed_seconds':time.time()-begin};json_save(dest/'completion.json',complete);json_save(out/'completion.json',complete)
if __name__=='__main__':main()
