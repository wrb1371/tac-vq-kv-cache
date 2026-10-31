"""Budget-admitted native7B private banks, same old calibration and GSKM recipe."""
import hashlib,json,os,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from ar26kv.runtime import save
from r14kv.fitting import fit_rq,gskm,logweights,nearest
from safetensors import safe_open
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')in['0','2','3','4','6','7']
    start=time.time();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(26*2**30/torch.cuda.get_device_properties(0).total_memory)
    out=ROOT/'data/runs/AR29_PRIVATE_FIT_v1';out.mkdir(exist_ok=False,parents=True)
    admission=json.loads((ROOT/'data/ar29/GSRQ_ADMISSION.json').read_text());tiers=sorted(set(admission['Rmax_by_group'].values()));maximum=max(tiers)
    source=ROOT/'data/runs/AR26_VL_MEDIA_CAL24_v1/completion.json';cal=json.loads(source.read_text());assert cal['completed']and len(cal['rows'])==24
    samples=[ROOT/r['sample_file']for r in cal['rows']]
    protected=[source,ROOT/'data/ar29/GSRQ_ADMISSION.json',ROOT/'configs/ar29/protocol.json',pathlib.Path(__file__),ROOT/'src/r14kv/fitting.py',ROOT/'src/r15kv/fitting.py',*samples]
    protected +=list((ROOT/'artifacts/ar26/packages').glob('GSRQ*/R5/*.pt'))
    hashes={str(p.relative_to(ROOT)):sha(p)for p in protected};save(out/'run_manifest.json',{'source_SHA256':hashes,'GPU':os.environ['CUDA_VISIBLE_DEVICES'],'labels_used':False,'maximum_stages':maximum,'tiers':tiers})
    pools=[torch.load(p,map_location='cpu',weights_only=True)for p in samples]
    model=ROOT/'artifacts/ar26/models/Qwen2.5-VL-7B-Instruct';index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    differences=[]
    for l in range(28):
        key=f'model.layers.{l}.self_attn.o_proj.weight'
        if key not in index:key=f'model.language_model.layers.{l}.self_attn.o_proj.weight'
        with safe_open(model/index[key],framework='pt')as f:w=f.get_tensor(key).float()
        metrics={'V':w.reshape(3584,4,7,128).square().sum((0,2)).cuda(),
                 'K':torch.stack([v[l]['q_moment']for v in pools]).mean(0).cuda()/128.}
        result={f:{}for f in ['GSRQExport','GSRQAware']}
        for si,side in enumerate(['K','V']):
            x=torch.cat([v[l][side]for v in pools],1).cuda().float();norm=(x.square()*metrics[side][:,None]).sum(-1).sqrt()
            seeds=[14261021+10000*si+l*10+h for h in range(4)]
            export,trace=fit_rq(x,norm,seeds,stages=maximum,iterations=40);result['GSRQExport'][side]=export.to(torch.float8_e4m3fn)
            olde=torch.load(ROOT/f'artifacts/ar26/packages/GSRQExport/R5/layer{l:02}.pt',map_location='cpu',weights_only=True)[side]
            current=result['GSRQExport'][side][:,:5];differences.append({'layer':l,'side':side,
                'FP8_prefix_different_elements':int((current.float()!=olde.float()).sum()),'FP8_prefix_elements':olde.numel(),
                'FP8_prefix_max_abs_difference':float((current.float()-olde.float()).abs().max())})
            save(out/f'export_layer{l:02}_{side}_trace.json',trace)
            old=torch.load(ROOT/f'artifacts/ar26/packages/GSRQAware/R5/layer{l:02}.pt',map_location='cpu',weights_only=True)[side]
            residual=x.clone();weights,diag=logweights(norm)
            for r in range(5):
                book=old[:,r].cuda().float();ids=nearest(residual,book);residual-=torch.gather(book,1,ids[...,None].expand(-1,-1,128))
            books=[old[:,r]for r in range(5)];history=[]
            for r in range(5,maximum):
                master,hist=gskm(residual,weights,[s+r for s in seeds],iterations=40)
                deployed=master.to(torch.float8_e4m3fn);assert torch.isfinite(deployed.float()).all()
                ids=nearest(residual,deployed.float());residual-=torch.gather(deployed.float(),1,ids[...,None].expand(-1,-1,128))
                books.append(deployed.cpu());history.append({'stage':r+1,'training':hist,'residual_MSE':residual.square().sum(-1).mean(-1).cpu().tolist()})
            result['GSRQAware'][side]=torch.stack(books,1);assert torch.equal(result['GSRQAware'][side][:,:5].view(torch.uint8),old.view(torch.uint8))
            save(out/f'aware_layer{l:02}_{side}_trace.json',{'weighting':diag,'exact_old5_prefix':True,'appended_stages':history})
            del x,norm,residual,weights,master,deployed,export;torch.cuda.empty_cache()
        for family in result:
            for r in tiers:
                folder=ROOT/f'artifacts/ar29/packages/{family}/R{r}';folder.mkdir(exist_ok=True,parents=True)
                if r<=5:bank=torch.load(ROOT/f'artifacts/ar26/packages/{family}/R5/layer{l:02}.pt',map_location='cpu',weights_only=True)
                else:bank=result[family]
                torch.save({s:t[:,:r].clone()for s,t in bank.items()},folder/f'layer{l:02}.pt')
        print(json.dumps({'layer_complete':l,'elapsed_s':time.time()-start}),flush=True)
    for family in ['GSRQExport','GSRQAware']:
        for r in tiers:
            folder=ROOT/f'artifacts/ar29/packages/{family}/R{r}'
            save(folder/'completion.json',{'completed':True,'model':'Qwen2.5-VL-7B-Instruct','family':family,'Rmax':r,'labels_used':False,
                'static_tensor_bytes':28*4*2*r*256*128,'same24calibration':True,'seeds_and_40_iterations_unchanged':True,
                'exact_old_prefix':r<=5 or family=='GSRQAware','records':[{'file':f'layer{l:02}.pt','SHA256':sha(folder/f'layer{l:02}.pt')}for l in range(28)]})
    assert all(sha(ROOT/p)==v for p,v in hashes.items())
    save(ROOT/'data/ar29/EXPORT_PREFIX_DIAGNOSTIC.json',{'completed':True,'records':differences,
         'different_elements':sum(r['FP8_prefix_different_elements']for r in differences),'total_elements':sum(r['FP8_prefix_elements']for r in differences),
         'scope':'higher-stage export bank refit recovers FP16 training residuals; exact old R3 is reused for the short tier'})
    save(out/'completion.json',{'completed':True,'elapsed_s':time.time()-start,'source_SHA256':hashes,'GPU':os.environ['CUDA_VISIBLE_DEVICES'],
         'GPU_peak_MiB':torch.cuda.max_memory_allocated()/2**20,'labels_used':False,'maximum_R':maximum,'banks_completed':6})
if __name__=='__main__':main()
