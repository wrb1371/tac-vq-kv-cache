"""Independent compact normalized RVQ; only raw historical CAL vectors/gradients."""
import argparse,hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from r14kv.common import sha,json_save
def main():
    import torch
    from r14kv.fitting import nearest
    p=argparse.ArgumentParser();p.add_argument('--groups',type=int,choices=[1,4],required=True);p.add_argument('--run-id',required=True);a=p.parse_args()
    import os
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(22*2**30/torch.cuda.get_device_properties(0).total_memory)
    start=time.time();cal=ROOT/'artifacts/round14/calibration/round14_CAL64_capture_v1';cm=json.loads((cal/'completion.json').read_text());cap=cm['capacity'];n=cm['actual_reservoir_positions']
    run=ROOT/'data/runs'/a.run_id;run.mkdir(exist_ok=True,parents=True);dest=ROOT/'artifacts/ar17/packages'/f'CRVQ_G{a.groups}';dest.mkdir(parents=True,exist_ok=False)
    ids=np.memmap(cal/'source_and_token.i32',mode='r',dtype=np.int32,shape=(cap,2));gen=np.random.default_rng(17261023);tr=gen.choice(np.flatnonzero(ids[:n,0]%4!=0),1024,replace=False);va=gen.choice(np.flatnonzero(ids[:n,0]%4==0),256,replace=False)
    assert not set(ids[tr,0])&set(ids[va,0]);json_save(run/'slots.json',{'training':tr.tolist(),'validation':va.tolist(),'train_source_indices':sorted(set(map(int,ids[tr,0]))),'validation_source_indices':sorted(set(map(int,ids[va,0]))),'source_table_SHA256':sha(cal/'source_and_token.i32')})
    manifest={'run_id':a.run_id,'source_SHA256':sha(__file__),'groups':a.groups,'CAL_completion_SHA256':sha(cal/'completion.json'),'started_epoch':start,'slots_SHA256':sha(run/'slots.json'),'independent_codebook_initialization':True,'no_parent_centroids':True,'iterations':20,'stages':12,'seed':17261023,'training_label_scope':'historical CAL gradient weights only','independent_confirmation':False};json_save(run/'run_manifest.json',manifest)
    raw={k:{'train':[],'val':[],'weights':[]} for k in ['K','V']};table={};inputs=[]
    for l in range(36):
        f=cal/f'layer{l:02}_vectors.u16';g=cal/f'layer{l:02}_gradient_norms.f32'
        assert sha(f)==cm['artifacts'][f.name]['SHA256'] and sha(g)==cm['artifacts'][g.name]['SHA256']
        pool=np.memmap(f,mode='r',dtype=np.uint16,shape=(3,2,cap,128));grad=np.memmap(g,mode='r',dtype=np.float32,shape=(2,2,cap))
        for kind,ki,gi in [('K',0,0),('V',2,1)]:
            tx=torch.from_numpy(np.array(pool[ki][:,tr])).view(torch.bfloat16).cuda().float();vx=torch.from_numpy(np.array(pool[ki][:,va])).view(torch.bfloat16).cuda().float();w=torch.from_numpy(np.array(grad[gi][:,tr])).cuda().float();w=torch.log1p(w/(w.median(-1,keepdim=True).values+1e-12));w=w.clamp_min(1e-8)
            raw[kind]['train'].append(tx);raw[kind]['val'].append(vx);raw[kind]['weights'].append(w)
        inputs.append({'layer':l,'vectors_SHA256':cm['artifacts'][f.name]['SHA256'],'gradients_SHA256':cm['artifacts'][g.name]['SHA256']})
    traces={}
    for side,kind in enumerate(['K','V']):
        tx=torch.stack(raw[kind]['train']);vx=torch.stack(raw[kind]['val']);w=torch.stack(raw[kind]['weights']);mean=tx.mean(-2).bfloat16();scale=(tx-mean.float()[...,None,:]).square().mean(-2).sqrt().clamp_min(1e-4).bfloat16();table[kind+'_mean']=mean.cpu();table[kind+'_scale']=scale.cpu()
        def normalize(x):
            z=(x-mean.float()[...,None,:])/scale.float()[...,None,:];norm=z.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6).bfloat16();return z/norm.float(),norm
        z,tn=normalize(tx);v,vn=normalize(vx);fit=z.clone();valid=v.clone();books=[];trace=[]
        for group in range(a.groups):
            sl=slice(group*(36//a.groups),(group+1)*(36//a.groups));x=fit[sl].reshape(1,-1,128);y=valid[sl].reshape(1,-1,128);ww=w[sl].reshape(1,-1);cb=[];hist=[]
            for stage in range(12):
                rng=torch.Generator(device='cuda').manual_seed(17261023+10000*side+100*group+stage);ix=torch.randperm(x.shape[1],generator=rng,device='cuda')[:256];c=x[:,ix].clone()
                for _ in range(20):
                    ix=nearest(x,c);sums=torch.zeros_like(c);count=torch.zeros(1,256,device='cuda');sums.scatter_add_(1,ix[...,None].expand(-1,-1,128),x*ww[...,None]);count.scatter_add_(1,ix,ww);c=torch.where((count>0)[...,None],sums/count.clamp_min(1e-20)[...,None],c)
                deployed=c.half();assert torch.isfinite(deployed).all();c=deployed.float();ix=nearest(x,c);x-=torch.gather(c,1,ix[...,None].expand(-1,-1,128));iv=nearest(y,c);y-=torch.gather(c,1,iv[...,None].expand(-1,-1,128));cb.append(deployed[0].cpu())
                train_error=(x.reshape_as(fit[sl])*tn[sl].float()*scale[sl].float()[...,None,:]).square().sum()/tx[sl].square().sum();val_error=(y.reshape_as(valid[sl])*vn[sl].float()*scale[sl].float()[...,None,:]).square().sum()/vx[sl].square().sum()
                row={'stage':stage+1,'train_raw_relative_squared_error':float(train_error),'validation_raw_relative_squared_error':float(val_error),'train_weighted_normalized_MSE':float((x.square().sum(-1)*ww).sum()/ww.sum()),'occupied_centroids':int(ix.unique().numel())};hist.append(row);print(json.dumps({'side':kind,'group':group,**row,'elapsed_s':time.time()-start}),flush=True)
            books.append(torch.stack(cb));trace.append({'group':group,'history':hist})
        table[kind+'_books']=torch.stack(books).contiguous();traces[kind]=trace;json_save(run/'training_trace.json',traces)
        del tx,vx,z,v,fit,valid,x,y,w;torch.cuda.empty_cache()
    f=dest/'global.pt';torch.save(table,f);record={'layer':0,'file':f.name,'SHA256':sha(f),'tensors':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'storage_bytes':v.untyped_storage().nbytes()} for k,v in table.items()}}
    completion={**manifest,'completed':True,'family':'CRVQ','CPU_export':True,'groups':a.groups,'Rmax_K':12,'Rmax_V':12,'records':[record],'static_tensor_bytes':sum(v.untyped_storage().nbytes() for v in table.values()),'finished_epoch':time.time(),'elapsed_s':time.time()-start,'input_file_identities':inputs,'source_disjoint_CAL_diagnostic_only':True,'task_quality_evaluated':False};json_save(dest/'completion.json',completion);json_save(run/'completion.json',completion);print(json.dumps({'completed':True,'fixed_MiB':completion['static_tensor_bytes']/2**20,'seconds':time.time()-start}),flush=True)
if __name__=='__main__':main()
