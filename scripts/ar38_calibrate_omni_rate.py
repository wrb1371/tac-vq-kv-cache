"""Measure native media-only curves; no question answer or new bank fitting."""
import hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'third_party/OmniKVQuant_snapshot')]
import torch
from r14kv.common import runtime,json_save,sha
from ar19kv.inputs import input_for,positions_for
from ar18kv.prefix import split_index,media_prefill
from ar20kv.attention import KernelCache,install_reference_attention
from ar21kv.calibration import RateCurveCapture
from r15kv.budget import RequestMeta

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--replicate',type=int,choices=[1,2],required=True);a=p.parse_args();base=ROOT/f'artifacts/ar38/replicas/r{a.replicate}'
    from model.codec.modality import modality_tags
    from safetensors import safe_open
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(22*2**30/torch.cuda.get_device_properties(0).total_memory);start=time.time();rid=f'AR38_Omni_RATE_r{a.replicate}_v1';dest=ROOT/'data/runs'/rid;dest.mkdir(parents=True,exist_ok=True);inputs_file=ROOT/'manifests/ar21/CAL16.json';items=json.loads(inputs_file.read_text());assert all('answer' not in x for x in items)
    sources=[pathlib.Path(__file__),ROOT/'src/ar21kv/calibration.py',ROOT/'src/ar21kv/planner.py',ROOT/'src/ar21kv/allocated.py'];source={str(p.relative_to(ROOT)):sha(p) for p in sources};manifest={'run_id':rid,'started_epoch':start,'CAL_manifest_SHA256':sha(inputs_file),'source_SHA256':source,'scope':'native media prefix, no QA scoring, original fixed bank, output/query-energy weighted rate costs'};json_save(dest/'run_manifest.json',manifest)
    model,processor,meta,_=runtime();install_reference_attention(model,'gqa');versions={n:p._version for n,p in model.named_parameters()};codec=RateCurveCapture(ROOT/'artifacts/ar17/packages/PSRVQ_d16',16,mode='sparse',gop=8,keep_fraction=.125)
    for l,layer in enumerate(model.thinker.model.layers):
        w=layer.self_attn.o_proj.weight.float().reshape(2048,2,8,128);codec.output_diagonals[l]=w.square().sum((0,2)).detach()
        def capture(module,args,out,index=l):
            q=out.detach().view(1,-1,2,8,128);n=q.shape[1];take=torch.linspace(0,n-1,min(128,n),device=q.device).long();codec.query_moments[index]=q[:,take].float().square().mean((0,1,3))
        layer.self_attn.q_proj.register_forward_hook(capture)
    all_records=[]
    with (dest/'curves_by_source.jsonl').open('x') as out:
        for i,item in enumerate(items):
            codec.curve_records=[];inputs,imeta=input_for(model,processor,item);positions=positions_for(model,inputs,False);tags=modality_tags(inputs['input_ids'],model);end=split_index(tags,inputs);meta=RequestMeta(inputs['input_ids'].shape[1],1,tuple(positions.amax(1).tolist()),imeta['input_sha256']);codec.set_context(tags=tags,positions=positions,input_ids=inputs['input_ids'],item_id=item['id'],video_grid_thw=inputs['video_grid_thw'],generation_cap=1);rotary=model.thinker.model.rotary_emb;cache=KernelCache(codec,rotary.inv_freq,meta,rotary.attention_scaling)
            with torch.inference_mode():media_prefill(model,inputs,cache,positions,end,False)
            torch.cuda.synchronize();record={'id':item['id'],'source_id':item['source_id'],'dataset':item['dataset'],'input_identity':imeta,'media_end':end,'hybrid_dense':codec._hybrid_dense,'curves':codec.curve_records};all_records.append(record);out.write(json.dumps(record,ensure_ascii=False)+'\n');out.flush();print(json.dumps({'index':i,'source':item['source_id'],'curves':len(codec.curve_records),'hybrid_dense':codec._hybrid_dense}),flush=True);codec.finish_request();del cache,inputs;torch.cuda.empty_cache()
    curves={}
    for objective in ['plain','fanout']:
        matrices={k:torch.zeros(36,7,dtype=torch.float64) for k in ['K','V']};counts={k:torch.zeros(36,1,dtype=torch.float64) for k in ['K','V']}
        for sample in all_records:
            for row in sample['curves']:
                weight=row['population_slots_per_head']*(row['fanout_factor'] if objective=='fanout' else 1.)/max(1,sample['media_end']);matrices[row['kind']][row['layer']]+=torch.tensor(row['plain_loss'],dtype=torch.float64)*weight;counts[row['kind']][row['layer']]+=weight
        for k in matrices:
            assert (counts[k]>0).all();matrices[k]/=counts[k];matrices[k]/=matrices[k][:,1].sum().clamp_min(1e-12)
        curves[objective]={k:x.tolist() for k,x in matrices.items()}
    unchanged=versions=={n:p._version for n,p in model.named_parameters()};assert unchanged and all(sha(ROOT/p)==h for p,h in source.items());result={**manifest,**curves,'completed':True,'sources':len(all_records),'source_ids':[r['source_id'] for r in all_records],'fixed_bank_completion_SHA256':codec.package_sha256,'question_answers_used':False,'model_parameters_unchanged':unchanged,'finished_epoch':time.time(),'elapsed_seconds':time.time()-start,'model':meta.__dict__ if False else 'same frozen Qwen2.5-Omni revision as AR20'};json_save(base/'data/ar21/rate_curves.json',result);json_save(dest/'completion.json',result);print(json.dumps({'completed':True,'seconds':time.time()-start}))
if __name__=='__main__':
    import argparse
    args=argparse.ArgumentParser();args.add_argument('--replicate',type=int,required=True);a=args.parse_args()
    from ar38replica.paths import use
    with use(a.replicate,prefixes=['artifacts/ar17/packages/PSRVQ_d16']):main()
