"""Deterministic native visual K/V and neutral-readout Fisher capture, no QA labels."""
import argparse,gc,hashlib,json,os,pathlib,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ar37cq.runtime import load,Supplier,Teacher,fisher_readout
from ar26kv.runtime import save

def main():
    import torch
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['Omni','VL'],required=True);ap.add_argument('--run-id',required=True);ap.add_argument('--limit',type=int,default=24);a=ap.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')in['0','2','3','4','6','7']
    start=time.time();out=ROOT/'data/runs'/a.run_id;out.mkdir(parents=True,exist_ok=False);rows=json.loads((ROOT/'manifests/ar22/MEDIA_CAL24.json').read_text())[:a.limit];assert all('answer'not in r and 'question'not in r and r['prompt']=='Describe the video.'for r in rows)
    source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [pathlib.Path(__file__),*sorted((ROOT/'src/ar37cq').glob('*.py')),ROOT/'configs/ar37/protocol.json']}
    save(out/'run_manifest.json',{'run_id':a.run_id,'model':a.model,'source_SHA256':source,'inputs_SHA256':hashlib.sha256((ROOT/'manifests/ar22/MEDIA_CAL24.json').read_bytes()).hexdigest(),'GPU':os.environ['CUDA_VISIBLE_DEVICES'],'limit':a.limit,'start_epoch':start})
    # Regenerable media-feature cache; source frames and original receipts stay intact.
    import ar22kv.inputs as media_inputs
    media_inputs.CACHE_LIMIT=256*2**20
    try:
        model,p=load(a.model);versions=[v._version for v in model.parameters()];supplier=Supplier(model,p,a.model);teacher=Teacher(model,a.model);records=[]
        for item in rows:
            x,pos,vi,ti,end,identity=supplier.get(item,verify=True);teacher.capture(x,pos,end,identity)
            samples,record=fisher_readout(model,p,a.model,teacher,x,pos,end,vi)
            path=ROOT/'artifacts/ar38/CQ_calibration'/a.model/(item['id']+'.pt');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();torch.save(samples,path)
            records.append({'id':item['id'],'source_id':item['source_id'],'duration_group':item['duration_group'],'identity':identity,'path':str(path.relative_to(ROOT)),'SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,**record});save(out/'progress.json',{'rows':records,'elapsed_s':time.time()-start});print(json.dumps({k:records[-1][k]for k in ['id','loss','target_text','GPU_peak_MiB']}),flush=True)
            del x,pos,vi,ti,samples;teacher.clear();gc.collect();torch.cuda.empty_cache()
        assert versions==[v._version for v in model.parameters()];teacher.close();save(out/'completion.json',{'completed':True,'model':a.model,'rows':records,'elapsed_s':time.time()-start,'no_test_questions_answers_used':True,'model_parameters_unchanged':True,'GPU1_used':False,'native_reader_training':False,'source_SHA256':source})
    except BaseException as e:save(out/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'partial_capture_used':False});raise
if __name__=='__main__':main()
