"""Exact CPU processor tensors, shared by all methods; no KV/model-output cache."""
import fcntl,hashlib,json,pathlib,torch
from r14kv.common import ROOT,sha,json_save
def processed_inputs(processor,item):
    base=ROOT/'artifacts/round15/processor_inputs';base.mkdir(parents=True,exist_ok=True)
    source=base/item['video'].removesuffix('.mp4');source.mkdir(exist_ok=True)
    prompt_sha=hashlib.sha256(item['prompt'].encode()).hexdigest();question=source/(prompt_sha+'.pt');media=source/'media.pt'
    lock=open(source/'.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
    if not question.exists():
        from inference.inputs import build_inputs_cpu,build_conversation
        inputs=build_inputs_cpu(processor,build_conversation(str(ROOT/'assets/round15/WorldSense/videos'/item['video']),item['prompt']))
        shared={k:v for k,v in inputs.items() if k not in ['input_ids','attention_mask']};specific={k:v for k,v in inputs.items() if k in ['input_ids','attention_mask']}
        if not media.exists():
            tmp=media.with_suffix('.tmp');torch.save(shared,tmp);tmp.replace(media)
            json_save(source/'media_identity.json',{'source':item['video'],'media_tensor_file_SHA256':sha(media),'media_tensor_file_bytes':media.stat().st_size,'scope':'exact unmodified CPU processor tensors, never KV or model activations/outputs','processor_contract':'inherited R14 build_inputs_cpu; default sampling; complete AV','fields':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} if torch.is_tensor(v) else v for k,v in shared.items()}})
        else:
            old=torch.load(media,map_location='cpu',weights_only=True,mmap=True);assert shared.keys()==old.keys()
            for k,v in shared.items():assert torch.equal(v,old[k]) if torch.is_tensor(v) else v==old[k]
        tmp=question.with_suffix('.tmp');torch.save(specific,tmp);tmp.replace(question)
    specific=torch.load(question,map_location='cpu',weights_only=True,mmap=True);shared=torch.load(media,map_location='cpu',weights_only=True,mmap=True)
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close();return {**shared,**specific}
