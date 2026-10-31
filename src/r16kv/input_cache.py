"""Exact native processor tensors, inherited cache reuse, bounded new disk cache."""
import fcntl,hashlib,json,os,pathlib,time,torch
from r14kv.common import ROOT,json_save,sha
def media_path(item):
    return pathlib.Path(item['media_path']) if item.get('media_path') else ROOT/'assets/round15/WorldSense/videos'/item['video']
def trim(base,keep,limit=24*2**30):
    lock=(base/'.eviction.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    files=sorted(base.glob('*/media.pt'),key=lambda p:p.stat().st_mtime);total=sum(p.stat().st_size for p in files)
    for p in files:
        if total<=limit:break
        if p.parent==keep:continue
        guard=(p.parent/'.lock').open('a')
        try:
            fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);size=p.stat().st_size;p.unlink();total-=size
        except (BlockingIOError,FileNotFoundError):pass
        finally:guard.close()
    lock.close()
def processed_inputs(processor,item):
    key=hashlib.sha256(item['prompt'].encode()).hexdigest();native=item.get('dataset','WorldSense')=='WorldSense'
    if native:
        old=ROOT/'artifacts/round15/processor_inputs'/pathlib.Path(item['video']).stem
        if (old/(key+'.pt')).exists() and (old/'media.pt').exists():
            shared=torch.load(old/'media.pt',map_location='cpu',weights_only=True,mmap=True);specific=torch.load(old/(key+'.pt'),map_location='cpu',weights_only=True,mmap=True);return {**shared,**specific}
    base=ROOT/'artifacts/round16/processor_inputs';base.mkdir(exist_ok=True,parents=True)
    folder=base/(item.get('dataset','WorldSense')+'__'+pathlib.Path(item['video']).stem);folder.mkdir(exist_ok=True)
    lock=(folder/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX);media=folder/'media.pt';question=folder/(key+'.pt')
    if not media.exists() or not question.exists():
        from inference.inputs import build_inputs_cpu,build_conversation
        x=build_inputs_cpu(processor,build_conversation(str(media_path(item)),item['prompt']));shared={k:v for k,v in x.items() if k not in ['input_ids','attention_mask']};specific={k:v for k,v in x.items() if k in ['input_ids','attention_mask']}
        if media.exists():
            old=torch.load(media,map_location='cpu',weights_only=True,mmap=True);assert old.keys()==shared.keys()
            for k,v in shared.items():assert torch.equal(v,old[k]) if torch.is_tensor(v) else v==old[k]
        else:
            temp=media.with_suffix('.tmp');torch.save(shared,temp);temp.replace(media)
            json_save(folder/'media_identity.json',{'source':str(media_path(item)),'media_tensor_file_SHA256':sha(media),'bytes':media.stat().st_size,'recreatable_CPU_processor_only':True,'fields':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} if torch.is_tensor(v) else v for k,v in shared.items()}})
        if not question.exists():temp=question.with_suffix('.tmp');torch.save(specific,temp);temp.replace(question)
    specific=torch.load(question,map_location='cpu',weights_only=True,mmap=True);shared=torch.load(media,map_location='cpu',weights_only=True,mmap=True);os.utime(media,None);lock.close();trim(base,folder)
    return {**shared,**specific}
