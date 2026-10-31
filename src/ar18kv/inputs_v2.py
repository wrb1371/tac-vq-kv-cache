"""Native video-only/AV input processing, official temporal bounds and 64GiB recreatable DISK cache; model/codec mathematics unchanged."""
import fcntl, hashlib, json, os, pathlib, torch
from r14kv.common import ROOT, json_save, sha
from ar17kv.input_cache import trim
from inference.inputs import build_conversation, move_inputs_to_model

def load_items(path):
    items=json.loads(pathlib.Path(path).read_text());assert isinstance(items,list) and items
    if 'conversations' in items[0]:
        from inference.data import load_items as old
        return old(path)
    assert len({r['id'] for r in items})==len(items)
    for r in items:assert r['answer'] in [chr(65+i) for i in range(len(r['options']))]
    return items

def process(processor,item):
    use_audio=item.get('use_audio_in_video',True)
    if item.get('dataset','WorldSense')=='WorldSense':
        from ar17kv.input_cache import processed_inputs
        return processed_inputs(processor,item),{'media_feature_identity_scope':'inherited pinned WorldSense input pipeline'}
    base=ROOT/'artifacts/ar18/processor_inputs';base.mkdir(exist_ok=True,parents=True)
    source_key=hashlib.sha256(json.dumps({k:item.get(k) for k in ['media_path','use_audio_in_video','video_end']},sort_keys=True).encode()).hexdigest()
    folder=base/source_key;folder.mkdir(exist_ok=True);text_key=hashlib.sha256(item['prompt'].encode()).hexdigest();media=folder/'media.pt';specific=folder/(text_key+'.pt')
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not media.exists() or not specific.exists():
            from qwen_omni_utils import process_mm_info
            conv=build_conversation(item['media_path'],item['prompt'])
            if item.get('video_end') is not None:conv[1]['content'][0]['video_end']=item['video_end']
            text=processor.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
            audios,images,videos=process_mm_info(conv,use_audio_in_video=use_audio)
            inputs=dict(processor(text=text,audio=audios,images=images,videos=videos,return_tensors='pt',padding=True,use_audio_in_video=use_audio))
            common={k:v for k,v in inputs.items() if k not in ['input_ids','attention_mask']};question={k:v for k,v in inputs.items() if k in ['input_ids','attention_mask']}
            if not media.exists():
                tmp=media.with_suffix('.tmp');torch.save(common,tmp);tmp.replace(media)
                tensor_hashes={k:hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in common.items() if torch.is_tensor(v)}
                json_save(folder/'identity.json',{'native_tensor_SHA256':tensor_hashes,'source_id':item['source_id'],'media_path':item['media_path'],'use_audio_in_video':use_audio,'video_end':item.get('video_end'),'raw_media_SHA256':sha(item['media_path']),'media_file_SHA256':sha(media),'recreatable_CPU_processor_only':True,'bytes':media.stat().st_size,'fields':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in common.items() if torch.is_tensor(v)}})
            tmp=specific.with_suffix('.tmp');torch.save(question,tmp);tmp.replace(specific)
        common=torch.load(media,map_location='cpu',weights_only=True,mmap=True);question=torch.load(specific,map_location='cpu',weights_only=True,mmap=True);identity=json.loads((folder/'identity.json').read_text());os.utime(media,None)
    trim(base,folder,64*2**30)
    return {**common,**question},{'native_media_feature_SHA256':identity['native_tensor_SHA256'],'raw_media_SHA256':identity['raw_media_SHA256'],'official_video_end':item.get('video_end'),'use_audio_in_video':use_audio}

def input_for(model,processor,item):
    cpu,meta=process(processor,item)
    return move_inputs_to_model(cpu,model),{**meta,'input_sha256':hashlib.sha256(cpu['input_ids'].numpy().tobytes()).hexdigest(),'prompt_tokens':cpu['input_ids'].shape[1],'video_grid_thw':cpu['video_grid_thw'].tolist()}

def positions_for(model,inputs,use_audio):
    return model.thinker.get_rope_index(input_ids=inputs['input_ids'],video_grid_thw=inputs.get('video_grid_thw'),image_grid_thw=inputs.get('image_grid_thw'),attention_mask=inputs['attention_mask'],use_audio_in_video=use_audio,audio_seqlens=inputs['feature_attention_mask'].sum(-1) if 'feature_attention_mask' in inputs else None,second_per_grids=inputs.get('video_second_per_grid'))[0][:,0]
