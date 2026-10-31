"""Shared input/runtime and immutable evidence helpers; no quantizer semantics."""
import hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[2]
AUTHOR=ROOT/'third_party/OmniKVQuant_snapshot'
sys.path[:0]=[str(ROOT/'src'),str(AUTHOR)]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def json_save(path,value):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)
def runtime():
    import torch
    from transformers import Qwen2_5OmniConfig
    from model.modeling_qwen2_5_omni import load_backbone
    from inference.inputs import seed_everything
    torch.set_num_threads(4);seed_everything(42)
    model_path=ROOT/'assets/round13/models/Qwen2.5-Omni-3B'
    cfg=Qwen2_5OmniConfig.from_pretrained(model_path,local_files_only=True);c=cfg.thinker_config.text_config
    assert (c.num_hidden_layers,c.num_attention_heads,c.num_key_value_heads,c.hidden_size//c.num_attention_heads)==(36,16,2,128)
    assert c.rope_scaling['mrope_section']==[16,24,24]
    model,processor=load_backbone(str(model_path),config=cfg,attn_implementation='flash_attention_2')
    model.requires_grad_(False)
    params=sum(p.numel() for p in model.parameters());assert params==4703464448 and params<=10_000_000_000
    from r13kv.media_cache import install_media_cache
    media_stats=install_media_cache()
    metadata={'model_revision':'f75b40e3da2003cdd6e1829b1f420ca70797c34e','actual_loaded_parameters':params,
        'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'geometry':{'layers':36,'query_heads':16,'physical_KV_heads':2,'head_dim':128,'MRoPE_sections':[16,24,24]},
        'torch':torch.__version__,'transformers':__import__('transformers').__version__,'GPU':torch.cuda.get_device_name(),'weights_frozen':True,'seed':42}
    return model,processor,metadata,media_stats
def input_for(model,processor,item):
    from inference.inputs import build_conversation,build_inputs_cpu,move_inputs_to_model
    t=time.perf_counter();cpu=build_inputs_cpu(processor,build_conversation(str(ROOT/'assets/round13/WorldSense/videos'/item['video']),item['prompt']))
    meta={'input_ids_SHA256':hashlib.sha256(cpu['input_ids'].numpy().tobytes()).hexdigest(),'prompt_tokens':cpu['input_ids'].shape[1],
        'prepare_seconds':time.perf_counter()-t,'video_grid_thw':cpu.get('video_grid_thw').tolist() if cpu.get('video_grid_thw') is not None else None}
    return move_inputs_to_model(cpu,model),meta
def prefill(model,inputs,cache):
    from model.modeling_qwen2_5_omni import prefill_mcq_prefix
    from r13kv.execution import final_prefix_projection
    with final_prefix_projection(model):return prefill_mcq_prefix(model,inputs,cache)
def last_inputs(model,inputs,cache):
    import torch
    x=dict(inputs);ids=x.pop('input_ids')
    x.update(past_key_values=cache,cache_position=torch.tensor([ids.shape[1]-1],device=ids.device),use_audio_in_video=True,use_cache=True)
    prepared=model.thinker.prepare_inputs_for_generation(ids,**x)
    assert prepared['input_ids'].shape==(1,1) and torch.equal(prepared['input_ids'],ids[:,-1:])
    return prepared
def selected_rope(cos,sin,sections=(16,24,24)):
    import torch
    widths=list(sections)*2
    return (torch.cat([x[i%3] for i,x in enumerate(cos.split(widths,dim=-1))],dim=-1).unsqueeze(1),
            torch.cat([x[i%3] for i,x in enumerate(sin.split(widths,dim=-1))],dim=-1).unsqueeze(1))
def apply_key_rope(x,cos,sin):
    import torch
    return x*cos+torch.cat((-x[...,64:],x[...,:64]),dim=-1)*sin
