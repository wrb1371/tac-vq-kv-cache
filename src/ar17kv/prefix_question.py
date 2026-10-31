"""Finite complete video is encoded/packed before question tokens read it."""
import torch
from r13kv.execution import final_prefix_projection
from model.modeling_qwen2_5_omni import _THINKER_PREFILL_KEYS
def split_index(tags,inputs):
    nontext=torch.nonzero(tags!=0).flatten();assert nontext.numel()>0
    split=int(nontext[-1])+1;assert split<inputs['input_ids'].shape[1]-1
    assert (tags[split:]==0).all()
    return split
def media_prefill(model,inputs,cache,positions,split):
    kw={k:inputs[k] for k in _THINKER_PREFILL_KEYS if k in inputs};kw.update(input_ids=inputs['input_ids'][:,:split],attention_mask=inputs['attention_mask'][:,:split],position_ids=positions[:,:split][:,None],past_key_values=cache,use_cache=True,return_dict=True,use_audio_in_video=True,cache_position=torch.arange(split,device=inputs['input_ids'].device))
    with final_prefix_projection(model):model.thinker(**kw)
def question_forward(model,inputs,cache,positions,split):
    n=inputs['input_ids'].shape[1]
    with final_prefix_projection(model):
        result=model.thinker(input_ids=inputs['input_ids'][:,split:],attention_mask=inputs['attention_mask'],position_ids=positions[:,split:][:,None],past_key_values=cache,use_cache=True,return_dict=True,use_audio_in_video=True,cache_position=torch.arange(split,n,device=inputs['input_ids'].device))
    return result.logits[:,-1]
