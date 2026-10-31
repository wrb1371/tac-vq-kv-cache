"""Native prefix/Q capture, with no future question visible during selection."""
import hashlib
import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from ar26kv.runtime import Inputs,sha_tensor,last_projection,media_kwargs,answer as vl_answer

def geometry(model,kind):
    language=model.thinker.model if kind=='Omni'else model.model.language_model
    c=language.config
    return language,(c.num_hidden_layers,c.num_attention_heads,c.num_key_value_heads,c.hidden_size//c.num_attention_heads)

class Supplier:
    def __init__(self,model,processor,kind):
        self.model=model;self.p=processor;self.kind=kind;self.vl=Inputs(processor)if kind=='VL'else None;self.video=None
    def get(self,item,verify=False):
        if self.kind=='VL':return self.vl.get(self.model,item,verify)
        from ar22kv.inputs import process,positions_for
        from inference.inputs import move_inputs_to_model
        from ar18kv.prefix import split_index
        from model.codec.modality import modality_tags
        cpu,identity=process(self.p,item,verify_native=verify)
        x=move_inputs_to_model(cpu,self.model);pos=positions_for(self.model,x,False)
        vi=(modality_tags(x['input_ids'],self.model)==2).nonzero().flatten().to(x['input_ids'].device)
        end=split_index(modality_tags(x['input_ids'],self.model),x)
        ti=(modality_tags(x['input_ids'],self.model)[:end]!=2).nonzero().flatten().to(x['input_ids'].device)
        grid=x['video_grid_thw'][0].tolist();grid=[int(grid[0]),int(grid[1])//2,int(grid[2])//2]
        identity['AR32_grid']=grid
        return x,pos,vi,ti,end,identity

class Teacher:
    def __init__(self,model,kind):
        self.model=model;self.kind=kind;self.language,self.geo=geometry(model,kind);self.active=False;self.handles=[]
        L,N,H,D=self.geo
        for l,block in enumerate(self.language.layers):
            def qhook(module,args,out,l=l):
                if not self.active:return
                q=out.detach().view(1,-1,N,D).transpose(1,2)[:,:,self.rows]
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
                q,_=apply_multimodal_rotary_pos_emb(q,q,self.cos[:,:,self.rows],self.sin[:,:,self.rows],self.sections)
                self.queries[l]=q[0].contiguous().cpu()
            self.handles.append(block.self_attn.q_proj.register_forward_hook(qhook))
    @torch.inference_mode()
    def capture(self,x,pos,end):
        L,N,H,D=self.geo;self.end=end;self.queries={}
        self.anchor_rows=torch.linspace(0,end-1,max(1,int(.1*end)),device='cuda').long()
        self.rows=torch.unique(torch.cat([self.anchor_rows,torch.arange(max(0,end-64),end,device='cuda')]),sorted=True)
        pp=pos[:,:end][:,None]if self.kind=='Omni'else pos[:,:,:end]
        self.cos,self.sin=self.language.rotary_emb(torch.empty((1,1,D),device='cuda',dtype=torch.bfloat16),pp)
        self.sections=self.language.layers[0].self_attn.rope_scaling['mrope_section'];cache=DynamicCache();self.active=True
        try:
            if self.kind=='VL':
                with last_projection(self.model):self.model(**media_kwargs(x,pos,end),past_key_values=cache)
            else:
                from ar18kv.prefix import media_prefill
                media_prefill(self.model,x,cache,pos,end,False)
        finally:self.active=False
        self.states={l:{'K_post':cache.key_cache[l][0].contiguous().cpu(),'V':cache.value_cache[l][0].contiguous().cpu()}for l in range(L)}
        assert all(v.shape==(H,end,D)for s in self.states.values()for v in s.values())
        self.ids_sha=sha_tensor(x['input_ids'][:,:end].cpu());self.pos_sha=sha_tensor(pp.cpu());del cache
        torch.cuda.synchronize()
    def assert_same(self,x,pos,end):
        pp=pos[:,:end][:,None]if self.kind=='Omni'else pos[:,:,:end]
        assert end==self.end and self.ids_sha==sha_tensor(x['input_ids'][:,:end].cpu())and self.pos_sha==sha_tensor(pp.cpu())
    def bf16_cache(self):
        c=DynamicCache();c.key_cache=[s['K_post'][None].cuda()for s in self.states.values()];c.value_cache=[s['V'][None].cuda()for s in self.states.values()];c._seen_tokens=self.end;return c
    def close(self):
        for h in self.handles:h.remove()

@torch.inference_mode()
def answer(model,p,kind,x,pos,end,cache,item):
    if kind=='Omni':
        from ar22kv.evaluation import answer as original
        return original(model,p,cache,x,pos,end,item)
    import re
    pred=vl_answer(model,p,x,pos,end,cache,item['options']);text=p.tokenizer.decode([pred['generated_token_id']],skip_special_tokens=True).strip();parsed=re.search(r'\b([A-H])\b',text);prediction=parsed.group(1)if parsed else None
    return {**pred,'prediction':prediction,'correct':prediction==item['answer'],'primary_decoded_token':text,'token_ids':[pred['generated_token_id']]}
