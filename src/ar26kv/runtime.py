"""Native processor/positions and before-question RAM teacher for Qwen2.5-VL."""
import hashlib,json,pathlib,time
from contextlib import contextmanager
import numpy as np
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from transformers.cache_utils import DynamicCache

ROOT=pathlib.Path(__file__).resolve().parents[2]
MODEL=ROOT/'artifacts/ar26/models/Qwen2.5-VL-7B-Instruct'

def sha_tensor(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def save(path,obj):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)

def load():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(40*2**30/torch.cuda.get_device_properties(0).total_memory)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='flash_attention_2',local_files_only=True).eval()
    model.requires_grad_(False)
    c=model.config.text_config
    assert (c.num_hidden_layers,c.num_attention_heads,c.num_key_value_heads,c.hidden_size//c.num_attention_heads)==(28,28,4,128)
    assert sum(p.numel() for p in model.parameters())<10_000_000_000
    return model,AutoProcessor.from_pretrained(MODEL,local_files_only=True)

@contextmanager
def last_projection(model):
    original=model.lm_head.forward
    def final_only(x):return original(x[:,-1:])
    model.lm_head.forward=final_only
    try:yield
    finally:model.lm_head.forward=original

class Inputs:
    def __init__(self,processor):self.p=processor;self.video=None
    def get(self,model,item,verify=False):
        p=self.p
        message=[{'role':'user','content':[{'type':'video','video':item['frame_store']},{'type':'text','text':item['prompt']}]}]
        text=p.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
        fresh=self.video!=item['video'];native=None
        if fresh or verify:
            with np.load(item['frame_store']) as a:frames=a['frames'];fps=float(a['effective_fps'])
            native=dict(p(text=[text],videos=[frames],fps=fps,min_pixels=4*28*28,max_pixels=item['max_pixels'],return_tensors='pt'))
            common={k:v for k,v in native.items() if k not in ['input_ids','attention_mask']}
            if fresh:
                self.common=common;self.video=item['video'];self.frame_sha=hashlib.sha256(frames.tobytes()).hexdigest()
            else:
                for k,v in common.items():
                    assert torch.equal(v,self.common[k]) if torch.is_tensor(v) else v==self.common[k]
            del frames
        count=int(self.common['video_grid_thw'].prod())//p.video_processor.merge_size**2
        assert text.count(p.video_token)==1
        expanded=text.replace(p.video_token,p.video_token*count)
        q=dict(p.tokenizer([expanded],return_tensors='pt',padding=True))
        if native is not None:
            assert torch.equal(q['input_ids'],native['input_ids'])
            assert torch.equal(q['attention_mask'],native['attention_mask'])
        cpu={**self.common,**q}
        x={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in cpu.items()}
        pos,_=model.model.get_rope_index(input_ids=x['input_ids'],video_grid_thw=x['video_grid_thw'],second_per_grid_ts=x['second_per_grid_ts'],attention_mask=x['attention_mask'])
        vi=(x['input_ids'][0]==model.config.video_token_id).nonzero().flatten()
        end=int(vi[-1])+1;ti=(x['input_ids'][0,:end]!=model.config.video_token_id).nonzero().flatten()
        grid=x['video_grid_thw'][0].tolist();grid=[int(grid[0]),int(grid[1])//2,int(grid[2])//2]
        assert len(vi)==np.prod(grid)
        identity={'input_ids_SHA256':sha_tensor(q['input_ids']),'frames_SHA256':self.frame_sha,'positions_SHA256':sha_tensor(pos.cpu()),'grid':grid,'second_per_grid_ts':[float(v) for v in cpu['second_per_grid_ts']],'native_processor_verified':native is not None,'tokens':int(q['input_ids'].shape[1]),'media_end':end,'video_tokens':len(vi),'prefix_text':len(ti)}
        return x,pos,vi,ti,end,identity

def media_kwargs(x,pos,end):
    return {**{k:v for k,v in x.items() if k not in ['input_ids','attention_mask']},'input_ids':x['input_ids'][:,:end],'attention_mask':x['attention_mask'][:,:end],'position_ids':pos[:,:,:end],'cache_position':torch.arange(end,device='cuda'),'use_cache':True}

class Teacher:
    def __init__(self,model):
        self.model=model;self.handles=[];self.active=False
        for l,b in enumerate(model.model.language_model.layers):
            def kh(m,a,o,l=l):
                if self.active:self.raw[l]=o.detach().view(1,-1,4,128).transpose(1,2).contiguous().cpu()
            def qh(m,a,o,l=l):
                if self.active:
                    q=o.detach().view(1,-1,4,7,128);ix=torch.linspace(0,q.shape[1]-1,min(128,q.shape[1]),device=q.device).long()
                    self.moments[l]=q[:,ix].float().square().mean((0,1,3)).cpu()
            self.handles.extend([b.self_attn.k_proj.register_forward_hook(kh),b.self_attn.q_proj.register_forward_hook(qh)])
    @torch.inference_mode()
    def capture(self,x,pos,end):
        self.raw={};self.moments={};self.active=True;cache=DynamicCache()
        try:
            with last_projection(self.model):self.model(**media_kwargs(x,pos,end),past_key_values=cache)
        finally:self.active=False
        self.states={l:{'K_pre':self.raw[l][0],'K_post':cache.key_cache[l][0].contiguous().cpu(),'V':cache.value_cache[l][0].contiguous().cpu()} for l in range(28)}
        self.end=end;self.ids_sha=sha_tensor(x['input_ids'][:,:end].cpu());self.pos_sha=sha_tensor(pos[:,:,:end].cpu())
        assert all(v.shape==(4,end,128) for s in self.states.values() for v in s.values())
        del cache;torch.cuda.synchronize()
    def assert_same(self,x,pos,end):
        assert end==self.end and self.ids_sha==sha_tensor(x['input_ids'][:,:end].cpu()) and self.pos_sha==sha_tensor(pos[:,:,:end].cpu())
    def bf16_cache(self):
        cache=DynamicCache();cache.key_cache=[self.states[l]['K_post'][None].cuda() for l in range(28)];cache.value_cache=[self.states[l]['V'][None].cuda() for l in range(28)];cache._seen_tokens=self.end;return cache
    def close(self):
        for h in self.handles:h.remove()

@torch.inference_mode()
def answer(model,p,x,pos,end,cache,options):
    with last_projection(model):
        out=model(input_ids=x['input_ids'][:,end:],attention_mask=x['attention_mask'],position_ids=pos[:,:,end:],cache_position=torch.arange(end,x['input_ids'].shape[1],device='cuda'),past_key_values=cache,use_cache=True)
    z=out.logits[0,-1].float();token=int(z.argmax());text=p.tokenizer.decode([token]).strip()
    letters=[chr(65+i) for i in range(len(options))];ids=[p.tokenizer.encode(a,add_special_tokens=False)[0] for a in letters]
    logits=z[ids].cpu().tolist()
    return {'pred':text if text in letters else None,'generated_first_token':text,'generated_token_id':token,'option_logits':logits,'option_argmax':letters[int(np.argmax(logits))]}
