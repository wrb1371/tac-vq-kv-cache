"""Whole-AV streaming NOVA moments and aligned pre-MRoPE GSRQ CE gradients.

Reservoir slots are shared across all layers/heads/kinds. Incomplete attempts
are never fitted; reruns use a new output directory and preserve failures.
"""
import argparse,gc,hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'third_party/OmniKVQuant_snapshot')]
from r14kv.common import runtime,input_for,prefill,last_inputs,selected_rope,apply_key_rope,json_save,sha

class Collector:
    def __init__(self,model,folder,capacity):
        import torch
        self.model,self.folder,self.capacity=model,folder,capacity
        self.pool=[];self.grad=[]
        for l in range(36):
            self.pool.append(np.memmap(folder/f'layer{l:02}_vectors.u16',mode='w+',dtype=np.uint16,shape=(3,2,capacity,128)))
            self.grad.append(np.memmap(folder/f'layer{l:02}_gradient_norms.f32',mode='w+',dtype=np.float32,shape=(2,2,capacity)))
        self.positions=np.memmap(folder/'coordinates.i32',mode='w+',dtype=np.int32,shape=(capacity,3))
        self.ids=np.memmap(folder/'source_and_token.i32',mode='w+',dtype=np.int32,shape=(capacity,2))
        self.mods=np.memmap(folder/'modality.u8',mode='w+',dtype=np.uint8,shape=(capacity,))
        self.priorities=np.full(capacity,np.inf);self.n=0;self.seen=0;self.stage='idle';self.active=None;self.pre={}
        self.stats={k:torch.zeros(36,2,128,128,dtype=torch.float64) for k in ['q2','k2','o2','q2_answer']}
        self.stats['ksum']=torch.zeros(36,2,128,dtype=torch.float64)
        self.count=0;self.answer_count=0;self.handles=[]
        import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as hf
        self.hf=hf;self.old_rope=hf.apply_multimodal_rotary_pos_emb
        def wrapped_rope(q,k,cos,sin,sections,**kw):
            q,k=self.old_rope(q,k,cos,sin,sections,**kw)
            if self.stage in ['prefix','answer']:
                with torch.no_grad():self.observe_rope(q,k)
            return q,k
        hf.apply_multimodal_rotary_pos_emb=wrapped_rope
        for l,layer in enumerate(model.thinker.model.layers):
            attn=layer.self_attn;assert attn.layer_idx==l
            def before(m,args,kwargs,l=l):
                self.active=l
                if l==0 and self.stage=='prefix':
                    pos=kwargs['position_ids'];cos,sin=kwargs['position_embeddings'];self.cos,self.sin=selected_rope(cos,sin)
                    self.positions[self.slots]=pos[:,0,self.take].T.cpu().numpy()
            def key_hook(m,args,out,l=l):
                if self.stage=='prefix':self.pre[l]=out.view(1,out.shape[1],2,128).transpose(1,2)
            def out_hook(m,args,l=l):
                if self.stage=='prefix':
                    with torch.no_grad():
                        o=args[0].view(1,-1,16,128).transpose(1,2);self.stats['o2'][l]+=self.gqa_moment(o)
            self.handles += [attn.register_forward_pre_hook(before,with_kwargs=True),attn.k_proj.register_forward_hook(key_hook),attn.o_proj.register_forward_pre_hook(out_hook)]
    @staticmethod
    def gqa_moment(x):
        import torch
        out=[]
        for h in range(2):
            z=x[0,h*8:(h+1)*8].reshape(-1,128).double();out.append((z.T@z).cpu())
        return torch.stack(out)
    def plan(self,source,source_index,length,tags):
        seed=int(hashlib.sha256(('R14-reservoir:'+source).encode()).hexdigest()[:16],16)
        priorities=np.random.default_rng(seed).random(length)
        combined=np.concatenate([self.priorities[:self.n],priorities]);keep=min(self.capacity,len(combined))
        selected=np.argpartition(combined,keep-1)[:keep]
        take=np.sort(selected[selected>=self.n]-self.n)
        occupied=set(selected[selected<self.n].tolist());slots=np.array([i for i in range(keep) if i not in occupied],dtype=np.int64)
        assert len(take)==len(slots)
        self.take,self.slots=take,slots;self.priorities[slots]=priorities[take];self.n=keep;self.seen+=length
        self.ids[slots]=np.stack([np.full(len(take),source_index),take],axis=-1)
        self.mods[slots]=tags[take].cpu().numpy().astype(np.uint8)
    def observe_rope(self,q,k):
        import torch
        l=self.active;assert l is not None and q.shape[1]==16 and k.shape[1]==2
        if self.stage=='answer':
            assert q.shape[2]==1;self.stats['q2_answer'][l]+=self.gqa_moment(q)
            if l==0:self.answer_count+=1
            return
        assert k.shape[2]==len(self.tags)
        self.stats['q2'][l]+=self.gqa_moment(q)
        self.stats['ksum'][l]+=k[0].double().sum(1).cpu()
        for h in range(2):
            z=k[0,h].double();self.stats['k2'][l,h]+=(z.T@z).cpu()
        ix=torch.as_tensor(self.take,device=k.device,dtype=torch.long)
        self.pool[l][0,:,self.slots,:]=self.pre[l][0].index_select(1,ix).permute(1,0,2).contiguous().view(torch.uint16).cpu().numpy()
        self.pool[l][1,:,self.slots,:]=k[0].index_select(1,ix).permute(1,0,2).contiguous().view(torch.uint16).cpu().numpy()
        if l==0:self.count+=k.shape[2]
    def save_values(self,cache):
        import torch
        ix=torch.as_tensor(self.take,device=cache.value_cache[0].device,dtype=torch.long)
        for l in range(36):
            self.pool[l][2,:,self.slots,:]=cache.value_cache[l][0].index_select(1,ix).permute(1,0,2).contiguous().view(torch.uint16).cpu().numpy()
    def finish(self):
        self.hf.apply_multimodal_rotary_pos_emb=self.old_rope
        for h in self.handles:h.remove()
        for x in self.pool+self.grad+[self.ids,self.mods,self.positions]:x.flush()

def clone_cache(cache):
    from transformers.cache_utils import DynamicCache
    other=DynamicCache();other.key_cache=list(cache.key_cache);other.value_cache=list(cache.value_cache);other._seen_tokens=cache._seen_tokens
    return other

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--data',required=True);p.add_argument('--capacity',type=int,default=262144);a=p.parse_args()
    import torch
    from transformers.cache_utils import DynamicCache
    from inference.data import load_items
    from model.codec.modality import modality_tags
    folder=ROOT/'artifacts/round14/calibration'/a.run_id;folder.mkdir(parents=True,exist_ok=False)
    out=ROOT/'runs'/a.run_id;out.mkdir(parents=True,exist_ok=True)
    data=ROOT/a.data;assert a.data in ['manifests/round14/CAL_pilot4.json','manifests/round14/CAL64_one_question.json']
    code={str(path.relative_to(ROOT)):sha(path) for path in [pathlib.Path(__file__),ROOT/'src/r14kv/common.py',ROOT/'src/r13kv/execution.py',ROOT/'src/r13kv/media_cache.py']}
    for name,h in code.items():
        path=ROOT/'artifacts/round14/source_snapshots'/(h+'.py');path.parent.mkdir(exist_ok=True,parents=True)
        if not path.exists():path.write_bytes((ROOT/name).read_bytes())
    model,processor,modelmeta,media=runtime();versions={n:x._version for n,x in model.named_parameters()}
    collector=Collector(model,folder,a.capacity)
    manifest={'model':modelmeta,'data':a.data,'data_SHA256':sha(data),'capacity':a.capacity,'code_SHA256':code,
        'gradient_loss':'CAL gold option first-token CE after unmodified FP prompt[:-1]; all historical pre-MRoPE K/V leafs, native last prompt token once',
        'loss_scale':4096.,'sampling':'source-keyed uniform priority reservoir, natural token distribution shared exact slots for all layers/heads/kinds',
        'NOVA_statistic':'all prefix post-MRoPE Q/K; real pre-W_O O; GQA sum, normalize by prefix token count',
        'pilot_only':a.data.endswith('CAL_pilot4.json'),'started_epoch':time.time()}
    json_save(out/'run_manifest.json',manifest);progress=[]
    for si,item in enumerate(load_items(data)):
        start=time.time();collector.stage='idle';collector.pre.clear();inputs,imeta=input_for(model,processor,item)
        tags=modality_tags(inputs['input_ids'],model)[:-1];collector.tags=tags
        collector.plan(item['video'],si,len(tags),tags)
        cache=DynamicCache();collector.stage='prefix'
        with torch.no_grad():prefill(model,inputs,cache)
        collector.stage='idle';collector.save_values(cache)
        kleaves=[];vleaves=[];gcach=DynamicCache();rope_exact=True
        for l in range(36):
            kp=collector.pre[l].detach().requires_grad_(True);v=cache.value_cache[l].detach().requires_grad_(True)
            kr=apply_key_rope(kp,collector.cos,collector.sin)
            rope_exact=rope_exact and torch.equal(kr.detach(),cache.key_cache[l])
            assert torch.equal(kr.detach(),cache.key_cache[l]),'pre-MRoPE capture/reapply must match original cached K exactly'
            gcach.update(kr,v,l);kleaves.append(kp);vleaves.append(v)
        # Same native generation call as R13, reusing this immutable FP prefix.
        collector.stage='teacher_check';teacher_cache=clone_cache(cache)
        with torch.no_grad():
            ids=model.generate(**inputs,use_audio_in_video=True,return_audio=False,thinker_max_new_tokens=1,do_sample=False,thinker_past_key_values=teacher_cache)
        teacher_token=int(ids[0,-1]);del ids,teacher_cache,cache
        collector.stage='answer';prepared=last_inputs(model,inputs,gcach)
        output=model.thinker(**prepared,return_dict=True);logits=output.logits[:,-1].float()
        direct_token=int(logits.argmax(-1));assert direct_token==teacher_token,'direct differentiable boundary must produce native first-answer token'
        target=processor.tokenizer.encode(item['answer'],add_special_tokens=False);assert len(target)==1
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor(target,device=logits.device))
        (loss*4096.).backward();details={};all_nonzero=True
        for l in range(36):
            for kind,leaf in enumerate([kleaves[l],vleaves[l]]):
                assert leaf.grad is not None
                norms=leaf.grad.detach().float().norm(dim=-1)[0]/4096.
                assert torch.isfinite(norms).all() and bool(norms.max()>0),'missing/nonfinite gradients cannot become unweighted fitting'
                collector.grad[l][kind,:,collector.slots]=norms[:,collector.take].T.cpu().numpy()
                details[f'{l}_{kind}']={'zero_fraction':float((norms==0).float().mean()),'median':float(norms.median()),'max':float(norms.max())}
        collector.stage='idle'
        row={'source':item['video'],'question_id':item['id'],'source_index':si,'actual_prefix_tokens':len(tags),'new_reservoir_positions':len(collector.take),
            'reservoir_positions':collector.n,'total_seen_positions':collector.seen,'input':imeta,'modality_counts':{str(int(x)):int((tags==x).sum()) for x in tags.unique()},
            'teacher_first_answer_token':teacher_token,'differentiable_first_answer_token':direct_token,'native_MRoPE_reapply_bitwise_exact':rope_exact,
            'CE':float(loss.detach()),'gradients':details,'seconds':time.time()-start}
        progress.append(row);json_save(folder/'source_progress.json',progress);json_save(out/'source_progress.json',[{'source':x['source']} for x in progress])
        print(json.dumps({'source_completed':si+1,'video':item['video'],'prefix_tokens':len(tags),'reservoir':collector.n,'seconds':row['seconds'],'all_gradient_layers_nonzero':True}),flush=True)
        collector.pre.clear();del inputs,prepared,output,logits,loss,gcach,kleaves,vleaves,leaf,kp,v,kr,norms,tags;gc.collect();torch.cuda.empty_cache()
    collector.finish();assert versions=={n:x._version for n,x in model.named_parameters()}
    assert all(sha(ROOT/path)==h for path,h in code.items())
    stats={**collector.stats,'prefix_token_count':collector.count,'answer_query_count':collector.answer_count}
    torch.save(stats,folder/'NOVA_moments.pt')
    record={**manifest,'completed':True,'completed_epoch':time.time(),'actual_sources':len(progress),'actual_reservoir_positions':collector.n,
        'total_prefix_positions_seen':collector.seen,'parameter_versions_unchanged':True,'sample_layout':{'vector_file_shape':[3,2,a.capacity,128],'vector_kinds':['preMRoPE_K','postMRoPE_K','V'],'gradient_file_shape':[2,2,a.capacity]},
        'artifacts':{p.name:{'bytes':p.stat().st_size,'SHA256':sha(p)} for p in folder.iterdir() if p.is_file()},'model_media_cache_stats':media}
    json_save(folder/'completion.json',record);json_save(out/'completion.json',record)
    print(json.dumps({'completed':True,'sources':len(progress),'reservoir':collector.n,'output':str(folder.relative_to(ROOT))}),flush=True)
if __name__=='__main__':main()
