"""Native media-only statistics plus gold-option KV-leaf gradients on WorldSense."""
import argparse,gc,json,pathlib,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'third_party/OmniKVQuant_snapshot')]
import torch
from ar39cal.common import *
from ar37cq.runtime import load,Supplier,Teacher,native_rope
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['Omni','VL'],required=True);ap.add_argument('--limit',type=int,default=64);ap.add_argument('--attempt',default='v1');a=ap.parse_args();device(42);started=time.time();out=ROOT/f'data/runs/AR39_CAPTURE_{a.model}_{a.attempt}';out.mkdir(parents=True,exist_ok=False)
 items=read(ROOT/'manifests/ar39/WorldSense_CAL64.json')[:a.limit];model,p=load(a.model);device(42);teacher=Teacher(model,a.model);supplier=Supplier(model,p,a.model);language=teacher.language;L,H,N,width=geometry(a.model);ratio=N//H;versions=[v._version for v in model.parameters()];mom={k:torch.zeros(L,H,128,128,dtype=torch.float64)for k in ['q2','k2','o2']};mom['ksum']=torch.zeros(L,H,128,dtype=torch.float64);mom['q_energy']=torch.zeros(L,H,128,dtype=torch.float64);count=0;handles=[];active=[False];context={};records=[]
 save(out/'run_manifest.json',{'model':a.model,'inputs_SHA256':sha(ROOT/'manifests/ar39/WorldSense_CAL64.json'),'protocol_SHA256':sha(ROOT/'configs/ar39/protocol.json'),'source_SHA256':sha(__file__),'physical_GPU':int(__import__('os').environ['CUDA_VISIBLE_DEVICES']),'limit':a.limit,'gradients':'WorldSense gold option CE, loss scale4096; frozen model parameters; no benchmark QA','prefix':'video-text only, before question','samples_per_video_per_head':256})
 def cov(z):
  z=z.reshape(H,ratio,-1,128).reshape(H,-1,128).double();return(z.transpose(1,2)@z).cpu()
 for l,b in enumerate(language.layers):
  def qhook(m,args,o,l=l):
   if not active[0]:return
   q=o.detach().view(1,-1,N,128).transpose(1,2)[0];mom['q_energy'][l]+=q.reshape(H,ratio,-1,128).double().square().sum((1,2)).cpu()/ratio
   post=native_rope(model,a.model,q,context['pos'],context['end'])[0];mom['q2'][l]+=cov(post)
  def ohook(m,args,l=l):
   if active[0]:mom['o2'][l]+=cov(args[0].detach().view(1,-1,N,128).transpose(1,2)[0])
  handles.extend([b.self_attn.q_proj.register_forward_hook(qhook),b.self_attn.o_proj.register_forward_pre_hook(ohook)])
 try:
  for index,it in enumerate(items):
   before=time.time();x,pos,vi,ti,end,identity=supplier.get(it,verify=True);context.update(pos=pos,end=end);active[0]=True
   try:teacher.capture(x,pos,end,identity)
   finally:active[0]=False
   count+=end
   for l,st in teacher.states.items():
    k=st['K_post'].double();mom['k2'][l]+=k.transpose(1,2)@k;mom['ksum'][l]+=k.sum(1)
   from transformers.cache_utils import DynamicCache
   cache=DynamicCache();cache._seen_tokens=end;leaves=[]
   for l,st in teacher.states.items():
    k=st['K_pre'].cuda().clone().requires_grad_(True);v=st['V'].cuda().clone().requires_grad_(True);kr=native_rope(model,a.model,k,pos,end);assert torch.equal(kr.detach()[0],st['K_post'].cuda());cache.key_cache.append(kr);cache.value_cache.append(v[None]);leaves.append((k,v))
   with torch.autograd.graph.save_on_cpu(pin_memory=True):
    if a.model=='Omni':
     from ar18kv.prefix import question_forward
     logits=question_forward(model,x,cache,pos,end,False)[0].float()
    else:
     from ar26kv.runtime import last_projection
     with last_projection(model):z=model(input_ids=x['input_ids'][:,end:],attention_mask=x['attention_mask'],position_ids=pos[:,:,end:],cache_position=torch.arange(end,x['input_ids'].shape[1],device='cuda'),past_key_values=cache,use_cache=True)
     logits=z.logits[0,-1].float()
    token=p.tokenizer.encode(it['answer'],add_special_tokens=False);assert len(token)==1;loss=torch.nn.functional.cross_entropy(logits[None],torch.tensor(token,device='cuda'));assert torch.isfinite(loss);(loss*4096.).backward()
   take=vi[torch.linspace(0,len(vi)-1,min(256,len(vi)),device='cuda').long()];sample={}
   for l,(k,v)in enumerate(leaves):
    sample[l]={key:teacher.states[l][key][:,take.cpu()].clone()for key in ['K_pre','K_post','V']};sample[l]['K']=sample[l].pop('K_pre')
    for side,leaf in [('K',k),('V',v)]:
     assert leaf.grad is not None;g=leaf.grad[:,take].float().cpu()/4096.;assert torch.isfinite(g).all()and g.square().sum()>0;sample[l][side+'_grad']=g.clone()
   path=pool(a.model)/(it['id']+'.pt');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();torch.save(sample,path)
   rec={'id':it['id'],'source_id':it['source_id'],'geometry_group':it['geometry_group'],'identity':identity,'grid':identity.get('grid',identity.get('AR32_grid')),'path':str(path.relative_to(ROOT)),'SHA256':sha(path),'bytes':path.stat().st_size,'loss':float(loss.detach()),'gold_option':it['answer'],'sample_positions':take.cpu().tolist(),'prefix_end':end,'elapsed_s':time.time()-before,'GPU_peak_MiB':torch.cuda.max_memory_allocated()/2**20};records.append(rec);save(out/'progress.json',{'rows':records,'completed_videos':len(records),'elapsed_s':time.time()-started});print(json.dumps({k:rec[k]for k in ['id','loss','elapsed_s','GPU_peak_MiB']}),flush=True)
   teacher.clear();del x,pos,vi,ti,cache,leaves,sample,logits,loss,k,v,g,kr
   if a.model=='VL':del z
   gc.collect();torch.cuda.empty_cache()
  mom['prefix_token_count']=count;torch.save(mom,pool(a.model)/'moments.pt');assert versions==[v._version for v in model.parameters()]and all(v.grad is None for v in model.parameters());save(out/'completion.json',{'completed':True,'model':a.model,'rows':records,'source_count':len(records),'elapsed_s':time.time()-started,'moments_SHA256':sha(pool(a.model)/'moments.pt'),'model_weights_unchanged':True,'evaluation_media_used':False,'GPU1_used':False})
 except BaseException as e:save(out/'failure.json',{'completed':False,'error':repr(e),'traceback':traceback.format_exc(),'complete_sources':len(records),'failed_attempt_used_for_fit':False});raise
 finally:
  active[0]=False
  for h in handles:h.remove()
  teacher.close()
if __name__=='__main__':main()
