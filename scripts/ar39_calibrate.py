"""Measure every codec-dependent planning table on WorldSense prefixes only."""
import argparse,copy,gc,json,pathlib,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np,torch
from ar39cal.common import *
from ar37cq.runtime import load,Supplier,Teacher
from ar26kv.calibration import prefixes,FRACTIONS,GOPS
from ar26kv.matching import fixed_reference,references

@torch.no_grad()
def rates(raw,t,l,vi,grid,metric):
 nt,h,w=grid;p=h*w;ordinal=torch.arange(len(vi),device='cuda');ao=ordinal[(ordinal//p)%8==0];to=ordinal[(ordinal//p)%8!=0];aa=ao[torch.linspace(0,len(ao)-1,min(128,len(ao)),device='cuda').long()];ts=to[torch.linspace(0,len(to)-1,min(128,len(to)),device='cuda').long()];codes,scores=fixed_reference(raw,vi,ts,[t['K_scale'][l],t['V_scale'][l]],h,w,8);refs=vi[references(ts,codes,h,w,8)];pick=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:max(1,int(np.ceil(len(ts)/8)))];heads=torch.arange(raw['K'].shape[0],device='cuda')[:,None];output={}
 for s in ['K','V']:
  anchor=raw[s][:,vi[aa]];ref=raw[s].gather(1,refs[...,None].expand(-1,-1,128));prediction=prefixes(ref,t,s,l)[-1].float();residual=(raw[s][:,vi[ts]]-prediction)[heads,pick];parts=[]
  for vals,independent,pop in [(anchor,True,len(ao)),(residual,False,int(np.ceil(len(to)/8)))]:
   dec=prefixes(vals,t,s,l,independent=independent);zero=t[s+'_mean'][l].float()[:,None].expand_as(vals)if independent else torch.zeros_like(vals);losses=[float(((vals-y.float()).square()*metric[s][:,None]).sum(-1).mean())for y in [zero]+dec];parts.append({'loss':losses,'population':pop,'anchor':independent})
  output[s]=parts
 return output

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['Omni','VL'],required=True);ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,default=3);a=ap.parse_args();device(26);start=time.time();out=ROOT/f'data/runs/AR39_CURVES_{a.model}_part{a.shard}_v1';out.mkdir(parents=True,exist_ok=False);items=read(ROOT/'manifests/ar39/WorldSense_CAL64.json')[a.shard::a.shards];model,p=load(a.model);supplier=Supplier(model,p,a.model);teacher=Teacher(model,a.model);L,H,N,width=geometry(a.model);ratio=N//H;tables={};importance={};vm=[b.self_attn.o_proj.weight.detach().float().reshape(width,H,ratio,128).square().sum((0,2)).cpu()for b in teacher.language.layers]
 for rep in range(3):
  path=base(rep)/('artifacts/ar17/packages/PSRVQ_d16'if a.model=='Omni'else'artifacts/ar26/packages/STI_RVQ');assert read(path/'completion.json')['completed'];tables[rep]={k:v.cuda()for k,v in torch.load(path/'global.pt',weights_only=True,map_location='cpu').items()}
  if a.model=='Omni':
   from ar22kv.calibration import ranking_importance
   importance[rep],_=ranking_importance(model,tables[rep])
  else:importance[rep]=None
 from ar22kv.calibration import measure_layer as omni_measure
 from ar26kv.calibration import measure_layer as vl_measure
 measure=omni_measure if a.model=='Omni'else vl_measure;save(out/'run_manifest.json',{'model':a.model,'shard':a.shard,'source_SHA256':sha(__file__),'input_SHA256':sha(ROOT/'manifests/ar39/WorldSense_CAL64.json'),'sources':[x['source_id']for x in items],'replicates':[0,1,2],'no_question_or_answer_used_in_loss_tables':True})
 try:
  with(out/'by_video.jsonl').open('x')as f:
   for index,it in enumerate(items):
    x,pos,vi,ti,end,identity=supplier.get(it);teacher.capture(x,pos,end,identity);grid=identity.get('grid',identity.get('AR32_grid'));qmom=teacher.base.q_moments if a.model=='Omni'else teacher.base.moments
    for rep,t in tables.items():
     layers=[];rate=[]
     for l in range(L):
      st=teacher.states[l];native={k:v[None]for k,v in st.items()};layers.append(measure(l,native,qmom[l],vm[l],t,vi.to(torch.int32),ti,grid,importance[rep],96));raw={'K':st['K_pre'].cuda().float(),'V':st['V'].cuda().float()};rate.append(rates(raw,t,l,vi.long(),grid,{'K':qmom[l].cuda()/128.,'V':vm[l].cuda()}))
     row={'id':it['id'],'source_id':it['source_id'],'geometry_group':it['geometry_group'],'replicate':rep,'identity':identity,'layers':layers,'rate':rate,'media_end':end,'QA_used':False};f.write(json.dumps(row)+'\n');f.flush();print(json.dumps({'model':a.model,'source':index+1,'replicate':rep,'seconds':time.time()-start}),flush=True)
    teacher.clear();del x,pos,vi,ti,raw,layers,rate;gc.collect();torch.cuda.empty_cache();save(out/'progress.json',{'complete_sources':index+1,'expected_sources':len(items),'elapsed_s':time.time()-start})
  save(out/'completion.json',{'completed':True,'model':a.model,'sources':len(items),'replicates':[0,1,2],'by_video_SHA256':sha(out/'by_video.jsonl'),'elapsed_s':time.time()-start,'importance':{str(r):{s:[v.tolist()for v in values]for s,values in imp.items()}if imp else None for r,imp in importance.items()},'no_benchmark_media_used':True,'GPU1_used':False})
 except BaseException as e:save(out/'failure.json',{'completed':False,'error':repr(e),'traceback':traceback.format_exc()});raise
 finally:teacher.close()

def aggregate(kind,shards=3):
 records=[];metadata=[];L,H,N,width=geometry(kind)
 for part in range(shards):
  folder=ROOT/f'data/runs/AR39_CURVES_{kind}_part{part}_v1';d=read(folder/'completion.json');assert d['completed']and d['by_video_SHA256']==sha(folder/'by_video.jsonl');metadata.append(d);records.extend(json.loads(line)for line in(folder/'by_video.jsonl').read_text().splitlines())
 assert len(records)==64*3 and len({(r['source_id'],r['replicate'])for r in records})==192
 for rep in range(3):
  rows=[r for r in records if r['replicate']==rep];groups={};ranks=['norm','sensitivity']if kind=='Omni'else['norm']
  for group in ['short','medium','long']:
   rr=[r for r in rows if r['geometry_group']==group];assert len(rr)in[21,22];groups[group]={rank:{str(g):{s:[{term:np.mean([r['layers'][l][rank][str(g)][s][term]for r in rr],0).tolist()for term in ['anchor','target','query']}for l in range(L)]for s in ['K','V']}for g in GOPS}for rank in ranks}
  scales={}
  for s in ['K','V']:
   scales[s]=float(np.mean([sum(c['anchor'][0]/8+c['target'][3][0]*7/8 for c in groups[g]['norm']['8'][s])for g in groups]));assert scales[s]>0
   for g in groups:
    for rank in ranks:
     for gg in GOPS:
      for cell in groups[g][rank][str(gg)][s]:
       for term in cell:cell[term]=(np.asarray(cell[term])/scales[s]).tolist()
  rateout={}
  for objective in ['plain','fanout']:
   result={}
   for s in ['K','V']:
    acc=np.zeros((L,7));den=np.zeros(L)
    for r in rows:
     for l in range(L):
      for c in r['rate'][l][s]:
       factor=7.125 if objective=='fanout'and c['anchor']else 1.;weight=c['population']*factor/r['media_end'];acc[l]+=np.array(c['loss'])*weight;den[l]+=weight
    acc/=den[:,None];acc/=acc[:,1].sum();assert np.isfinite(acc).all();result[s]=acc.tolist()
   rateout[objective]=result
  bank=base(rep)/('artifacts/ar17/packages/PSRVQ_d16'if kind=='Omni'else'artifacts/ar26/packages/STI_RVQ');common={'completed':True,'actual_calibration_media':64,'actual_media':64,'sources':[r['source_id']for r in rows],'calibration_dataset':'WorldSense','no_benchmark_media_used':True,'bank_SHA256':sha(bank/'global.pt'),'bank_completion_SHA256':sha(bank/'completion.json'),'fractions':FRACTIONS,'gops':GOPS,'groups':groups,'ranking_importance':metadata[0]['importance'][str(rep)],'normalization_per_side':scales,'question_or_answer_labels_used':False,'QA_used':False,'replicate':rep}
  prefix='data/ar22/conditional_curves.json'if kind=='Omni'else'data/ar26/VL_conditional_curves.json';rp='data/ar21/rate_curves.json'if kind=='Omni'else'data/ar26/VL_rate_curves.json';save(base(rep)/prefix,common);save(base(rep)/rp,{'completed':True,'actual_media':64,'calibration_dataset':'WorldSense','no_benchmark_media_used':True,'bank_SHA256':sha(bank/'global.pt'),'fixed_bank_completion_SHA256':sha(bank/'completion.json'),**rateout});save(ROOT/f'data/ar39/CURVES_{kind}_r{rep}.json',{'completed':True,'conditional_path':str((base(rep)/prefix).relative_to(ROOT)),'conditional_SHA256':sha(base(rep)/prefix),'rate_path':str((base(rep)/rp).relative_to(ROOT)),'rate_SHA256':sha(base(rep)/rp)})
 if kind=='Omni':
  for rep in range(3):
   source=base(rep)/'data/ar22/conditional_curves.json';fixed=copy.deepcopy(read(source));fi=fixed['fractions'].index(.125);fixed['fractions']=[.125]
   for group in fixed['groups'].values():
    for ranking in group.values():
     for g in ranking.values():
      for cells in g.values():
       for cell in cells:cell['target']=[cell['target'][fi]]
   fixed.update(fixed_innovation_fraction=.125,source_full_conditional_SHA256=sha(source),slicing_is_not_new_fitting=True);save(base(rep)/'data/ar22/conditional_fixedI125.json',fixed)
 print(json.dumps({'completed':True,'model':kind,'replicates':3,'sources_per_fit':64}))
if __name__=='__main__':
 if '--aggregate'in sys.argv:aggregate(sys.argv[sys.argv.index('--aggregate')+1])
 else:main()
