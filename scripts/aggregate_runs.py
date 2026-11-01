"""Aggregate three complete run files without rounding or averaging subpanels early."""
import argparse,collections,json,math,pathlib

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inputs',nargs=3,required=True);ap.add_argument('--output',required=True);a=ap.parse_args();runs=[];replicates=[];configurations=[]
 for file in a.inputs:
  with pathlib.Path(file).open()as f:rows=[json.loads(line)for line in f]
  byid={r['id']:r for r in rows};assert len(rows)==len(byid),'One observation per question per input file is required'
  reps={r['replicate']for r in rows};assert len(reps)==1;replicates.append(next(iter(reps)))
  cfg={(r.get('backbone',r.get('model')),r['method'],r.get('ablation'))for r in rows};assert len(cfg)==1;configurations.append(next(iter(cfg)))
  runs.append(byid)
 assert sorted(replicates)==[0,1,2],'Three distinct fitted-run identifiers are required'
 assert len(set(configurations))==1,'Backbone/method/ablation must match across runs'
 assert all(set(r)==set(runs[0])for r in runs),'Run question sets differ'
 for id in runs[0]:
  for run in runs[1:]:
   for key in ['source_id','dataset','duration_group','task_type','budget_mib']:
    assert run[id].get(key)==runs[0][id].get(key),(id,key)
 panels={}
 for ds in sorted({r['dataset']for r in runs[0].values()}):
  ids=[id for id,r in runs[0].items()if r['dataset']==ds]
  if ds in ['VideoMME','TempCompass-MC']:assert len(ids)=={'VideoMME':2700,'TempCompass-MC':1580}[ds],'Paper metrics require the complete benchmark'
  axes=['ALL']+(['short','medium','long']if ds=='VideoMME'else['macro']if ds=='TempCompass-MC'else[])
  for axis in axes:
   values=[]
   for run in runs:
    rows=[run[id]for id in ids if axis in ['ALL','macro']or run[id]['duration_group']==axis]
    assert rows
    if any(r['status']!='COMPLETE'for r in rows):values.append(None);continue
    if axis=='macro':
     groups=collections.defaultdict(list)
     for r in rows:groups[r['task_type']].append(r)
     assert set(groups)=={'action','direction','speed','order','attribute_change'}
     score=math.fsum(100*sum(r['correct']for r in groups[t])/len(groups[t])for t in ['action','direction','speed','order','attribute_change'])/5
    else:score=100*sum(r['correct']for r in rows)/len(rows)
    values.append(score)
   if any(v is None for v in values):assert all(v is None for v in values);mean=std=None
   else:
    mean=math.fsum(values)/3;offsets=[v-values[0]for v in values];center=math.fsum(offsets)/3;std=math.sqrt(math.fsum((d-center)**2 for d in offsets)/2)
   panels[ds+'/'+axis]={'mean_accuracy_pct':mean,'run_accuracies_pct':values,'sample_std_pct':std,'three_actual_run_files':a.inputs}
 out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(panels,indent=2)+'\n');print(json.dumps(panels,indent=2))
if __name__=='__main__':main()
