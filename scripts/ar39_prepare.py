"""Lock the common WorldSense pixels, clock, question and source identities."""
import hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ar39cal.common import sha,save,read
def main():
 import numpy as np,decord
 from qwen_omni_utils.v2_5.vision_process import smart_resize
 source=ROOT/'manifests/round14/CAL64_one_question.json';items=read(source)
 assert len(items)==64 and len({x['video']for x in items})==64
 ordered=sorted(items,key=lambda x:hashlib.sha256(('AR39:'+x['video']).encode()).hexdigest());rows=[];records=[]
 for i,x in enumerate(ordered):
  group=['short','medium','long'][i%3];nf={'short':64,'medium':128,'long':256}[group];raw=ROOT/'assets/round13/WorldSense/videos'/x['video'];assert raw.exists()
  dest=ROOT/'artifacts/ar39/frames'/(pathlib.Path(x['video']).stem+'.npz');rec=ROOT/'data/ar39/frames'/(pathlib.Path(x['video']).stem+'.json')
  if dest.exists():
   record=read(rec);assert record['archive_SHA256']==sha(dest)
  else:
   v=decord.VideoReader(str(raw),num_threads=2);n=len(v);fps=float(v.get_avg_fps());hh,ww=smart_resize(*v[0].shape[:2],factor=28,min_pixels=4*28*28,max_pixels=384**2);del v
   v=decord.VideoReader(str(raw),width=ww,height=hh,num_threads=2);ix=np.linspace(0,n-1,nf).round().astype(np.int64);frames=v.get_batch(ix).asnumpy();del v
   dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,frames=frames,source_indices=ix,source_fps=np.float64(fps),source_total_frames=np.int64(n),duration_seconds=np.float64(n/fps),effective_fps=np.float64(nf/(n/fps)))
   record={'video':x['video'],'source_SHA256':sha(raw),'archive_SHA256':sha(dest),'pixels_SHA256':hashlib.sha256(frames.tobytes()).hexdigest(),'source_indices':ix.tolist(),'frames':nf,'shape':list(frames.shape),'geometry_group':group,'actual_duration_seconds':n/fps,'bytes':dest.stat().st_size};save(rec,record)
  save(ROOT/'data/ar22/frame_receipts'/('AR39CAL_'+pathlib.Path(x['video']).stem+'.json'),{**record,'raw_video_SHA256':record['source_SHA256'],'scope':'AR39 common WorldSense calibration input adapter receipt'})
  human=next(c['value']for c in x['conversations']if c['from']=='human');answer=next(c['value']for c in x['conversations']if c['from']=='gpt');prompt=human.replace('<image>\n','').replace('video and audio','video')
  item={'id':x['id'],'source_id':'WorldSense/'+x['video'],'dataset':'WorldSense','video':'AR39CAL_'+x['video'],'original_video':x['video'],'media_path':str(raw),'frame_store':str(dest),'sample_frames':nf,'max_pixels':384**2,'duration_group':group,'geometry_group':group,'prompt':prompt,'answer':answer,'options':['A','B','C','D'],'task_type':x['task_type'],'query_reservation_tokens':288,'budget_mib':64}
  rows.append(item);records.append(record);save(ROOT/'data/ar39/PREPARE_PROGRESS.json',{'videos':len(rows),'expected':64,'records':records});print(json.dumps({'prepared':i+1,'source':x['video'],'frames':nf}),flush=True)
 save(ROOT/'manifests/ar39/WorldSense_CAL64.json',rows);save(ROOT/'data/ar39/INPUT_AUDIT.json',{'completed':True,'source_manifest_SHA256':sha(source),'sources':64,'WorldSense_only':True,'question_labels_from_WorldSense_only':True,'same_pixels_for_both_backbones':True,'geometry_counts':{g:sum(x['geometry_group']==g for x in rows)for g in ['short','medium','long']},'records':records,'bytes':sum(x['bytes']for x in records)})
if __name__=='__main__':main()
