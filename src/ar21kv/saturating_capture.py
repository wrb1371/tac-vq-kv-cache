import json,pathlib,torch
from r14kv.common import ROOT
from .saturating import SaturatingInnovations
class CaptureOccupancy:
    def set_context(self,**kwargs):
        self._capture_item=kwargs['item_id'];self._capture_positions=kwargs['positions'].detach().cpu();return super().set_context(**kwargs)
    def append(self,l,state,kpost,v,pre=None):
        first=state['n']==0;base=ROOT/'artifacts/ar21b/captures';base.mkdir(parents=True,exist_ok=True);name=self._capture_item+'_'+str(l);target=base/(name+'.pt');profile=first and l==0 and not target.exists() and not self._hybrid_dense
        if profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:result=super().append(l,state,kpost,v,pre)
            events=[{'op':e.key,'calls':e.count,'CPU_us':e.cpu_time_total,'CUDA_us':e.device_time_total} for e in prof.key_averages()];(ROOT/'data/ar21b'/(name+'_encoder_profile.json')).write_text(json.dumps(events,indent=2)+'\n')
        else:result=super().append(l,state,kpost,v,pre)
        if first and l in [0,18,35] and not target.exists() and not self._hybrid_dense:
            total=sum(p.stat().st_size for p in base.glob('*.pt'));assert total<400*2**20
            snap={'layer':l,'policy':self.policy,'plan':self.plan,'state':{k:x.detach().cpu().contiguous() for k,x in state.items() if torch.is_tensor(x)},'n':state['n'],'end':self._end,'nt':self._nt,'height':self._h,'width':self._w,'video_list':self._video_list,'text_list':self._text_list,'anchor_ord':self._anchor_ord,'raw_pre_K':pre[0].detach().cpu(),'raw_V':v[0].detach().cpu(),'post_K':kpost[0].detach().cpu(),'positions':self._capture_positions,'item_id':self._capture_item};torch.save(snap,target);assert sum(p.stat().st_size for p in base.glob('*.pt'))<=512*2**20
        return result

class CaptureSaturating(CaptureOccupancy,SaturatingInnovations):pass
