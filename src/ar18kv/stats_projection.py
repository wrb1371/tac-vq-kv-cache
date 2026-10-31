"""Fixed combination, proxy evaluated with actual stored per-video decoder stats."""
import torch
from ar18kv.video_stats import VideoStatsFused
from ar18kv.hybrid import DenseFirst
from ar17kv.text_protection import PaidPrefixText
from r14kv.fitting import nearest
class StatsProjectionFused(VideoStatsFused):
    def _choose(self,l,side,raw,width):
        if side=='K':return super()._choose(l,side,raw,width)
        t=self.tables[0];sample=raw.index_select(1,self._sample.long());space,rho=self._normalized(l,side,sample);mean,scale=self._stats(l,side);metric=t['V_output_metric'][l].float();assert metric.ndim==3;possible=[];detail=[]
        for bank,spec in enumerate(self.package['bank_specs']):
            d=spec['dim'];g=128//d;step=g*spec['bits']//8;r=min(spec['Rmax'],width//step)
            if r<1:continue
            z=space.reshape(1,-1,d).clone();decoded=torch.zeros_like(z)
            for j in range(r):
                cb=t[f'{side}_books_b{bank}'][j].float()[None];ix=nearest(z,cb);chosen=torch.gather(cb,1,ix[...,None].expand(-1,-1,d));z-=chosen;decoded+=chosen
            pred=(torch.matmul(decoded.reshape_as(space),t['H'].T)*rho.float()*scale+mean).bfloat16().float();error=pred-sample.float();loss=float((error*torch.matmul(error,metric)).sum());assert 0<=loss<float('inf');possible.append((loss,r*spec['clusters'],bank,r));detail.append({'bank':bank,'depth':r,'proxy_loss':loss,'raw_squared_error':float(error.square().sum())})
        assert possible;_,_,bank,r=min(possible)
        import json,sys
        from r14kv.common import ROOT
        rid=sys.argv[sys.argv.index('--run-id')+1]
        with (ROOT/'data/runs'/rid/'output_metric_choices.jsonl').open('a') as f:f.write(json.dumps({'item_id':self._trace_item,'layer':l,'metric_kind':'full_with_stored_video_stats','choices':detail,'selected_bank':bank,'selected_depth':r})+'\n')
        return bank,r
class ProtectedStatsProjection(PaidPrefixText,StatsProjectionFused):pass
class HybridStatsProjection(DenseFirst,ProtectedStatsProjection):pass
