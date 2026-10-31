"""Equal-paid metadata control: stored old global BF16 stats, no video fit."""
from ar18kv.video_stats import VideoStatsFused
from ar18kv.fused import FusedProduct
from ar18kv.hybrid import DenseFirst
from ar17kv.text_protection import PaidPrefixText
class StaticStatFused(VideoStatsFused):
    def make_plan(self,meta):
        p=super().make_plan(meta);p.update(statistic_source='old fixed package parameters copied to paid BF16 request fields; no video statistic fit',static_stat_control=True);return p
    def append(self,l,state,kpost,v,pre=None):
        assert pre is not None;self._current_state=state
        try:
            if state['n']==0:
                for side in ['K','V']:
                    state[side+'_video_mean'].copy_(self.tables[0][side+'_mean'][l]);state[side+'_video_scale'].copy_(self.tables[0][side+'_scale'][l])
            return FusedProduct.append(self,l,state,kpost,v,pre)
        finally:self._current_state=None
class ProtectedStaticStats(PaidPrefixText,StaticStatFused):pass
class HybridStaticStats(DenseFirst,ProtectedStaticStats):pass
