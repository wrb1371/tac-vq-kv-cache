"""Keep video encoding conservative; fuse only short-query RVQ searches."""
from .exact_gemm import ExactStable,ExactPacked,ExactGemm
from .codec import FusedRVQ


class QueryFast:
    def append(self,l,state,kpost,v,pre=None):
        self._short_query_RVQ=state['n']>0 and v.shape[-2]<=4
        try:return super().append(l,state,kpost,v,pre)
        finally:self._short_query_RVQ=False

    def _innovation(self,l,state,raw,slots,kind):
        fn=FusedRVQ._innovation if getattr(self,'_short_query_RVQ',False) else ExactGemm._innovation
        return fn(self,l,state,raw,slots,kind)


class QueryFastStable(QueryFast,ExactStable):pass
class QueryFastPacked(QueryFast,ExactPacked):pass
