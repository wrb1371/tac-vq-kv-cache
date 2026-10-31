"""Logging-only repair: the cache and codec share the same heterogeneous plan."""
from .cache import R16Cache
class R16CacheLogged(R16Cache):
    def receipt(self):
        plan=self.plan;self.plan={k:v for k,v in plan.items() if k not in ['modes','descriptions']}
        try:return super().receipt()
        finally:self.plan=plan
