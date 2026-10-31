"""Observe current media layer0 before allocating the joint persistent body."""
from ar20kv.attention import KernelCache


class ContentPlannedCache(KernelCache):
    def encode_after_attention(self, layer):
        if layer == 0 and not self.states and not self.codec._hybrid_dense and self.codec.policy.get('content_adaptive', False):
            assert self.pre.get(0) is not None
            value = self.incoming[0][1] if 0 in self.incoming else self.value_cache[0]
            changed = self._time(lambda: self.codec.observe_prefix(self.pre[0], value), 'content_probe')
            if changed:
                assert not self.states
                self.plan = self.codec.plan
                self.registry.observe('content_plan_frozen_before_first_layer_arena')
        return super().encode_after_attention(layer)

    def receipt(self):
        result = super().receipt()
        result.update(content_probe_event_ms=sum(start.elapsed_time(end)
                      for kind, start, end in self.events if kind == 'content_probe'),
                      content_probe_scope='current native layer0, temporary workspace, before persistent body allocation; event span includes host planner gaps',
                      content_plan_before_first_persistent_layer_arena=True)
        return result
