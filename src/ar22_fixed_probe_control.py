"""Fixed occupancy catalog with the identical zero-occupancy content probe.

The main frozen implementation uses target[0] as the zero-correction reference
for its content multiplier. A one-entry occupancy catalog changes that meaning.
This control supplies the original complete table to the observer, and the
fixed-I table exclusively to planning. Encoding/decoding remain inherited.
"""
import json

from ar22kv.joint import JointTemporal
from r14kv.common import ROOT, sha


class FixedOccupancyStableProbe(JointTemporal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed_conditional = self.conditional
        path = ROOT/self.policy['observer_conditional_file']
        self._observer_conditional = json.loads(path.read_text())
        assert self._observer_conditional['fractions'][0] == 0.
        assert self._fixed_conditional['fractions'] == [.125]
        source_index = self._observer_conditional['fractions'].index(.125)
        for group, rankings in self._fixed_conditional['groups'].items():
            for ranking, spacings in rankings.items():
                for spacing, sides in spacings.items():
                    for side, layers in sides.items():
                        for layer, row in enumerate(layers):
                            reference = self._observer_conditional['groups'][group][ranking][spacing][side][layer]
                            assert row['target'][0] == reference['target'][source_index]
                            assert row['anchor'] == reference['anchor'] and row['query'] == reference['query']
        self.metadata.update(observer_conditional_SHA256=sha(path),
                             fixed_occupancy_probe_reference='full originalf0 copy loss; no occupancy/probe normalization confounding')

    def make_plan(self, meta):
        saved = self.conditional
        self.conditional = getattr(self, '_fixed_conditional', saved)
        try:
            return super().make_plan(meta)
        finally:
            self.conditional = saved

    def observe_prefix(self, pre, value):
        saved = self.conditional
        self.conditional = self._observer_conditional
        try:
            return super().observe_prefix(pre, value)
        finally:
            self.conditional = saved
