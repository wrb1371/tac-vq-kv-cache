"""Strict planning attribution; the emitted format and reader stay unchanged."""
import copy
import hashlib
import json
import os

from ar22kv.joint import JointTemporal

DECODE_CHECKS = []


class CheckedDecode:
    def decode(self, layer, state, positions, inv_freq, attention_scaling=1.):
        observed = super().decode(layer, state, positions, inv_freq, attention_scaling)
        if os.environ.get('AR24_INDEPENDENT_DECODE_GATE') == '1' and layer in [0, 18, 35]:
            done = getattr(self, '_ar24_decode_checked', set())
            if layer not in done:
                from ar19kv.packed_sparse import HybridPackedSparse
                slow = object.__new__(HybridPackedSparse)
                slow.__dict__.update(self.__dict__)
                if not self._hybrid_dense:
                    slow.plan = {**self.plan,
                        'innovation_slots': self.plan['layer_innovation_slots'][layer],
                        'encoded_slots_per_head': self.plan['layer_encoded_slots_per_head'][layer]}
                expected = slow.decode(layer, state, positions, inv_freq, attention_scaling)
                diffs = [int((a != b).sum()) for a,b in zip(expected,observed)]
                assert diffs == [0,0], (type(self).__name__,layer,diffs)
                DECODE_CHECKS.append({'class':type(self).__name__,'layer':layer,
                                      'BF16_differences':diffs})
                done.add(layer)
                self._ar24_decode_checked = done
        return observed


class AdditiveMarginals(CheckedDecode, JointTemporal):
    """Remove the rate/occupancy interaction, with the same feasible catalog.

    d_sep(r,f) = d(r,f0) + d(rmax,f) - d(rmax,f0), f0=1/8, rmax=6.
    Only make_plan sees this separable target objective. The first-layer
    observer uses the exact original f=0 reference and original calibration.
    Negative anchored surrogate cells, if any, are retained rather than
    clipped: clipping would reintroduce an interaction. DP supports finite
    signed costs. The original conditional table is never changed on disk.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._separable = copy.deepcopy(self.conditional)
        ref = self.conditional['fractions'].index(.125)
        values = []
        for group, rankings in self._separable['groups'].items():
            for rank, spacings in rankings.items():
                for gop, sides in spacings.items():
                    for side,layers in sides.items():
                        for layer,c in enumerate(layers):
                            original = self.conditional['groups'][group][rank][gop][side][layer]
                            c['target'] = [[original['target'][ref][r] + row[5]
                                            - original['target'][ref][5]
                                            for r in range(6)]
                                           for row in original['target']]
                            values.extend(v for row in c['target'] for v in row)
        encoded = json.dumps(self._separable,sort_keys=True,separators=(',',':')).encode()
        self._objective_sha = hashlib.sha256(encoded).hexdigest()
        self.metadata.update(ar24_objective='additive depth and occupancy marginals; no interaction',
            ar24_objective_SHA256=self._objective_sha,
            ar24_observer='original conditional f=0 reference unchanged',
            ar24_negative_anchored_target_cells=sum(x<0 for x in values))

    def make_plan(self,meta):
        original = self.conditional
        self.conditional = getattr(self,'_separable',original)
        try:
            plan = super().make_plan(meta)
        finally:
            self.conditional = original
        plan.update(ar24_objective='separable_additive_marginals_fref_1of8_rref_6',
                    ar24_objective_SHA256=getattr(self,'_objective_sha',None))
        return plan
