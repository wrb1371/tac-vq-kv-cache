"""Versioned planning mutations; original AR22/23 source is never edited.

make_plan is copied from frozen JointTemporal with only: positive response
weights, a configurable parent-depth margin, and the physically defined
one-stage-equivalent budget-pressure trigger. Encoding/decoding are inherited.
"""
import json
import math
import time
import torch
from ar22kv.joint import JointTemporal
from ar21kv.saturating import SaturatingInnovations
from ar22kv.planner import allocate
from r14kv.common import ROOT,sha
from r15kv.persistent_registry import Arena
from .attribution import CheckedDecode

class SelectiveJoint(CheckedDecode, JointTemporal):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._response_weights=None
        self._response_sha=None
        filename=self.policy.get('response_weights_file')
        if filename:
            path=ROOT/filename
            weights=json.loads(path.read_text())
            assert weights['completed'] and not weights['task_QA_labels_used']
            self._response_weights=weights['weights']
            assert all(len(self._response_weights[s])==36 and all(0<w<1000 for w in self._response_weights[s]) for s in ['K','V'])
            self._response_sha=sha(path)
            self.metadata.update(response_weights_SHA256=self._response_sha,
                response_weight_scope='offline fixed-suffix single-side final-distribution response; host planner metadata, no new persistent GPU tensors')

    def make_plan(self, meta):
        if self._hybrid_dense:
            return SaturatingInnovations.make_plan(self, meta)
        started = time.perf_counter()
        m, p = len(self._video_list), self._h*self._w
        self._end = self._video_list[-1]+1
        q, text = meta.capacity-self._end, len(self._text_list)
        dtype = torch.int16 if max(meta.position_max) <= 32767 else torch.int32
        global_bytes = 3*meta.capacity*torch.empty((), dtype=dtype).element_size()+256+4*m+4*text
        fixed = self.static_bytes+global_bytes+36*(2*meta.capacity+1024*text)
        rank = self.policy.get('ranking', 'norm')
        variable_budget = self.budget_bytes-fixed
        pressure = 36864*(m+q)/max(1, variable_budget)
        margin = int(self.policy.get('guard_margin', 1))
        if self.policy.get('pressure_guard', False) and pressure > 32.:
            margin = 0
        floors = [(1, 1)]*36
        parent = None
        if self.policy.get('guarded_parent_depth', False):
            try:
                self.policy['gop'] = self._base_gop
                parent = SaturatingInnovations.make_plan(self, meta)
                minimum = int(self.policy.get('low_risk_min_depth', 1))
                assert self._response_weights is not None and minimum in [1, 2]
                floors = [(max(minimum, k-margin) if self._response_weights['K'][l] >= 1. else minimum,
                           max(minimum, v-margin) if self._response_weights['V'][l] >= 1. else minimum)
                          for l, (k, v) in enumerate(zip(parent['layer_R_K'], parent['layer_R_V']))]
            except RuntimeError as exc:
                if 'INFEASIBLE' not in str(exc):
                    raise
        best = None
        for gop in self.policy.get('gop_choices', [2, 4, 8, 16]):
            anchors = math.ceil(self._nt/gop)*p
            targets = m-anchors
            curves = self.conditional['groups'][self._request_group][rank][str(gop)]
            layer_options = []
            for layer in range(36):
                options = []
                for fi, fraction in enumerate(self.conditional['fractions']):
                    innovations = math.ceil(targets*fraction)
                    slots = anchors+innovations+q
                    for rk in range(floors[layer][0], 7):
                        for rv in range(floors[layer][1], 7):
                            loss = 0.
                            for side, depth in [('K', rk), ('V', rv)]:
                                c = curves[side][layer]
                                factor = self._content_factors[str(gop)][side] if self._content_factors else 1.
                                local = anchors*c['anchor'][depth-1]+targets*factor*c['target'][fi][depth-1]+q*c['query'][depth-1]
                                weight = self._response_weights[side][layer] if self._response_weights and self.policy.get('response_weighted_objective', True) else 1.
                                loss += weight*local
                            options.append({'bytes': slots*(16*(rk+rv)+8), 'loss': loss,
                                            'rK': rk, 'rV': rv, 'innovation_slots': innovations,
                                            'slots': slots, 'fraction_index': fi})
                layer_options.append(options)
            try:
                if self.policy.get('uniform_layers', False):
                    choices = []
                    for oi in range(len(layer_options[0])):
                        same = [layer[oi] for layer in layer_options]
                        cost = sum(x['bytes'] for x in same)
                        if cost <= self.budget_bytes-fixed:
                            choices.append((sum(x['loss'] for x in same), cost, oi, same))
                    if not choices:
                        raise RuntimeError('INFEASIBLE_BY_CONSERVATIVE_JOINT_OPTION_DP')
                    objective, cost, _, selected = min(choices, key=lambda x: x[:3])
                    report = {'objective': objective, 'actual_variable_bytes': cost,
                              'scope': 'complete finite uniform-across-layer joint enumeration'}
                else:
                    selected, report = allocate(layer_options, self.budget_bytes-fixed,
                                                quantum=int(self.policy.get('quantum_bytes', 4096)))
            except RuntimeError as exc:
                if 'INFEASIBLE' not in str(exc):
                    raise
                continue
            key = (report['objective'], fixed+report['actual_variable_bytes'], gop)
            if best is None or key < best[0]:
                best = (key, gop, selected, report)
        if best is None:
            raise RuntimeError('INFEASIBLE_BY_JOINT_PHYSICAL_OPTION_ORACLE')
        _, gop, selected, report = best
        self.policy['gop'] = gop
        self._anchor_ord = [i for i in range(m) if (i//p) % gop == 0]
        anchors = len(self._anchor_ord)
        descriptions = []
        for choice in selected:
            e = choice['slots']
            desc = [('K_index', (2, e, choice['rK'], 8), torch.uint8, 'payload'),
                    ('V_index', (2, e, choice['rV'], 8), torch.uint8, 'payload')]
            desc += [(side+'_norm', (2, e, 1), torch.bfloat16, 'metadata') for side in ['K', 'V']]
            desc += [('T_code', (2, meta.capacity), torch.uint8, 'metadata')]
            desc += [(side, (2, text, 128), torch.bfloat16, 'payload') for side in ['text_FP_K', 'text_FP_V']]
            descriptions.append(desc)
        actual = self.static_bytes+global_bytes+sum(Arena.bytes(x) for x in descriptions)
        assert actual == fixed+report['actual_variable_bytes'] <= self.budget_bytes
        return {'description': descriptions[0], 'descriptions': descriptions,
                'capacity': meta.capacity, 'R_K': selected[0]['rK'], 'R_V': selected[0]['rV'],
                'layer_R_K': [x['rK'] for x in selected], 'layer_R_V': [x['rV'] for x in selected],
                'position_dtype': str(dtype), 'group_dim': 16, 'groups': 8,
                'anchor_tokens': anchors, 'innovation_slots': selected[0]['innovation_slots'],
                'layer_innovation_slots': [x['innovation_slots'] for x in selected],
                'encoded_slots_per_head': selected[0]['slots'],
                'layer_encoded_slots_per_head': [x['slots'] for x in selected],
                'query_slots': q, 'media_original': m, 'protected_prefix_text_tokens': text,
                'gop': gop, 'ranking': rank, 'merged_video_grid': [self._nt, self._h, self._w],
                'estimate': {'codec_static_bytes': self.static_bytes, 'global_request_bytes': global_bytes,
                             'layer_arena_bytes': Arena.bytes(descriptions[0]),
                             'all_layer_arena_bytes': sum(Arena.bytes(x) for x in descriptions),
                             'resident_capacity_bytes': actual, 'kind': 'INTEGER_ESTIMATE_NOT_MEASUREMENT'},
                'conditional_proxy_SHA256': self.conditional_sha, 'joint_allocation': report,
                'ar24_response_weight_SHA256': self._response_sha,
                'ar24_pressure_ratio': pressure, 'ar24_effective_guard_margin': margin,
                'ar24_precision_risk_control': 'response-threshold selective floor with unchanged physical arena schema',
                'ar24_low_risk_min_depth': self.policy.get('low_risk_min_depth', 1),
                'planning_CPU_ms': (time.perf_counter()-started)*1000,
                'guarded_parent_depth': bool(self.policy.get('guarded_parent_depth', False)),
                'content_adaptive': self._content_observation is not None,
                'content_observation': self._content_observation,
                'representation': 'all sampled native positions; one-hop decoded-anchor innovations; joint per-layer rates and occupancy',
                'reference_rate_independent': True, 'hybrid_dense': False}
