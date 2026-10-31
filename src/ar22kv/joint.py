"""Joint physical rate/innovation alternatives with rate-independent references.

Reference/ranking decisions use original feature geometry; closed-loop residuals
still correct the actually decoded anchor. This separates the fixed reference
decision from layer rates, enabling depth/occupancy conditional distortion.
"""
import contextlib
import json
import math
import time

import torch

from ar21kv.saturating import SaturatingInnovations
from r14kv.common import ROOT, sha
from r15kv.persistent_registry import Arena
from .matching import fixed_reference
from .planner import allocate


class JointTemporal(SaturatingInnovations):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = ROOT/self.policy['conditional_file']
        self.conditional = json.loads(path.read_text())
        assert self.conditional['completed']
        assert self.conditional['bank_completion_SHA256'] == self.package_sha256
        self.conditional_sha = sha(path)
        self._base_gop = int(self.policy.get('gop', 8))
        self.metadata.update(conditional_proxy_SHA256=self.conditional_sha,
                             reference_policy='rate-independent raw feature local correspondence; decoded-anchor innovations',
                             calibration_planner_storage='CPU metadata; any sensitivity vector copied to GPU only within encoding workspace')

    def prepare_request(self, meta, registry):
        self._request_group = self.context.get('duration_group', 'short')
        self._content_factors = None
        self._content_observation = None
        return super().prepare_request(meta, registry)

    def make_plan(self, meta):
        if self._hybrid_dense:
            return super().make_plan(meta)
        started = time.perf_counter()
        m, p = len(self._video_list), self._h*self._w
        self._end = self._video_list[-1]+1
        q, text = meta.capacity-self._end, len(self._text_list)
        dtype = torch.int16 if max(meta.position_max) <= 32767 else torch.int32
        global_bytes = 3*meta.capacity*torch.empty((), dtype=dtype).element_size()+256+4*m+4*text
        fixed = self.static_bytes+global_bytes+36*(2*meta.capacity+1024*text)
        rank = self.policy.get('ranking', 'norm')
        floors = [(1, 1)]*36
        parent = None
        if self.policy.get('guarded_parent_depth', False):
            try:
                self.policy['gop'] = self._base_gop
                parent = super().make_plan(meta)
                floors = [(max(1, k-1), max(1, v-1))
                          for k, v in zip(parent['layer_R_K'], parent['layer_R_V'])]
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
                                loss += anchors*c['anchor'][depth-1]+targets*factor*c['target'][fi][depth-1]+q*c['query'][depth-1]
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
                'planning_CPU_ms': (time.perf_counter()-started)*1000,
                'guarded_parent_depth': bool(self.policy.get('guarded_parent_depth', False)),
                'content_adaptive': self._content_observation is not None,
                'content_observation': self._content_observation,
                'representation': 'all sampled native positions; one-hop decoded-anchor innovations; joint per-layer rates and occupancy',
                'reference_rate_independent': True, 'hybrid_dense': False}

    @torch.no_grad()
    def observe_prefix(self, pre, value):
        """Profile current layer0 before any persistent layer arena is allocated.

        Relative copy/anchor distortion measures content dependence while
        canceling the scale of the calibrated sensitivity proxy. It uses no
        question, future layer trace, or whole-history CPU/dense checkpoint.
        """
        if not self.policy.get('content_adaptive', False) or self._hybrid_dense:
            return False
        from .calibration import prefixes
        before = self.plan
        raw = {'K': pre[0].float(), 'V': value[0].float()}
        table = self.tables[0]
        p = self._h*self._w
        ordinal = torch.arange(len(self._video_list), device=value.device)
        factors, diagnostics = {}, {}
        rank = self.policy.get('ranking', 'norm')
        count = int(self.policy.get('content_probe_targets', 48))
        for gop in self.policy.get('gop_choices', [2, 4, 8, 16]):
            anchors = ordinal[(ordinal//p) % gop == 0]
            targets = ordinal[(ordinal//p) % gop != 0]
            anchors = anchors[torch.linspace(0, len(anchors)-1, min(count, len(anchors)), device=value.device).long()]
            targets = targets[torch.linspace(0, len(targets)-1, min(count, len(targets)), device=value.device).long()]
            importance = None if rank == 'norm' else [torch.tensor(
                self.conditional['ranking_importance'][s][0], device=value.device, dtype=torch.float32) for s in ['K', 'V']]
            codes, _ = fixed_reference(raw, self._vid_gpu, targets,
                    [table['K_scale'][0], table['V_scale'][0]], self._h, self._w, gop, importance)
            ry = (targets[None]//self._w % self._h+codes.long()//3-1).clamp(0, self._h-1)
            rx = (targets[None] % self._w+codes.long() % 3-1).clamp(0, self._w-1)
            refs = self._vid_gpu[(targets[None]//p//gop*gop*p+ry*self._w+rx)].long()
            factors[str(gop)], diagnostics[str(gop)] = {}, {}
            for side in ['K', 'V']:
                anchor_values = raw[side][:, self._vid_gpu[anchors].long()]
                ref_values = raw[side].gather(1, refs[..., None].expand(-1, -1, 128))
                decoded_anchor = prefixes(anchor_values, table, side, 0)[-1].float()
                decoded_ref = prefixes(ref_values, table, side, 0)[-1].float()
                metric = table[side+'_scale'][0].float().square().reciprocal()
                if importance is not None:
                    metric = metric*importance[['K', 'V'].index(side)]
                anchor_loss = float(((anchor_values-decoded_anchor).square()*metric[:, None]).sum(-1).mean())
                copy_loss = float(((raw[side][:, self._vid_gpu[targets].long()]-decoded_ref).square()*metric[:, None]).sum(-1).mean())
                c = self.conditional['groups'][self._request_group][rank][str(gop)][side][0]
                relative = (copy_loss/max(anchor_loss, 1e-12))/(c['target'][0][5]/max(c['anchor'][5], 1e-12))
                factor = max(.25, min(4., relative))
                factors[str(gop)][side] = factor
                diagnostics[str(gop)][side] = {'measured_copy_loss': copy_loss,
                         'measured_anchor_loss': anchor_loss, 'conditional_target_multiplier': factor}
        self._content_factors = factors
        self._content_observation = {'observed_layer': 0, 'targets_per_G_per_head': count,
             'scope': 'current native layer0 before any layer arena; no query or future-layer state',
             'diagnostics': diagnostics, 'multiplier_range': [.25, 4.]}
        self.plan = self.make_plan(self.meta)
        self.plan['initial_geometry_plan_gop'] = before['gop']
        self.plan['planning_CPU_ms'] += before['planning_CPU_ms']
        return True

    @contextlib.contextmanager
    def _layer(self, layer):
        plan = self.plan
        if self._hybrid_dense:
            yield
            return
        self.plan = {**plan, 'innovation_slots': plan['layer_innovation_slots'][layer],
                     'encoded_slots_per_head': plan['layer_encoded_slots_per_head'][layer]}
        try:
            yield
        finally:
            self.plan = plan

    @torch.no_grad()
    def append(self, layer, state, kpost, value, pre=None):
        with self._layer(layer):
            if self._hybrid_dense or state['n']:
                return super().append(layer, state, kpost, value, pre)
            assert pre is not None
            n = value.shape[-2]
            assert n == self._end
            table = self.tables[0]
            raw = {'K': pre[0].float(), 'V': value[0].float()}
            a, nr = self.plan['anchor_tokens'], self.plan['innovation_slots']
            aslots = torch.arange(a, device=value.device)
            aseq = self._vid_gpu.long()[torch.tensor(self._anchor_ord, device=value.device)]
            decoded = {}
            for side in ['K', 'V']:
                self._innovation(layer, state, raw[side][:, aseq]-table[side+'_mean'][layer].float()[:, None], aslots, side)
                decoded[side] = torch.zeros((2, n, 128), device=value.device, dtype=torch.bfloat16)
                decoded[side][:, aseq] = self._read_slots(layer, state, side, aslots[None].expand(2, -1), True)
            order = torch.arange(len(self._video_list), device=value.device)
            targets = order[(order//(self._h*self._w)) % int(self.policy['gop']) != 0]
            seq = self._vid_gpu.long()[targets]
            if len(targets):
                importance = None
                if self.policy.get('ranking', 'norm') == 'sensitivity':
                    importance = [torch.tensor(self.conditional['ranking_importance'][s][layer],
                                               device=value.device, dtype=torch.float32) for s in ['K', 'V']]
                codes, scores = fixed_reference(raw, self._vid_gpu, targets,
                        [table['K_scale'][layer], table['V_scale'][layer]],
                        self._h, self._w, int(self.policy['gop']), importance)
                refs = self._reference_indices(targets, codes)
                state['T_code'][:, seq] = codes
                if nr:
                    picked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :nr]
                    picked = picked.sort(1).values
                    chosen, ref = seq[picked], refs.gather(1, picked)
                    heads = torch.arange(2, device=value.device)[:, None]
                    state['T_code'][heads, chosen] = codes.gather(1, picked) | 128
                    slots = torch.arange(a, a+nr, device=value.device)
                    for side in ['K', 'V']:
                        current = raw[side].gather(1, chosen[..., None].expand(-1, -1, 128))
                        prediction = decoded[side].gather(1, ref[..., None].expand(-1, -1, 128)).float()
                        self._innovation(layer, state, current-prediction, slots, side)
                state['temporal_targets'] = len(targets)
                state['temporal_innovation_heads'] = 2*nr
                state['temporal_copy_heads'] = 2*(len(targets)-nr)
                state['distortion_raw_reference_score_sum'] = float(scores.sum())
            text = self._text_gpu.long()
            state['text_FP_K'].copy_(kpost[0][:, text])
            state['text_FP_V'].copy_(value[0][:, text])
            state['text_filled'] = len(self._text_list)
            state['n'] = n
            return state

    def decode(self, layer, state, positions, inv_freq, attention_scaling=1.):
        with self._layer(layer):
            return super().decode(layer, state, positions, inv_freq, attention_scaling)

    def finish_request(self):
        super().finish_request()
        self.policy['gop'] = self._base_gop
