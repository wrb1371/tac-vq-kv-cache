"""Instrumentation only: bounded native states and original layer operation profiles."""
import json
import pathlib
import torch
from ar19kv.stable import HybridStablePacked
from ar19kv.packed_sparse import HybridPackedSparse
from r14kv.common import ROOT


class Capture:
    def set_context(self, **kwargs):
        self._capture_item = kwargs['item_id']
        self._capture_positions = kwargs['positions'].detach().cpu()
        return super().set_context(**kwargs)

    def append(self, l, state, kpost, v, pre=None):
        first = state['n'] == 0
        base = ROOT / 'artifacts/ar20/captures'
        base.mkdir(parents=True, exist_ok=True)
        name = str(self.budget_bytes // 2**20) + '_' + self._capture_item + '_' + str(l)
        target = base / (name + '.pt')
        profile = first and l == 0 and not target.exists() and not self._hybrid_dense
        if profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                result = super().append(l, state, kpost, v, pre)
            events = [{'op': x.key, 'calls': x.count, 'cpu_us': x.cpu_time_total, 'cuda_us': x.device_time_total} for x in prof.key_averages()]
            out = ROOT / 'data/ar20' / (name + '_encoder_profile.json')
            out.write_text(json.dumps(events, indent=2) + '\n')
        else:
            result = super().append(l, state, kpost, v, pre)
        if first and l in [0, 18, 35] and not target.exists() and not self._hybrid_dense:
            total = sum(p.stat().st_size for p in base.glob('*.pt'))
            if total < 400 * 2**20:
                tensors = {k: x.detach().cpu().contiguous() for k, x in state.items() if torch.is_tensor(x)}
                snapshot = {'layer': l, 'policy': self.policy, 'plan': self.plan, 'state': tensors, 'n': state['n'], 'end': self._end, 'nt': self._nt, 'height': self._h, 'width': self._w, 'video_list': self._video_list, 'text_list': self._text_list, 'anchor_ord': self._anchor_ord, 'raw_pre_K': pre[0].detach().cpu(), 'raw_V': v[0].detach().cpu(), 'post_K': kpost[0].detach().cpu(), 'positions': self._capture_positions, 'item_id': self._capture_item}
                torch.save(snapshot, target)
        return result


class CaptureStable(Capture, HybridStablePacked):
    pass


class CapturePacked(Capture, HybridPackedSparse):
    pass
