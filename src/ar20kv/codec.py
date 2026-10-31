"""Independent implementation variants; each retains the frozen AR19 rate rule."""
import torch
from ar19kv.stable import HybridStablePacked
from ar19kv.packed_sparse import HybridPackedSparse
from r14kv.fitting import nearest
from .kernels import read_slots, encode_rvq, match9, compose


class DeadWorkRemoved:
    @torch.no_grad()
    def _innovation(self, l, state, raw, slots, kind):
        # Stable/Packed append never uses the former reconstruction return value.
        # Keep chunk, norm, GEMM, distance and update order unchanged.
        t = self.tables[0]
        g = 8
        for b in range(0, len(slots), 512):
            x = raw[:, b:b + 512].float()
            nn = x.shape[1]
            u = x / t[kind + '_scale'][l].float()[:, None]
            rho = u.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6).bfloat16()
            where = slots[b:b + nn]
            state[kind + '_norm'][:, where] = rho
            z = ((u / rho.float()) @ t['H']).reshape(1, 2 * nn * g, 16)
            for rr in range(state[kind + '_index'].shape[2]):
                cb = t[kind + '_books'][rr][None].float()
                ix = nearest(z, cb)
                state[kind + '_index'][:, where, rr] = ix.reshape(2, nn, g).to(torch.uint8)
                z -= torch.gather(cb, 1, ix[..., None].expand(-1, -1, 16))


class FusedRead:
    def _read_slots(self, l, state, kind, slots, independent):
        return read_slots(l, state, kind, slots, independent, self.tables[0])


class FusedRVQ:
    @torch.no_grad()
    def _innovation(self, l, state, raw, slots, kind):
        t = self.tables[0]
        for b in range(0, len(slots), 512):
            x = raw[:, b:b + 512].float()
            nn = x.shape[1]
            u = x / t[kind + '_scale'][l].float()[:, None]
            rho = u.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6).bfloat16()
            where = slots[b:b + nn]
            state[kind + '_norm'][:, where] = rho
            z = ((u / rho.float()) @ t['H']).reshape(1, 2 * nn * 8, 16)
            encode_rvq(z, t[kind + '_books'], state[kind + '_index'], where)


class DeadStable(DeadWorkRemoved, HybridStablePacked): pass
class DeadPacked(DeadWorkRemoved, HybridPackedSparse): pass
class ReadStable(FusedRead, HybridStablePacked): pass
class ReadPacked(FusedRead, HybridPackedSparse): pass
class ComboStable(FusedRead, DeadWorkRemoved, HybridStablePacked): pass
class ComboPacked(FusedRead, DeadWorkRemoved, HybridPackedSparse): pass
class RVQStable(FusedRead, FusedRVQ, HybridStablePacked): pass
class RVQPacked(FusedRead, FusedRVQ, HybridPackedSparse): pass


class MatchedAppend:
    @torch.no_grad()
    def append(self, l, state, kpost, v, pre=None):
        if self._hybrid_dense:
            return super().append(l, state, kpost, v, pre)
        assert pre is not None
        start = state['n']; n = v.shape[-2]; t = self.tables[0]
        raw = {'K': pre[0].float(), 'V': v[0].float()}
        a = self.plan['anchor_tokens']; nr = self.plan['innovation_slots']
        heads = torch.arange(2, device='cuda')[:, None]
        if start == 0:
            assert n == self._end
            aslots = torch.arange(a, device='cuda')
            aseq = self._vid_gpu.long()[torch.tensor(self._anchor_ord, device='cuda')]
            for kind in ['K', 'V']:
                self._innovation(l, state, raw[kind][:, aseq] - t[kind + '_mean'][l].float()[:, None], aslots, kind)
            order = torch.arange(len(self._video_list), device='cuda')
            targets = order[(order // (self._h * self._w)) % int(self.policy['gop']) != 0]
            seq = self._vid_gpu.long()[targets]
            if len(targets):
                decoded = {}
                for kind in ['K', 'V']:
                    z = torch.zeros((2, n, 128), device='cuda', dtype=torch.bfloat16)
                    z[:, aseq] = self._read_slots(l, state, kind, aslots[None].expand(2, -1), True)
                    decoded[kind] = z
                codes, best = match9(raw, decoded, self._vid_gpu, targets, [t['K_scale'][l], t['V_scale'][l]], self._h, self._w, int(self.policy['gop']))
                refs = self._reference_indices(targets, codes)
                state['T_code'][:, seq] = codes
                if nr:
                    picked = torch.argsort(best, dim=1, descending=True, stable=True)[:, :nr]
                    if not self._stable_order:
                        picked = picked.sort(1).values
                    chosen = seq[picked]; selected = refs.gather(1, picked)
                    slots = torch.arange(a, a + nr, device='cuda')
                    state['T_code'][heads, chosen] = codes.gather(1, picked) | 128
                    for kind in ['K', 'V']:
                        cur = raw[kind].gather(1, chosen[..., None].expand(-1, -1, 128))
                        pred = decoded[kind].gather(1, selected[..., None].expand(-1, -1, 128)).float()
                        self._innovation(l, state, cur - pred, slots, kind)
                    if self._stable_order:
                        integer_order = torch.argsort(picked, dim=1, stable=True)
                        for kind in ['K', 'V']:
                            ix = state[kind + '_index'][:, a:a+nr]; norm = state[kind + '_norm'][:, a:a+nr]
                            ix.copy_(ix.gather(1, integer_order[..., None, None].expand_as(ix)))
                            norm.copy_(norm.gather(1, integer_order[..., None].expand_as(norm)))
                state['temporal_targets'] = len(targets)
                state['temporal_innovation_heads'] = 2 * nr
                state['temporal_copy_heads'] = 2 * (len(targets) - nr)
                state['distortion_copy_before_innovation_sum'] = float(best.sum())
            text = self._text_gpu.long()
            state['text_FP_K'].copy_(kpost[0][:, text]); state['text_FP_V'].copy_(v[0][:, text]); state['text_filled'] = len(self._text_list)
        else:
            slots = torch.arange(a+nr+start-self._end, a+nr+start-self._end+n, device='cuda')
            assert int(slots[-1]) < self.plan['encoded_slots_per_head']
            for kind in ['K', 'V']:
                self._innovation(l, state, raw[kind] - t[kind + '_mean'][l].float()[:, None], slots, kind)
        state['n'] += n
        return state


class FullDecode:
    def decode(self, l, state, positions, inv_freq, attention_scaling=1.):
        if self._hybrid_dense:
            return super().decode(l, state, positions, inv_freq, attention_scaling)
        a = self.plan['anchor_tokens']; nr = self.plan['innovation_slots']; nq = state['n'] - self._end
        arrays = []
        for start, count, independent in [(0, a, True), (a, nr, False), (a+nr, nq, True)]:
            slots = torch.arange(start, start+count, device='cuda')[None].expand(2, -1)
            for kind in ['K', 'V']:
                arrays.append(self._read_slots(l, state, kind, slots, independent) if count else torch.empty((2, 0, 128), device='cuda', dtype=torch.bfloat16))
        return compose(self, state, positions, inv_freq, attention_scaling, arrays)


class MatchStable(MatchedAppend, FusedRead, DeadWorkRemoved, HybridStablePacked): _stable_order = True
class MatchPacked(MatchedAppend, FusedRead, DeadWorkRemoved, HybridPackedSparse): _stable_order = False
class FullStable(FullDecode, MatchedAppend, FusedRead, DeadWorkRemoved, HybridStablePacked): _stable_order = True
class FullPacked(FullDecode, MatchedAppend, FusedRead, DeadWorkRemoved, HybridPackedSparse): _stable_order = False
class FastStable(FullDecode, MatchedAppend, FusedRead, FusedRVQ, HybridStablePacked): _stable_order = True
class FastPacked(FullDecode, MatchedAppend, FusedRead, FusedRVQ, HybridPackedSparse): _stable_order = False
