"""Rate-independent local reference decisions, optional bounded sensitivity."""
import torch
import triton
import triton.language as tl

from ar21kv.dynamic_kernels import match9


@triton.jit(do_not_specialize=['N', 'T', 'H', 'W', 'GOP'])
def _weighted(K, V, Vid, SK, SV, WK, WV, Targets, Codes, Best, N, T, H, W, GOP):
    target, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, 128)
    ordinal = tl.load(Targets+target)
    seq = tl.load(Vid+ordinal)
    frame, y, x = ordinal//(H*W), ordinal//W % H, ordinal % W
    k = tl.load(K+(head*N+seq)*128+d)
    v = tl.load(V+(head*N+seq)*128+d)
    sk = tl.load(SK+head*128+d).to(tl.float32)
    sv = tl.load(SV+head*128+d).to(tl.float32)
    wk = tl.load(WK+head*128+d).to(tl.float32)
    wv = tl.load(WV+head*128+d).to(tl.float32)
    best, code = float('inf'), 4
    for c in tl.static_range(9):
        yy = tl.minimum(tl.maximum(y+c//3-1, 0), H-1)
        xx = tl.minimum(tl.maximum(x+c % 3-1, 0), W-1)
        ref = tl.load(Vid+frame//GOP*GOP*H*W+yy*W+xx)
        a = tl.load(K+(head*N+ref)*128+d)
        b = tl.load(V+(head*N+ref)*128+d)
        ek, ev = (k-a)/sk, (v-b)/sv
        loss = tl.sum((ek*ek)*wk+(ev*ev)*wv, 0)/128.
        take = loss < best
        code = tl.where(take, c, code)
        best = tl.minimum(best, loss)
    tl.store(Codes+head*T+target, code.to(tl.uint8))
    tl.store(Best+head*T+target, best)


def fixed_reference(raw, video_indices, targets, scales, h, w, gop, importance=None):
    raw = {side: raw[side].contiguous() for side in ['K', 'V']}
    assert video_indices.is_cuda and targets.is_cuda
    if importance is None:
        return match9(raw, raw, video_indices, targets, scales, h, w, gop)
    count, n = len(targets), raw['K'].shape[1]
    codes = torch.empty((2, count), dtype=torch.uint8, device=raw['K'].device)
    scores = torch.empty((2, count), dtype=torch.float32, device=raw['K'].device)
    _weighted[(count, 2)](raw['K'], raw['V'], video_indices, *scales, *importance,
                         targets, codes, scores, n, count, h, w, gop,
                         num_warps=4, enable_fp_fusion=False)
    return codes, scores
