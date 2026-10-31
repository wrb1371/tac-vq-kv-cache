"""Same published-port trim arithmetic, with device indices instead of CPU fences."""
import torch

from ar19kv.selector_fast import FastVideoPrune


def asynchronous_mixed(K, V, budget, alpha):
    n = len(K)
    b = min(int(budget), n)
    device = K.device
    if b <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if b >= n:
        return torch.arange(n, device=device)
    a = min(1.0, max(0.0, float(alpha)))
    k, v = K.float(), V.float()
    dk = (k - k.mean(0, keepdim=True)).square().sum(-1)
    dv = (v - v.mean(0, keepdim=True)).square().sum(-1)
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    chosen = []
    for step in range(b):
        score = (a * dk + (1 - a) * dv).masked_fill(mask, -1)
        j = score.argmax().reshape(1)
        chosen.append(j)
        mask.scatter_(0, j, True)
        nk = (k - k.index_select(0, j)).square().sum(-1)
        nv = (v - v.index_select(0, j)).square().sum(-1)
        if step == 0:
            dk, dv = nk, nv
        else:
            dk, dv = torch.minimum(dk, nk), torch.minimum(dv, nv)
    return torch.cat(chosen).sort().values


class AsyncVideoPrune(FastVideoPrune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.selector == "cords"
        self.official.select_bicriteria_farthest_mixed = asynchronous_mixed
        self.metadata["implementation_optimization"] = (
            "identical incremental trim recurrence; device argmax indices avoid per-seed CPU synchronization"
        )
