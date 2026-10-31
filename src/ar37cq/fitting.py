"""Original CQ Fisher-weighted centroid fitting; assignments use ordinary L2."""

import torch


@torch.no_grad()
def nearest(x, z, chunk=1024):
    out = []
    zz = z.square().sum(-1)[:, None]
    for a in x.split(chunk, 1):
        out.append(
            (
                a.square().sum(-1, keepdim=True)
                + zz
                - 2 * torch.bmm(a, z.transpose(1, 2))
            ).argmin(-1)
        )
    return torch.cat(out, 1)


@torch.no_grad()
def kmeans(x, w, C, seed):
    B, N, c = x.shape
    w = w / w.mean(1, keepdim=True).clamp_min(1e-30)
    empty_weight = w.sum(1) == 0
    w[empty_weight] = 1
    best = None
    bestcost = torch.full((B,), float("inf"), device=x.device)
    traces = []
    for restart in range(2):
        gen = torch.Generator(device=x.device).manual_seed(seed + restart)
        z = torch.empty((B, C, c), device=x.device)
        dist = torch.full((B, N), float("inf"), device=x.device)
        prob = w.clone()
        for k in range(C):
            cdf = prob.cumsum(-1)
            u = torch.rand((B, 1), device=x.device, generator=gen) * cdf[:, -1:]
            ix = torch.searchsorted(cdf.contiguous(), u).squeeze(-1).clamp_max(N - 1)
            z[:, k] = x[torch.arange(B, device=x.device), ix]
            dist = torch.minimum(dist, (x - z[:, k : k + 1]).square().sum(-1))
            prob = w * dist
        previous = torch.full((B,), float("inf"), device=x.device)
        trace = []
        for it in range(100):
            ix = nearest(x, z)
            sums = torch.zeros_like(z)
            mass = torch.zeros((B, C), device=x.device)
            mass.scatter_add_(1, ix, w)
            sums.scatter_add_(1, ix[..., None].expand(-1, -1, c), x * w[..., None])
            z = torch.where(
                (mass > 0)[..., None], sums / mass.clamp_min(1e-30)[..., None], z
            )
            error = (
                (x - z.gather(1, ix[..., None].expand(-1, -1, c))).square().sum(-1) * w
            ).sum(1) / w.sum(1)
            rel = (previous - error).abs() / previous.clamp_min(1e-30)
            trace.append(float(error.mean()))
            if it > 0 and bool((rel <= 1e-5).all()):
                break
            previous = error
        improve = error < bestcost
        if best is None:
            best = z.clone()
        else:
            best[improve] = z[improve]
        bestcost = torch.minimum(bestcost, error)
        traces.append(
            {
                "restart": restart,
                "iterations": it + 1,
                "loss_trace": trace,
                "empty_clusters": int((mass == 0).sum()),
            }
        )
    return best.half(), {
        "restarts": traces,
        "zero_Fisher_groups": int(empty_weight.sum()),
        "mean_training_group_MSE": float(bestcost.mean()),
    }


@torch.no_grad()
def validate(x, grad, books, c):
    H, N, D = x.shape
    J = D // c
    xx = x.float().reshape(H, N, J, c).permute(0, 2, 1, 3).reshape(H * J, N, c).cuda()
    z = books.reshape(H * J, -1, c).float().cuda()
    ix = nearest(xx, z)
    rec = (
        z.gather(1, ix[..., None].expand(-1, -1, c))
        .reshape(H, J, N, c)
        .permute(0, 2, 1, 3)
        .reshape(H, N, D)
    )
    xx = x.float().cuda()
    ww = grad.float().cuda().square()
    return float(
        ((rec - xx).square() * ww).sum() / ((xx.square() * ww).sum().clamp_min(1e-30))
    )
