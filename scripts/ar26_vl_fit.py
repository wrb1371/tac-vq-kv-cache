"""Model-specific shared RVQ and two genuinely distinct private GSRQ fits."""

import argparse, hashlib, json, pathlib, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ar26kv.runtime import save


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    import torch
    from r14kv.fitting import nearest, hadamard, fit_rq
    from r15kv.fitting import fit_fp8_rq

    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    a = p.parse_args()
    start = time.time()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = ROOT / "data/runs" / a.run_id
    out.mkdir(parents=True, exist_ok=False)
    receipt = ROOT / "data/runs/AR26_VL_MEDIA_CAL24_v1/completion.json"
    while not receipt.exists():
        print("WAIT_MEDIA_ONLY_CALIBRATION", flush=True)
        time.sleep(20)
    done = json.loads(receipt.read_text())
    assert done["completed"] and len(done["rows"]) == 24
    pools = [
        torch.load(ROOT / r["sample_file"], map_location="cpu", weights_only=True)
        for r in done["rows"]
    ]
    # Model weights are read to derive the Value metric; no forward execution here.
    from safetensors import safe_open

    index = json.loads(
        (
            ROOT
            / "artifacts/ar26/models/Qwen2.5-VL-7B-Instruct/model.safetensors.index.json"
        ).read_text()
    )["weight_map"]
    metrics = {"K": [], "V": []}
    for l in range(28):
        key = f"model.layers.{l}.self_attn.o_proj.weight"
        if key not in index:
            key = f"model.language_model.layers.{l}.self_attn.o_proj.weight"
        with safe_open(
            ROOT / "artifacts/ar26/models/Qwen2.5-VL-7B-Instruct" / index[key],
            framework="pt",
        ) as f:
            w = f.get_tensor(key).float()
        metrics["V"].append(w.reshape(3584, 4, 7, 128).square().sum((0, 2)))
        metrics["K"].append(
            torch.stack([v[l]["q_moment"] for v in pools]).mean(0) / 128.0
        )
    table = {"H": hadamard(128).float()}
    raw = {
        s: torch.stack([torch.cat([v[l][s] for v in pools], 1) for l in range(28)])
        for s in ["K", "V"]
    }
    for s in ["K", "V"]:
        x = raw[s].float()
        table[s + "_mean"] = x.mean(2).half()
        table[s + "_scale"] = x.std(2, unbiased=False).clamp_min(1e-4).half()
    trace = {}
    h = table["H"].cuda()
    for si, s in enumerate(["K", "V"]):
        x = raw[s].cuda().float()
        metric = torch.stack(metrics[s]).cuda()
        mag = (x.square() * metric[:, :, None]).sum(-1).sqrt()
        weights = torch.log1p(
            mag / mag.median(2, keepdim=True).values.clamp_min(1e-12)
        ).clamp_min(1e-8)
        u = (x - table[s + "_mean"].cuda().float()[:, :, None]) / table[
            s + "_scale"
        ].cuda().float()[:, :, None]
        gain = u.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6).bfloat16()
        z = ((u / gain.float()) @ h).reshape(1, -1, 16)
        bw = weights[:, :, :, None].expand(-1, -1, -1, 8).reshape(1, -1)
        rng = np.random.default_rng(17261023 + 16 + si * 1000)
        pick = rng.choice(z.shape[1], 262144, replace=False)
        ix = torch.tensor(pick, device="cuda")
        res = z[:, ix].clone()
        wt = bw[:, ix].clone()
        books = []
        hist = []
        for r in range(6):
            gen = torch.Generator(device="cuda").manual_seed(
                17261023 + 1600 + 1000 * si + r
            )
            c = res[
                :, torch.randperm(res.shape[1], generator=gen, device="cuda")[:256]
            ].clone()
            for it in range(20):
                ids = nearest(res, c)
                ss = torch.zeros_like(c)
                counts = torch.zeros(1, 256, device="cuda")
                ss.scatter_add_(
                    1, ids[..., None].expand(-1, -1, 16), res * wt[..., None]
                )
                counts.scatter_add_(1, ids, wt)
                c = torch.where(
                    (counts > 0)[..., None], ss / counts.clamp_min(1e-20)[..., None], c
                )
            dep = c.half()
            ids = nearest(res, dep.float())
            res -= torch.gather(dep.float(), 1, ids[..., None].expand(-1, -1, 16))
            books.append(dep[0].cpu())
            row = {
                "side": s,
                "stage": r + 1,
                "weighted_MSE": float((res.square().sum(-1) * wt).sum() / wt.sum()),
                "seconds": time.time() - start,
            }
            hist.append(row)
            print(json.dumps(row), flush=True)
        table[s + "_books"] = torch.stack(books)
        trace[s] = hist
        del x, metric, mag, weights, u, gain, z, bw, res, wt
        torch.cuda.empty_cache()
    package = ROOT / "artifacts/ar26/packages/STI_RVQ"
    package.mkdir(parents=True, exist_ok=False)
    torch.save(table, package / "global.pt")
    save(
        package / "completion.json",
        {
            "completed": True,
            "model": "Qwen2.5-VL-7B-Instruct",
            "geometry": [28, 28, 4, 128],
            "static_tensor_bytes": sum(
                t.untyped_storage().nbytes() for t in table.values()
            ),
            "global_SHA256": sha(package / "global.pt"),
            "calibration_completion_SHA256": sha(receipt),
            "labels_used": False,
            "trace": trace,
        },
    )
    for family, fit in [("GSRQExport", fit_rq), ("GSRQAware", fit_fp8_rq)]:
        records = []
        for l in range(28):
            books = {}
            for si, s in enumerate(["K", "V"]):
                x = raw[s][l].cuda().float()
                met = metrics[s][l].cuda()
                norm = (x.square() * met[:, None]).sum(-1).sqrt()
                cb, tr = fit(
                    x,
                    norm,
                    [14261021 + 10000 * si + l * 10 + h for h in range(4)],
                    stages=5,
                    iterations=40,
                )
                if family == "GSRQExport":
                    cb = cb.to(torch.float8_e4m3fn)
                books[s] = cb
                save(out / f"{family}_layer{l:02}_{s}_trace.json", tr)
            for stages in [1, 2, 5]:
                folder = ROOT / "artifacts/ar26/packages" / family / f"R{stages}"
                folder.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {s: b[:, :stages].clone() for s, b in books.items()},
                    folder / f"layer{l:02}.pt",
                )
            print(
                json.dumps(
                    {"family": family, "layer": l, "seconds": time.time() - start}
                ),
                flush=True,
            )
        for stages in [1, 2, 5]:
            folder = ROOT / "artifacts/ar26/packages" / family / f"R{stages}"
            save(
                folder / "completion.json",
                {
                    "completed": True,
                    "model": "Qwen2.5-VL-7B-Instruct",
                    "family": family,
                    "Rmax": stages,
                    "FP8_aware_fit": family == "GSRQAware",
                    "static_tensor_bytes": 28 * 4 * 2 * stages * 256 * 128,
                    "records": [
                        {
                            "file": f"layer{l:02}.pt",
                            "SHA256": sha(folder / f"layer{l:02}.pt"),
                        }
                        for l in range(28)
                    ],
                    "calibration_completion_SHA256": sha(receipt),
                    "labels_used": False,
                },
            )
    save(
        out / "completion.json",
        {
            "completed": True,
            "elapsed_s": time.time() - start,
            "all_packages_completed": True,
            "labels_used": False,
        },
    )


if __name__ == "__main__":
    main()
