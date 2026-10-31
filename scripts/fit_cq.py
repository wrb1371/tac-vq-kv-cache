"""Fit one independent CQ bank using the evaluated weighted k-means implementation."""

import argparse, gc, hashlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/OmniKVQuant_snapshot")]
import torch
from ar37cq.fitting import kmeans, validate
from ar26kv.runtime import save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument(
        "--centroids",
        type=int,
        choices=[16, 32, 64, 128, 256, 512, 1024],
        required=True,
    )
    ap.add_argument(
        "--coupling", type=int, choices=[1, 2, 4, 8, 16, 32, 64, 128], required=True
    )
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(
        min(0.95, 26 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    )
    root = pathlib.Path(a.samples)
    cal = json.loads((root / "completion.json").read_text())
    assert cal["completed"] and len(cal["rows"]) == 24
    kind = cal["model"]
    L, H = (36, 2) if kind == "Omni" else (28, 4)
    C, c = a.centroids, a.coupling
    J = 128 // c
    out = pathlib.Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    for r in cal["rows"]:
        assert (
            hashlib.sha256((root / r["file"]).read_bytes()).hexdigest() == r["SHA256"]
        )
    samples = [torch.load(root / r["file"], weights_only=True) for r in cal["rows"]]
    records = []
    for l in range(L):
        bank = {}
        record = {"layer": l, "sides": {}}
        for side in ["K", "V"]:
            x = (
                torch.cat(
                    [s[l][side][:, torch.arange(512) % 4 != 0] for s in samples], 1
                )
                .float()
                .reshape(H, -1, J, c)
                .permute(0, 2, 1, 3)
                .reshape(H * J, -1, c)
                .cuda()
            )
            g = (
                torch.cat(
                    [
                        s[l][side + "_grad"][:, torch.arange(512) % 4 != 0]
                        for s in samples
                    ],
                    1,
                )
                .float()
                .reshape(H, -1, J, c)
                .permute(0, 2, 1, 3)
                .reshape(H * J, -1, c)
                .cuda()
            )
            w = g.square().sum(-1)
            del g
            z, training = kmeans(
                x, w, C, 20261031 + 100000 * l + 10000 * (side == "V") + C + c
            )
            del x, w
            z = z.reshape(H, J, C, c).cpu()
            bank[side] = z
            record["sides"][side] = {
                "training": training,
                "validation_by_source": [
                    validate(s[l][side][:, ::4], s[l][side + "_grad"][:, ::4], z, c)
                    for s in samples
                ],
            }
        path = out / f"layer{l:02}.pt"
        torch.save(bank, path)
        record["SHA256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(record)
        save(out / "progress.json", {"layers": records})
        gc.collect()
        torch.cuda.empty_cache()
    score = {
        group: sum(
            r["sides"][s]["validation_by_source"][i]
            for r in records
            for s in ["K", "V"]
            for i, it in enumerate(cal["rows"])
            if it["duration_group"] == group
        )
        / (L * 2 * 8)
        for group in ["short", "medium", "long"]
    }
    save(
        out / "completion.json",
        {
            "completed": True,
            "model": kind,
            "C": C,
            "c": c,
            "layers": records,
            "codec_bytes": 4 * L * H * 128 * C,
            "validation_loss": score,
            "no_test_QA_used": True,
        },
    )


if __name__ == "__main__":
    main()
