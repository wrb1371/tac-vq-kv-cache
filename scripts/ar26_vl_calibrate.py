"""Native 7B conditional curves, all24 declared media sources and all28layers."""

import argparse, hashlib, json, pathlib, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ar26kv.runtime import *
from ar26kv.calibration import measure_layer, prefixes, FRACTIONS, GOPS
from ar26kv.matching import fixed_reference, references


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    a = p.parse_args()
    start = time.time()
    out = ROOT / "data/runs" / a.run_id
    out.mkdir(parents=True, exist_ok=False)
    bank = ROOT / "artifacts/ar26/packages/STI_RVQ"
    assert json.loads((bank / "completion.json").read_text())["completed"]
    t = {
        k: v.cuda()
        for k, v in torch.load(
            bank / "global.pt", map_location="cpu", weights_only=True
        ).items()
    }
    model, processor = load()
    inp = Inputs(processor)
    teacher = Teacher(model)
    items = json.loads((ROOT / "manifests/ar22/MEDIA_CAL24.json").read_text())
    assert len(items) == 24 and all(
        "answer" not in i and "question" not in i for i in items
    )
    vm = [
        b.self_attn.o_proj.weight.float()
        .reshape(3584, 4, 7, 128)
        .square()
        .sum((0, 2))
        .cpu()
        for b in model.model.language_model.layers
    ]
    records = []
    rate_acc = {s: np.zeros((28, 7)) for s in ["K", "V"]}
    rate_den = np.zeros(28)
    with (out / "by_video.jsonl").open("x") as f:
        for i, item in enumerate(items):
            x, pos, vi, ti, end, identity = inp.get(model, item)
            teacher.capture(x, pos, end)
            grid = identity["grid"]
            nt, h, w = grid
            p = h * w
            ordinal = torch.arange(len(vi), device="cuda")
            ao = ordinal[(ordinal // p) % 8 == 0]
            to = ordinal[(ordinal // p) % 8 != 0]
            measured = []
            for l in range(28):
                native = {k: v[None] for k, v in teacher.states[l].items()}
                result = measure_layer(
                    l,
                    native,
                    teacher.moments[l],
                    vm[l],
                    t,
                    vi.to(torch.int32),
                    ti,
                    grid,
                    None,
                    96,
                )
                measured.append(result)
                # Parent reference: independent anchor and decoded-reference residual rate curves.
                raw = {
                    "K": teacher.states[l]["K_pre"].cuda().float(),
                    "V": teacher.states[l]["V"].cuda().float(),
                }
                ts = to[
                    torch.linspace(
                        0, len(to) - 1, min(128, len(to)), device="cuda"
                    ).long()
                ]
                aa = ao[
                    torch.linspace(
                        0, len(ao) - 1, min(128, len(ao)), device="cuda"
                    ).long()
                ]
                codes, scores = fixed_reference(
                    raw,
                    vi.to(torch.int32),
                    ts,
                    [t["K_scale"][l], t["V_scale"][l]],
                    h,
                    w,
                    8,
                )
                refs = vi[references(ts, codes, h, w, 8)]
                pick = torch.argsort(scores, dim=1, descending=True, stable=True)[
                    :, : max(1, int(np.ceil(len(ts) / 8)))
                ]
                heads = torch.arange(4, device="cuda")[:, None]
                for s in ["K", "V"]:
                    met = teacher.moments[l].cuda() / 128 if s == "K" else vm[l].cuda()
                    anchors = raw[s][:, vi[aa]]
                    ref = raw[s].gather(1, refs[..., None].expand(-1, -1, 128))
                    pred = prefixes(ref, t, s, l)[-1].float()
                    res = (raw[s][:, vi[ts]] - pred)[heads, pick]
                    total = np.zeros(7)
                    for vals, pop, independent in [
                        (anchors, len(ao), True),
                        (res, int(np.ceil(len(to) / 8)), False),
                    ]:
                        dec = prefixes(vals, t, s, l, independent=independent)
                        zero = (
                            t[s + "_mean"][l].float()[:, None].expand_as(vals)
                            if independent
                            else torch.zeros_like(vals)
                        )
                        losses = [
                            float(
                                ((vals - y.float()).square() * met[:, None])
                                .sum(-1)
                                .mean()
                            )
                            for y in [zero] + dec
                        ]
                        total += np.array(losses) * pop / end
                    rate_acc[s][l] += total
                rate_den[l] += (len(ao) + int(np.ceil(len(to) / 8))) / end
                if l % 7 == 6:
                    print(
                        json.dumps(
                            {"media": i, "layer": l, "seconds": time.time() - start}
                        ),
                        flush=True,
                    )
            record = {
                "id": item["id"],
                "group": item["duration_group"],
                "identity": identity,
                "layers": measured,
                "QA_used": False,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            records.append(record)
            save(
                out / "progress.json",
                {"completed_media": i + 1, "seconds": time.time() - start},
            )
            teacher.states = {}
            teacher.raw = {}
            del x, pos, raw
            torch.cuda.empty_cache()
    groups = {}
    for group in ["short", "medium", "long"]:
        rows = [r for r in records if r["group"] == group]
        assert len(rows) == 8
        groups[group] = {"norm": {}}
        for g in GOPS:
            groups[group]["norm"][str(g)] = {
                s: [
                    {
                        term: np.mean(
                            [r["layers"][l]["norm"][str(g)][s][term] for r in rows], 0
                        ).tolist()
                        for term in ["anchor", "target", "query"]
                    }
                    for l in range(28)
                ]
                for s in ["K", "V"]
            }
    for s in ["K", "V"]:
        scale = np.mean(
            [
                sum(
                    c["anchor"][0] / 8 + c["target"][3][0] * 7 / 8
                    for c in groups[g]["norm"]["8"][s]
                )
                for g in groups
            ]
        )
        assert scale > 0
        for g in groups:
            for gg in groups[g]["norm"]:
                for c in groups[g]["norm"][gg][s]:
                    for term in c:
                        c[term] = (np.asarray(c[term]) / scale).tolist()
        rate_acc[s] /= rate_den[:, None]
        rate_acc[s] /= rate_acc[s][:, 1].sum()
    banksha = hashlib.sha256((bank / "global.pt").read_bytes()).hexdigest()
    dest = ROOT / "data/ar26/VL_conditional_curves.json"
    save(
        dest,
        {
            "completed": True,
            "groups": groups,
            "fractions": FRACTIONS,
            "gops": GOPS,
            "model": "Qwen2.5-VL-7B-Instruct",
            "actual_media": 24,
            "layers": 28,
            "bank_SHA256": banksha,
            "QA_used": False,
        },
    )
    save(
        ROOT / "data/ar26/VL_rate_curves.json",
        {
            "completed": True,
            "plain": {s: v.tolist() for s, v in rate_acc.items()},
            "actual_media": 24,
            "layers": 28,
            "QA_used": False,
            "bank_SHA256": banksha,
        },
    )
    save(
        out / "completion.json",
        {
            "completed": True,
            "actual_media": 24,
            "layers": 28,
            "elapsed_s": time.time() - start,
            "QA_used": False,
        },
    )
    teacher.close()


if __name__ == "__main__":
    main()
