import argparse

"""Paired GPU-event reconstruction benchmark on actual evaluated packed states.

Records other resident processes explicitly. No attention/model execution is timed.
"""

import collections, hashlib, json, os, pathlib, statistics, subprocess, sys, time, gc, argparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from ar26kv.codec import Codec, apply_rope
from ar26kv.matching import references
from ar35kv.reader import install
from ar26kv.runtime import save


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def vectorized(c, l):
    if c.plan["hybrid_dense"]:
        return c._ar35_reference_decode(l)
    st = c.states[l]
    heads = st["K_index"].shape[0]
    nt, h, w = c.grid
    p = h * w
    g = c.plan["gop"]
    vi = c.global_maps["video"].long()
    ti = c.global_maps["text"].long()
    m = len(vi)
    o = torch.arange(m, device=vi.device)
    ao = o[(o // p) % g == 0]
    to = o[(o // p) % g != 0]
    code = st["T_code"][:, vi[to]]
    ref = references(to, code & 127, h, w, g)
    slot = (ref // p // g) * p + ref % p
    a = c.plan["anchors"]
    ni = c.plan["layer_choices"][l]["innovations"]
    mask = (code & 128) != 0
    rank = mask.int().cumsum(1).long()
    index = torch.where(mask, rank, 0)
    result = {}
    for side in ["K", "V"]:
        anchors = c.read_product(l, side, st, a, True)
        out = torch.empty((heads, c.end, 128), device=vi.device, dtype=torch.bfloat16)
        out[:, vi[ao]] = anchors
        prediction = anchors.gather(1, slot[..., None].expand(-1, -1, 128))
        if ni:
            innovation = c.read_product(l, side, st, ni, False, start=a)
            zero = torch.zeros(
                (heads, 1, 128), device=vi.device, dtype=innovation.dtype
            )
            delta = torch.cat([zero, innovation], 1).gather(
                1, index[..., None].expand(-1, -1, 128)
            )
            prediction = (prediction.float() + delta.float()).bfloat16()
        out[:, vi[to]] = prediction
        result[side] = out
    result["K"] = apply_rope(
        result["K"],
        c.global_maps["positions"][:, : c.end],
        c.global_maps["freq"],
        c.scale,
    )
    result["K"][:, ti] = st["text_FP_K"]
    result["V"][:, ti] = st["text_FP_V"]
    return result["K"], result["V"]


def snapshot(gpu):
    lines = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    dev = next(x for x in lines if int(x.split(",")[0]) == gpu)
    uuid = dev.split(",")[1].strip()
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    )
    return {
        "epoch": time.time(),
        "device": dev,
        "other_resident_processes": [
            x
            for x in raw.splitlines()
            if uuid in x and int(x.split(",")[1]) != os.getpid()
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    admission = snapshot(gpu)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.4)
    ids = args.state
    rows = []
    start = time.time()
    resource = [admission]
    for index, sid in enumerate(ids):
        path = pathlib.Path(sid)
        d = torch.load(path, map_location="cpu", weights_only=True)
        c = object.__new__(Codec)
        for k in ["plan", "grid", "scale", "end", "receipt"]:
            setattr(c, k, d[k])
        c.name = "STI-RVQ"
        c.M = len(d["global_maps"]["video"])
        c.states = {
            l: {k: v.cuda() for k, v in st.items()} for l, st in d["states"].items()
        }
        c.table = {k: v.cuda() for k, v in d["table"].items()}
        c.global_maps = {k: v.cuda() for k, v in d["global_maps"].items()}
        install(c)
        paths = {
            "reference": c._ar35_reference_decode,
            "vectorized": lambda l: vectorized(c, l),
            "fused": c.decode,
        }
        timings = {k: [] for k in paths}
        workspace = {}
        for l in range(28):
            a = paths["reference"](l)
            for name in ["vectorized", "fused"]:
                b = paths[name](l)
                assert all(torch.equal(x, y) for x, y in zip(a, b)), (sid, l, name)
                del b
            del a
        for fn in paths.values():
            for _ in range(5):
                for l in range(28):
                    z = fn(l)
                    del z
        torch.cuda.synchronize()
        for repeat in range(20):
            order = list(paths)
            offset = (index + repeat) % 3
            order = order[offset:] + order[:offset]
            if repeat % 2:
                order.reverse()
            for name in order:
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                for l in range(28):
                    z = paths[name](l)
                    del z
                b.record()
                b.synchronize()
                timings[name].append(a.elapsed_time(b))
                workspace[name] = max(
                    workspace.get(name, 0), torch.cuda.max_memory_allocated() - base
                )
        row = {
            "source_id": sid,
            "snapshot_SHA256": sha(path),
            "resident_bytes": c.receipt["resident_bytes"],
            "budget_mib": c.receipt["budget_bytes"] / 2**20,
            "full28layer_BF16_exact_all_three": True,
            "timings_ms": timings,
            "median_ms": {k: statistics.median(v) for k, v in timings.items()},
            "workspace_peak_bytes": workspace,
        }
        row["speedup_vs_reference"] = (
            row["median_ms"]["reference"] / row["median_ms"]["fused"]
        )
        row["speedup_vs_vectorized"] = (
            row["median_ms"]["vectorized"] / row["median_ms"]["fused"]
        )
        rows.append(row)
        resource.append(snapshot(gpu))
        print(
            json.dumps(
                {
                    "source": sid,
                    "speedup": row["speedup_vs_vectorized"],
                    "elapsed_s": time.time() - start,
                }
            ),
            flush=True,
        )
        del c, paths, d
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "completed": True,
        "epoch": time.time(),
        "GPU": gpu,
        "GPU_name": torch.cuda.get_device_name(),
        "resource_snapshots": resource,
        "exclusive_GPU": not any(x["other_resident_processes"] for x in resource),
        "native_source_states": len(ids),
        "layers_per_state": 28,
        "warmups": 5,
        "paired_repetitions": 20,
        "rows": rows,
        "elapsed_s": time.time() - start,
        "source_SHA256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [ROOT / "src/ar35kv/reader.py", pathlib.Path(__file__)]
        },
        "state_and_codebook_bitwise_equivalence": True,
        "no_attention_or_model_forward_in_microbenchmark": True,
        "GPU1_used": False,
    }
    save(pathlib.Path(args.output), result)


if __name__ == "__main__":
    main()
