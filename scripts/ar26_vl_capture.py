"""Native BF16 gate and media-only sampled calibration capture; no dense KV on disk."""

import argparse, json, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ar26kv.runtime import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--gate", action="store_true")
    a = p.parse_args()
    dest = ROOT / "data/runs" / a.run_id
    dest.mkdir(parents=True, exist_ok=False)
    start = time.time()
    model, processor = load()
    inputs = Inputs(processor)
    teacher = Teacher(model)
    rows = []
    items = json.loads((ROOT / a.data).read_text())
    versions = [p._version for p in model.parameters()]
    for i, item in enumerate(items):
        x, pos, vi, ti, end, identity = inputs.get(model, item, verify=True)
        teacher.capture(x, pos, end)
        row = {
            "id": item["id"],
            "source_id": item["source_id"],
            "identity": identity,
            "question_used_for_capture": False,
            "seconds": time.time() - start,
        }
        if a.gate:
            bf = answer(
                model, processor, x, pos, end, teacher.bf16_cache(), item["options"]
            )
            with torch.inference_mode(), last_projection(model):
                o = model(**x, position_ids=pos, use_cache=False)
            z = o.logits[0, -1].float()
            tok = int(z.argmax())
            err = max(
                abs(
                    v
                    - float(
                        z[
                            processor.tokenizer.encode(
                                chr(65 + j), add_special_tokens=False
                            )[0]
                        ]
                    )
                )
                for j, v in enumerate(bf["option_logits"])
            )
            assert bf["generated_token_id"] == tok
            row.update(
                native_vs_split_max_option_error=err,
                bf16=bf,
                native_first_token=processor.tokenizer.decode([tok]),
            )
        else:
            # Deterministic 512 per-layer/head samples, solely pre-question states.
            ix = vi[
                torch.linspace(
                    0, len(vi) - 1, min(512, len(vi)), device=vi.device
                ).long()
            ].cpu()
            pool = {
                l: {
                    "K": s["K_pre"][:, ix].clone(),
                    "V": s["V"][:, ix].clone(),
                    "q_moment": teacher.moments[l].clone(),
                }
                for l, s in teacher.states.items()
            }
            f = ROOT / "artifacts/ar26/calibration_samples" / (item["id"] + ".pt")
            f.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pool, f)
            row["sample_file"] = str(f.relative_to(ROOT))
            row["sample_file_bytes"] = f.stat().st_size
        rows.append(row)
        save(dest / "progress.json", rows)
        print(json.dumps(row), flush=True)
        del x, pos, vi, ti
        teacher.states = {}
        teacher.raw = {}
        torch.cuda.empty_cache()
    assert versions == [p._version for p in model.parameters()]
    done = {
        "completed": True,
        "run_id": a.run_id,
        "rows": rows,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_s": time.time() - start,
        "GPU_peak_MiB": torch.cuda.max_memory_allocated() / 2**20,
        "full_dense_KV_written_to_disk": False,
        "labels_used_for_fit": False,
    }
    save(dest / "completion.json", done)
    teacher.close()


if __name__ == "__main__":
    main()
