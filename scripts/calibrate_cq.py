"""Media-only label-free CQ empirical Fisher; no evaluation questions or answers."""

import argparse, gc, hashlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/OmniKVQuant_snapshot")]
import torch
from evaluate import load_model
from ar37cq.runtime import Supplier, Teacher, fisher_readout
from ar26kv.runtime import save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["omni3b", "vl7b"], required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    items = json.loads(pathlib.Path(a.manifest).read_text())
    assert len(items) == 24 and all(
        it["prompt"] == "Describe the video."
        and not any(k in it for k in ["question", "answer", "options"])
        for it in items
    )
    kind = "Omni" if a.backbone == "omni3b" else "VL"
    out = pathlib.Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    model, p = load_model(a.backbone, a.model_dir)
    supplier = Supplier(model, p, kind)
    teacher = Teacher(model, kind)
    rows = []
    for it in items:
        x, pos, vi, ti, end, ident = supplier.get(it, verify=True)
        teacher.capture(x, pos, end, ident)
        sample, record = fisher_readout(model, p, kind, teacher, x, pos, end, vi)
        dest = out / (it["id"] + ".pt")
        torch.save(sample, dest)
        rows.append(
            {
                "id": it["id"],
                "source_id": it["source_id"],
                "duration_group": it["duration_group"],
                "file": dest.name,
                "SHA256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                **record,
            }
        )
        teacher.clear()
        del x, pos, vi, ti, sample
        gc.collect()
        torch.cuda.empty_cache()
    teacher.close()
    save(
        out / "completion.json",
        {"completed": True, "model": kind, "rows": rows, "no_test_QA_used": True},
    )


if __name__ == "__main__":
    main()
