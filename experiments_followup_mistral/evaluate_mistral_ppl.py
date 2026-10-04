#!/usr/bin/env python3
"""Hash-bound, resumable Mistral Base PPL windows for Tables 1 and 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from experiments.train_full_model_closed_loop_vq import ClosedLoopRunContext, install_closed_loop_attention, run_loss
from experiments_followup_mistral.train_mistral_v_only import make_codec, sha
from jointkv.table1_protocol import tensor_sha256
from release_config import MODEL_BASE


ROOT = Path(__file__).resolve().parents[1]
MODEL = MODEL_BASE


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def endpoint_descriptor(name: str):
    if name == "fp16":
        return None, None, {"name": name, "config_sha256": sha(MODEL / "config.json")}
    bits = name.split("_")
    cell = bits[0]
    if cell not in ("gsrq1", "gsrq2", "cq1", "cq2"):
        raise ValueError(name)
    inventory = json.loads((ROOT / "protocols_followup_mistral/mistral_inventory.json").read_text())
    parent = inventory["existing_parent_codebooks"][f"base_{cell}"]
    assert len(parent["layer_files"]) == 32
    identity = {"name": name, "cell": cell, "parent_hashes": [x["sha256"] for x in parent["layer_files"]]}
    if bits[1] == "parent":
        return cell, Path(parent["path"]), identity
    assert len(bits) == 4 and bits[1] in ("c4", "mix") and bits[2].startswith("seed")
    seed = int(bits[2].removeprefix("seed")) if bits[3] == "qat" else None
    assert seed in (8601, 8607, 8613)
    receipt_path = ROOT / f"results_followup_mistral/training_runs/base_{cell}_{bits[1]}_seed{seed}.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "complete" and receipt["completed_steps"] == 512
    assert receipt["cell"] == cell and receipt["seed"] == seed
    assert receipt["identity"]["parent_layer_sha256"] == identity["parent_hashes"]
    assert len(receipt["value_layer_files"]) == 32
    identity.update({"training_receipt_sha256": sha(receipt_path),
                     "value_hashes": [x["sha256"] for x in receipt["value_layer_files"]]})
    return cell, Path(parent["path"]), identity


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=("table1", "table2"), required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    args = p.parse_args()
    manifest_path = ROOT / f"protocols_followup_mistral/{args.suite}_eval_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "frozen_before_gpu"
    entry = manifest["corpora"][args.corpus]
    token_file = Path(entry["token_file"])
    assert sha(token_file) == entry["token_file_sha256"]
    tokens = torch.load(token_file, map_location="cpu", weights_only=False)["token_ids"].long()
    assert tensor_sha256(tokens) == entry["token_sha256"]
    length = int(entry["sequence_length"])
    tokens = tokens.reshape(-1, length)
    assert len(tokens) == entry["windows"]
    cell, source, endpoint = endpoint_descriptor(args.endpoint)
    if args.suite == "table2":
        assert args.endpoint in ("gsrq1_parent", *(f"gsrq1_{schedule}_seed{s}_qat" for schedule in ("c4", "mix") for s in (8601, 8607, 8613)))
    identity = {"suite": args.suite, "endpoint": endpoint, "corpus": args.corpus,
                "manifest_sha256": sha(manifest_path), "token_sha256": entry["token_sha256"],
                "runner_sha256": sha(Path(__file__))}
    output = ROOT / f"results_followup_mistral/{args.suite}_window_records/{args.endpoint}/{args.corpus}"
    output.mkdir(parents=True, exist_ok=True)
    end = min(len(tokens), args.end if args.end is not None else len(tokens))
    todo = []
    for index in range(args.start, end):
        path = output / f"window_{index:04d}.json"
        if path.exists():
            old = json.loads(path.read_text())
            assert old["status"] == "complete" and old["identity"] == identity
        else:
            todo.append(index)
    print(json.dumps({"endpoint": args.endpoint, "corpus": args.corpus, "todo": len(todo),
                      "total_windows": len(tokens)}), flush=True)
    if not todo:
        return
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa", low_cpu_mem_usage=True,
                                                 local_files_only=True).to(args.device).eval()
    model.config.use_cache = False
    context = None
    if cell:
        codec = make_codec(cell, source, args.device)
        if "value_hashes" in endpoint:
            receipt_path = ROOT / f"results_followup_mistral/training_runs/base_{cell}_{args.endpoint.split('_')[1]}_{args.endpoint.split('_')[2]}.json"
            receipt = json.loads(receipt_path.read_text())
            for layer, row in enumerate(receipt["value_layer_files"]):
                path = Path(row["path"])
                assert sha(path) == row["sha256"]
                tensor = torch.load(path, map_location="cpu", weights_only=False)["value_table"]
                assert tensor.shape == codec.value_tables[layer].shape
                codec.value_tables[layer].data.copy_(tensor.to(args.device))
        context = ClosedLoopRunContext(codec, "uniform")
        install_closed_loop_attention(model, context, active_layers=32)
    for index in todo:
        row = tokens[index:index + 1].to(args.device)
        with torch.no_grad():
            if context is None:
                logits = model(input_ids=row, use_cache=False).logits
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                       row[:, 1:].reshape(-1), reduction="mean")
                rates = []
            else:
                loss, rates = run_loss(model, context, row, mode="uniform")
        assert not rates and math.isfinite(float(loss))
        item = {"status": "complete", "identity": identity, "window": index,
                "window_token_sha256": tensor_sha256(tokens[index:index + 1]),
                "mean_nll": float(loss), "perplexity": math.exp(float(loss)),
                "evaluated_tokens": length - 1}
        path = output / f"window_{index:04d}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(path)
        if index == todo[0] or (index + 1) % 16 == 0:
            print(json.dumps({"endpoint": args.endpoint, "corpus": args.corpus,
                              "window": index + 1, "mean_nll": item["mean_nll"]}), flush=True)


if __name__ == "__main__":
    main()
