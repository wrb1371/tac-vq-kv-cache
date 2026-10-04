#!/usr/bin/env python3
"""Fixed 512-step SGD V-codebook QAT on a frozen Mistral training stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from jointkv.cq_tac import ClosedLoopGroupedCQ
from jointkv.table1_protocol import tensor_sha256
from jointkv.uniform_depth_tac import ClosedLoopUniformResidualVQ
from experiments.train_full_model_closed_loop_vq import ClosedLoopRunContext, install_closed_loop_attention, run_loss
from experiments.train_pure_uniform16_closed_loop_vq import clip_codebook_gradients
from release_config import MODEL_BASE, MODEL_INSTRUCT


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "protocols_followup_mistral/mistral_inventory.json"
FROZEN = ROOT / "protocols_followup_mistral/frozen_job_manifest.json"
MODEL_PATHS = {
    "base": MODEL_BASE,
    "instruct": MODEL_INSTRUCT,
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_torch(value, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(target)


def atomic_json(value, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)


def make_codec(cell: str, source: Path, device: str):
    keys, values, selectors = [], [], []
    for layer in range(32):
        body = torch.load(source / f"layer_{layer:02d}.pt", map_location="cpu", weights_only=False)
        if cell.startswith("gsrq"):
            key = body["codebooks"]["key"].half().contiguous()
            value = body["codebooks"]["value"].half().contiguous()
            assert tuple(key.shape) == (8, 16 if cell == "gsrq1" else 32, 256, 128)
            selectors.append(body.get("key_assignment_whiteners"))
        else:
            key = body["key"]["codebooks"].half().contiguous()
            value = body["value"]["codebooks"].half().contiguous()
            assert tuple(key.shape) == (8, 16 if cell == "cq1" else 32, 256, 8 if cell == "cq1" else 4)
            selector = body["key"].get("selector_whiteners")
            if selector is None:
                selector = torch.eye(key.shape[-1]).repeat(key.shape[0], key.shape[1], 1, 1)
            selectors.append(selector)
        assert key.shape == value.shape
        keys.append(key); values.append(value)
    rate = 1.0 if cell.endswith("1") else 2.0
    if cell.startswith("gsrq"):
        codec = ClosedLoopUniformResidualVQ(keys, values, selectors, depth=keys[0].shape[1],
                                             dynamic_bpa=rate, token_chunk_size=1024)
    else:
        codec = ClosedLoopGroupedCQ(keys, values, selectors, dynamic_bpa=rate, token_chunk_size=64)
    return codec.to(device)


def run(args) -> None:
    frozen = json.loads(FROZEN.read_text())
    inventory = json.loads(INVENTORY.read_text())
    assert frozen["training"]["steps"] == 512 and args.seed in frozen["seeds"]
    schedule = "C4-only" if args.schedule == "c4" else "Mix"
    assert f"{args.cell}/{schedule}" in frozen["unique_training_jobs"][args.model]
    checks = json.loads((ROOT / "protocols_followup_mistral/model_and_attention_checks.json").read_text())
    assert checks["models"][args.model]["pass"] is True
    if args.model == "base":
        parent = inventory["existing_parent_codebooks"][f"base_{args.cell}"]
        source = Path(parent["path"])
    else:
        # This file may only be created by a completed independent Instruct-v0.1 parent fit.
        fit = json.loads((ROOT / "protocols_followup_mistral/instruct_parent_fit.json").read_text())
        assert fit["status"] == "complete" and fit["model"] == str(MODEL_PATHS["instruct"])
        parent = fit["parents"][args.cell]
        source = Path(parent["path"])
    assert parent["dynamic_bpa"] == (1.0 if args.cell.endswith("1") else 2.0)
    assert all(Path(row["path"]).is_file() for row in parent["layer_files"])
    data_manifest = ROOT / f"protocols_followup_mistral/{args.model}_{args.schedule}_training_manifest.json"
    data = json.loads(data_manifest.read_text())
    assert data["status"] == "frozen_before_gpu" and data["model"] == str(MODEL_PATHS[args.model])
    run_data = next(row for row in data["runs"] if row["seed"] == args.seed)
    data_file = Path(run_data["tokens"])
    assert sha(data_file) == run_data["tokens_file_sha256"]
    tokens = torch.load(data_file, map_location="cpu", weights_only=False)["tokens"].long()
    assert tuple(tokens.shape) == (512, 2048) and tensor_sha256(tokens) == run_data["token_sha256"]
    key = f"{args.model}_{args.cell}_{args.schedule}_seed{args.seed}"
    work = ROOT / f"experiments_followup_mistral/checkpoints/{key}.pt"
    output = ROOT / f"experiments_followup_mistral/endpoints/{key}"
    receipt = ROOT / f"results_followup_mistral/training_runs/{key}.json"
    if output.exists() or receipt.exists():
        raise FileExistsError(f"finished endpoint target already exists: {key}")
    codec = make_codec(args.cell, source, args.device)
    for table in codec.key_tables:
        table.requires_grad_(False)
    for table in codec.value_tables:
        table.requires_grad_(True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATHS[args.model], torch_dtype=torch.float16,
                                                 attn_implementation="sdpa", low_cpu_mem_usage=True,
                                                 local_files_only=True).to(args.device).eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    context = ClosedLoopRunContext(codec, "uniform")
    install_closed_loop_attention(model, context, active_layers=32)
    optimizer = torch.optim.SGD(codec.value_tables.parameters(), lr=0.001)
    identity = {"job": key, "model_config_sha256": sha(MODEL_PATHS[args.model] / "config.json"),
                "parent_layer_sha256": [x["sha256"] for x in parent["layer_files"]],
                "train_token_sha256": run_data["token_sha256"], "frozen_protocol_sha256": sha(FROZEN),
                "runner_sha256": sha(Path(__file__))}
    start = 0; losses = []; clips = 0; prior_seconds = 0.0
    if work.exists():
        old = torch.load(work, map_location="cpu", weights_only=False)
        if old["identity"] != identity:
            raise RuntimeError("resume identity changed")
        for table, saved in zip(codec.value_tables, old["values"]):
            table.data.copy_(saved.to(args.device))
        start = int(old["completed_steps"]); losses = list(old["losses"])
        clips = int(old["clipped_value_layer_steps"]); prior_seconds = float(old["elapsed_seconds"])
    begin = time.time()
    for step in range(start, 512):
        optimizer.zero_grad(set_to_none=True)
        loss, rates = run_loss(model, context, tokens[step:step + 1].to(args.device), mode="uniform")
        if rates or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite loss or unexpected rate at step {step + 1}")
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in codec.value_tables):
            raise RuntimeError(f"nonfinite/missing V gradient at step {step + 1}")
        clipped = clip_codebook_gradients(codec, maximum_norm=1.0, mode="per_layer_kv")
        clips += sum(bool(row["clip_activated"]) for row in clipped["per_group"] if row["kind"] == "value")
        optimizer.step()
        losses.append(float(loss.detach()))
        completed = step + 1
        if completed == 1 or completed % 16 == 0:
            print(json.dumps({"job": key, "step": completed, "loss": losses[-1],
                              "elapsed_seconds": prior_seconds + time.time() - begin}), flush=True)
        if completed % 64 == 0 and completed < 512:
            atomic_torch({"identity": identity, "completed_steps": completed,
                          "values": [p.detach().cpu() for p in codec.value_tables],
                          "losses": losses, "clipped_value_layer_steps": clips,
                          "elapsed_seconds": prior_seconds + time.time() - begin}, work)
    torch.cuda.synchronize()
    elapsed = prior_seconds + time.time() - begin
    output.mkdir(parents=True, exist_ok=False)
    layer_files = []
    for layer, table in enumerate(codec.value_tables):
        path = output / f"value_layer_{layer:02d}.pt"
        atomic_torch({"layer": layer, "value_table": table.detach().half().cpu().contiguous(),
                      "identity": identity}, path)
        layer_files.append({"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size})
    result = {"status": "complete", "identity": identity, "model": args.model, "cell": args.cell,
              "schedule": schedule, "seed": args.seed, "completed_steps": 512,
              "source_parent": str(source), "training_data_manifest": str(data_manifest),
              "optimizer": "SGD", "learning_rate": 0.001, "gradient_clip": 1.0,
              "K_frozen": True, "backbone_frozen": True, "V_only": True, "hard_forward": True,
              "deployed_dtype": "float16", "master_dtype": "float32", "weight_dtype": "float16",
              "losses": losses, "clipped_value_layer_steps": clips, "elapsed_seconds": elapsed,
              "value_layer_files": layer_files}
    atomic_json(result, receipt)
    if work.exists():
        work.unlink()
    print(json.dumps({"complete": key, "elapsed_seconds": elapsed,
                      "receipt_sha256": sha(receipt)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("base", "instruct"), required=True)
    parser.add_argument("--cell", choices=("gsrq1", "gsrq2", "cq1", "cq2"), required=True)
    parser.add_argument("--schedule", choices=("c4", "mix"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
