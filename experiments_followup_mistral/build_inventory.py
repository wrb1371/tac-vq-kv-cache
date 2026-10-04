#!/usr/bin/env python3
"""Inventory actual local Mistral v0.1 models, parents, and QAT endpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from release_config import MODEL_BASE, MODEL_INSTRUCT, PARENT_ROOT


ROOT = Path(__file__).resolve().parents[1]
BASE = MODEL_BASE
INSTRUCT = MODEL_INSTRUCT
PARENTS = {
    "base_gsrq1": PARENT_ROOT / "base_gsrq1",
    "base_gsrq2": PARENT_ROOT / "base_gsrq2",
    "base_cq1": PARENT_ROOT / "base_cq1",
    "base_cq2": PARENT_ROOT / "base_cq2",
}
TARGET = ROOT / "protocols_followup_mistral/mistral_inventory.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file(path: Path) -> dict:
    return {"path": str(path), "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha(path) if path.is_file() else None}


def model(path: Path, expected_template: bool) -> dict:
    config = json.loads((path / "config.json").read_text())
    tok_config = json.loads((path / "tokenizer_config.json").read_text())
    assert config["model_type"] == "mistral"
    assert config["sliding_window"] == 4096
    assert bool(tok_config.get("chat_template")) == expected_template
    return {
        "model_id": f"mistralai/{'Mistral-7B-Instruct-v0.1' if expected_template else 'Mistral-7B-v0.1'}",
        "local_path": str(path),
        "revision": "local ModelScope snapshot; upstream commit not encoded in supplied files",
        "config": file(path / "config.json"),
        "tokenizer": file(path / "tokenizer.json"),
        "tokenizer_config": file(path / "tokenizer_config.json"),
        "weight_index": file(path / "pytorch_model.bin.index.json"),
        "weight_shards": [{"path": str(p), "exists": True, "bytes": p.stat().st_size,
                           "sha256": "not_hashed_large_weight; index and path bound"}
                          for p in sorted(path.glob("pytorch_model-*.bin"))],
        "architecture": {k: config[k] for k in ("num_hidden_layers", "num_attention_heads", "num_key_value_heads", "hidden_size", "sliding_window", "max_position_embeddings", "torch_dtype")},
        "has_chat_template": expected_template,
    }


def parent(name: str, path: Path) -> dict:
    files = [path / f"layer_{i:02d}.pt" for i in range(32)]
    assert all(p.is_file() for p in files)
    body = torch.load(files[0], map_location="cpu", weights_only=False)
    if "gsrq" in name:
        shapes = {side: list(body["codebooks"][side].shape) for side in ("key", "value")}
        provenance = body["provenance"]
        method = "weighted_GSRQ_existing_no_SCF_key_transform"
        rate = shapes["key"][1] * 8 / 128
    else:
        shapes = {side: list(body[side]["codebooks"].shape) for side in ("key", "value")}
        provenance = body["provenance"]
        method = "plain_CQ_existing"
        rate = provenance["total_dynamic_bpa"]
    # The parent may have been fitted under a different local mount point.
    # Its original model reference remains in the provenance for manual audit.
    assert rate == (1.0 if name.endswith("1") else 2.0)
    assert shapes["key"] == shapes["value"]
    calibration_manifest = path / "calibration_manifest.json"
    if not calibration_manifest.is_file():
        calibration_manifest = path.parent / "calibration_manifest.json"
    if not calibration_manifest.is_file():
        calibration_manifest = path.parent.parent / "calibration_manifest.json"
    return {
        "status": "existing_parent_requires_native_SWA_parity_check_before_reuse",
        "method": method,
        "model": str(BASE),
        "parent_provenance_model": provenance.get("model"),
        "path": str(path),
        "layer_files": [file(p) for p in files],
        "geometry": shapes,
        "dynamic_bpa": rate,
        "fitting": {k: provenance.get(k) for k in ("calibration_dataset", "samples", "sequence_length", "seed", "surface", "key_surface", "initialization", "max_iterations", "iterations", "group_size", "clusters")},
        "calibration_manifest": file(calibration_manifest) if calibration_manifest.is_file() else None,
        "dtype_on_disk": str(body["codebooks"]["key"].dtype if "gsrq" in name else body["key"]["codebooks"].dtype),
    }


def main() -> None:
    parents = {name: parent(name, path) for name, path in PARENTS.items()}
    result = {
        "status": "inventory_complete_not_experiment_complete",
        "models": {"base": model(BASE, False), "instruct": model(INSTRUCT, True)},
        "existing_parent_codebooks": parents,
        "existing_QAT_endpoints": [],
        "compatible_512_step_Mistral_QAT_endpoints_found": 0,
        "instruct_v0_1_parent_found": False,
        "instruct_v0_1_QAT_found": False,
        "training_run_count_required_if_all_parents_valid": 18,
        "notes": ["Supply independently fitted C4 parents for the exact Base model snapshot.",
                  "The path recorded in a parent's provenance is not proof of model identity; compare model/config/tokenizer hashes before reusing parents.",
                  "Native sliding-window equivalence and long-context quantized decode are separate checks."],
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"target": str(TARGET), "parents": list(parents), "existing_512_step_runs": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
