#!/usr/bin/env python3
"""Freeze independent C4-only Mistral Base training streams for Table 2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from datasets import Dataset
import torch
from transformers import AutoTokenizer

from jointkv.table1_protocol import select_document_calibration_windows, tensor_sha256
from experiments.train_full_model_closed_loop_vq import DEFAULT_C4_TRAIN
from release_config import MODEL_BASE


ROOT = Path(__file__).resolve().parents[1]
MODEL = MODEL_BASE
C4 = Path(DEFAULT_C4_TRAIN).with_name("c4-train-00001-of-00002.arrow")
SEEDS = (8601, 8607, 8613)
OUT = ROOT / "experiments_followup_mistral/data"
MANIFEST = ROOT / "protocols_followup_mistral/base_c4_training_manifest.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False, local_files_only=True)
    dataset = Dataset.from_file(str(C4))
    used_rows: set[int] = set()
    used_hashes: set[str] = set()
    runs = []
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        tokens, rows = select_document_calibration_windows(
            tokenizer, dataset, samples=512, sequence_length=2048, seed=seed,
            excluded_document_rows=sorted(used_rows), excluded_token_sha256=sorted(used_hashes))
        assert tuple(tokens.shape) == (512, 2048)
        new_rows = {int(r["row"]) for r in rows}
        new_hashes = {str(r["token_sha256"]) for r in rows}
        assert len(new_rows) == len(new_hashes) == 512
        assert not (new_rows & used_rows or new_hashes & used_hashes)
        used_rows.update(new_rows)
        used_hashes.update(new_hashes)
        path = OUT / f"base_c4_seed{seed}.pt"
        torch.save({"tokens": tokens.long().contiguous(), "seed": seed,
                    "token_sha256": tensor_sha256(tokens)}, path)
        runs.append({"seed": seed, "steps": 512, "tokens": str(path), "tokens_file_sha256": sha(path),
                     "token_sha256": tensor_sha256(tokens), "source_rows": list(rows)})
        print(json.dumps({"seed": seed, "tokens": str(path), "sha256": tensor_sha256(tokens)}), flush=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"status": "frozen_before_gpu", "model": str(MODEL),
                                    "model_config_sha256": sha(MODEL / "config.json"),
                                    "tokenizer_sha256": sha(MODEL / "tokenizer.json"),
                                    "source": str(C4), "source_sha256": sha(C4),
                                    "schedule": "C4-only", "steps": 512, "sequence_length": 2048,
                                    "runs": runs}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
