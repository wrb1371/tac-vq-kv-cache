#!/usr/bin/env python3
"""Create Mistral-tokenized C4 and joined WikiText-2 test canonical streams."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from jointkv.table1_protocol import load_table1_eval_tokens, tensor_sha256
from release_config import MODEL_BASE, HF_DATASETS_ROOT


ROOT = Path(__file__).resolve().parents[1]
MODEL = MODEL_BASE
DATASET_ROOT = HF_DATASETS_ROOT
TARGET = ROOT / "protocols_followup_mistral/table1_eval_manifest.json"
DATA = ROOT / "experiments_followup_mistral/data"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(TARGET)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False, local_files_only=True)
    entries = {}
    DATA.mkdir(parents=True, exist_ok=True)
    for name in ("c4", "wikitext2"):
        batch, arrow = load_table1_eval_tokens(tokenizer, DATASET_ROOT, name,
                                               sequence_length=8192, max_windows=0)
        assert batch.selected_windows == 256 if name == "c4" else batch.selected_windows > 0
        assert batch.token_ids.shape == (1, batch.selected_windows * 8192)
        path = DATA / f"base_table1_{name}_8192.pt"
        torch.save({"token_ids": batch.token_ids.long().contiguous(),
                    "token_sha256": tensor_sha256(batch.token_ids)}, path)
        entries[name] = {"source_arrow": str(arrow), "source_arrow_sha256": sha(arrow),
                         "construction": batch.construction, "split": batch.split,
                         "windows": batch.selected_windows, "sequence_length": 8192,
                         "target_tokens_per_window": 8191,
                         "token_file": str(path), "token_file_sha256": sha(path),
                         "token_sha256": tensor_sha256(batch.token_ids),
                         "window_manifest": [row.to_dict() for row in batch.windows]}
        print(json.dumps({"corpus": name, "windows": batch.selected_windows,
                          "tokens": int(batch.token_ids.numel())}), flush=True)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps({"status": "frozen_before_gpu", "model": str(MODEL),
                                  "config_sha256": sha(MODEL / "config.json"),
                                  "tokenizer_sha256": sha(MODEL / "tokenizer.json"),
                                  "precision": "FP16 weights / FP16 deployed codewords; model native SDPA/SWA",
                                  "corpora": entries}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
