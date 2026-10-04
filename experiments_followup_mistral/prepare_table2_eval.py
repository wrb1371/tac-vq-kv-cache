#!/usr/bin/env python3
"""Re-encode the frozen seven-domain Table-2 text panel with Mistral Base."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import time

import requests
import torch
from transformers import AutoTokenizer

from table2_sources import local_texts, REMOTE
from jointkv.table1_protocol import tensor_sha256
from experiments_followup_mistral.train_mistral_v_only import sha
from release_config import MODEL_BASE


ROOT = Path(__file__).resolve().parents[1]
MODEL = MODEL_BASE
TARGET = ROOT / "protocols_followup_mistral/table2_eval_manifest.json"
DATA = ROOT / "experiments_followup_mistral/data"
WINDOWS = {"c4": 64, "wikitext": 64, "fineweb_edu": 64, "codeparrot": 64,
           "arc": 32, "hellaswag": 32, "openbookqa": 32}
LENGTH = 2048


def encode(tokenizer, texts, windows):
    need = windows * LENGTH
    chunks = []; count = 0; used = []
    for idx, text in enumerate(texts):
        text = str(text)
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        if ids:
            chunks.append(torch.tensor(ids, dtype=torch.long)); count += len(ids)
            used.append({"source_record_index": idx,
                         "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "encoded_tokens": len(ids)})
        if count >= need:
            break
    if count < need:
        raise RuntimeError(f"insufficient source text: {count} < {need}")
    return torch.cat(chunks)[:need].reshape(windows, LENGTH).contiguous(), used


def remote(tokenizer, domain, windows):
    spec = REMOTE[domain]
    offset = int(spec["offset"])
    need = windows * LENGTH
    cursor = offset; count = 0; chunks = []; used = []; pages = []; raw = []
    while count < need:
        for attempt in range(6):
            try:
                response = requests.get("https://datasets-server.huggingface.co/rows",
                                        params={"dataset": spec["dataset"], "config": spec["config"],
                                                "split": spec["split"], "offset": cursor, "length": 100},
                                        timeout=120)
                response.raise_for_status()
                break
            except (requests.RequestException, requests.HTTPError):
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 16))
        page = response.json().get("rows", [])
        if not page:
            raise RuntimeError(f"remote source exhausted at {domain}/{cursor}")
        pages.append({"offset": cursor, "length": len(page),
                      "response_sha256": hashlib.sha256(response.content).hexdigest()})
        for item in page:
            if item.get("truncated_cells"):
                raise RuntimeError(f"truncated remote row: {domain}/{item.get('row_idx')}")
            text = str(item["row"][spec["field"]])
            ids = tokenizer(text, add_special_tokens=True)["input_ids"]
            if ids:
                chunks.append(torch.tensor(ids, dtype=torch.long)); count += len(ids)
                used.append({"row_idx": int(item["row_idx"]),
                             "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                             "encoded_tokens": len(ids)})
                raw.append({"row_idx": int(item["row_idx"]), "text": text})
            if count >= need:
                break
        cursor += len(page)
    snapshot = DATA / f"table2_raw_{domain}.jsonl.gz"
    with gzip.open(snapshot, "wt", encoding="utf-8") as f:
        for row in raw:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return torch.cat(chunks)[:need].reshape(windows, LENGTH).contiguous(), used, {
        **spec, "offset": offset, "pages": pages,
        "raw_snapshot": str(snapshot), "raw_snapshot_sha256": sha(snapshot)}


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(TARGET)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False, local_files_only=True)
    entries = {}
    DATA.mkdir(parents=True, exist_ok=True)
    for domain, windows in WINDOWS.items():
        if domain in ("fineweb_edu", "codeparrot"):
            tokens, used, source = remote(tokenizer, domain, windows)
        else:
            texts, source = local_texts(domain)
            tokens, used = encode(tokenizer, texts, windows)
        path = DATA / f"base_table2_{domain}_2048.pt"
        torch.save({"token_ids": tokens, "token_sha256": tensor_sha256(tokens)}, path)
        entries[domain] = {"source": source, "used_records": used,
                           "construction": "joined per-record BOS-encoded text stream, first N full windows",
                           "windows": windows, "sequence_length": LENGTH,
                           "target_tokens_per_window": LENGTH - 1,
                           "token_file": str(path), "token_file_sha256": sha(path),
                           "token_sha256": tensor_sha256(tokens),
                           "window_token_sha256": [tensor_sha256(row[None]) for row in tokens]}
        print(json.dumps({"domain": domain, "windows": windows,
                          "used_source_records": len(used)}), flush=True)
    TARGET.write_text(json.dumps({"status": "frozen_before_gpu", "model": str(MODEL),
                                  "config_sha256": sha(MODEL / "config.json"),
                                  "tokenizer_sha256": sha(MODEL / "tokenizer.json"),
                                  "source_panel": "Meta-Llama-3-8B primary Table 2 seven-domain text rules; remote held-out offset 40000",
                                  "source_runner_sha256": sha(ROOT / "table2_sources.py"),
                                  "corpora": entries}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
