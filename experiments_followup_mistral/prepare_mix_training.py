#!/usr/bin/env python3
"""Freeze four-domain Mix streams, retaining exact remote source rows."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
from pathlib import Path

from datasets import Dataset
import requests
import torch
from transformers import AutoTokenizer

from jointkv.table1_protocol import select_document_calibration_windows, tensor_sha256
from experiments.train_full_model_closed_loop_vq import DEFAULT_C4_TRAIN
from release_config import MODEL_BASE, MODEL_INSTRUCT, WIKITEXT_TRAIN_ARROW


ROOT = Path(__file__).resolve().parents[1]
MODEL = {"base": MODEL_BASE, "instruct": MODEL_INSTRUCT}
C4 = Path(DEFAULT_C4_TRAIN).with_name("c4-train-00001-of-00002.arrow")
WIKI = WIKITEXT_TRAIN_ARROW
SEEDS = (8601, 8607, 8613)
LENGTH = 2048
WINDOWS = 128
DATA = ROOT / "experiments_followup_mistral/data"
REMOTE = {
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "text"),
    "codeparrot": ("codeparrot/codeparrot-clean", "content"),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_texts(tokenizer, texts, needed=WINDOWS * LENGTH):
    pieces = []; records = []; total = 0
    for identity, text in texts:
        ids = tokenizer(str(text), return_tensors="pt", add_special_tokens=True)["input_ids"].flatten().long()
        if not ids.numel():
            continue
        pieces.append(ids); total += ids.numel()
        records.append({"source_id": identity, "text_sha256": hashlib.sha256(str(text).encode()).hexdigest(),
                        "tokens": int(ids.numel())})
        if total >= needed:
            break
    if total < needed:
        raise RuntimeError(f"source has only {total} tokens, needs {needed}")
    return torch.cat(pieces)[:needed].view(-1, LENGTH).contiguous(), records


def fetch_remote(tokenizer, name, seed_index):
    dataset, field = REMOTE[name]
    offset = 80000 + seed_index * 10000
    cursor = offset; fetched = []; pages = []
    target = DATA / f"raw_{name}_seed{SEEDS[seed_index]}.jsonl.gz"
    session = requests.Session()
    while True:
        response = session.get("https://datasets-server.huggingface.co/rows",
                               params={"dataset": dataset, "config": "default", "split": "train",
                                       "offset": cursor, "length": 100}, timeout=120)
        response.raise_for_status()
        rows = response.json().get("rows", [])
        if not rows:
            raise RuntimeError(f"remote source exhausted {name} at {cursor}")
        pages.append({"offset": cursor, "length": len(rows), "response_sha256": hashlib.sha256(response.content).hexdigest()})
        for item in rows:
            if item.get("truncated_cells"):
                raise RuntimeError(f"truncated remote row {name}/{item.get('row_idx')}")
            fetched.append((int(item["row_idx"]), str(item["row"][field])))
        cursor += len(rows)
        try:
            tokens, used = encode_texts(tokenizer, fetched)
            break
        except RuntimeError as error:
            if "needs" not in str(error):
                raise
    used_ids = {x["source_id"] for x in used}
    with gzip.open(target, "wt", encoding="utf-8") as f:
        for row_id, text in fetched:
            if row_id in used_ids:
                f.write(json.dumps({"row_idx": row_id, "text": text}, ensure_ascii=False) + "\n")
    return tokens, {"dataset": dataset, "config": "default", "split": "train", "offset": offset,
                    "field": field, "pages": pages, "used_rows": used,
                    "raw_text_snapshot": str(target), "raw_text_snapshot_sha256": sha(target),
                    "token_sha256": tensor_sha256(tokens)}


def main() -> None:
    targets = [ROOT / f"protocols_followup_mistral/{model}_mix_training_manifest.json" for model in MODEL]
    if any(p.exists() for p in targets):
        raise FileExistsError("Mix manifest already exists")
    DATA.mkdir(parents=True, exist_ok=True)
    toks = {model: AutoTokenizer.from_pretrained(path, use_fast=False, local_files_only=True)
            for model, path in MODEL.items()}
    # Both v0.1 snapshots ship exactly the same tokenizer files; bind both
    # snapshots and verify tokenizer behavior before sharing their IDs.
    assert sha(MODEL["base"] / "tokenizer.json") == sha(MODEL["instruct"] / "tokenizer.json")
    c4_dataset = Dataset.from_file(str(C4))
    wiki_dataset = Dataset.from_file(str(WIKI))
    c4_only = json.loads((ROOT / "protocols_followup_mistral/base_c4_training_manifest.json").read_text())
    used_c4 = {int(row["row"]) for run in c4_only["runs"] for row in run["source_rows"]}
    used_c4_hashes = {str(row["token_sha256"]) for run in c4_only["runs"] for row in run["source_rows"]}
    model_runs = {model: [] for model in MODEL}
    wiki_texts = [(i, str(text)) for i, text in enumerate(wiki_dataset["text"])]
    # The WikiText stream follows the historical per-row BOS encoding; distinct
    # contiguous 128-window slices prevent a seed from reusing another seed's text.
    wiki_full, wiki_rows = encode_texts(toks["base"], wiki_texts, needed=len(SEEDS) * WINDOWS * LENGTH)
    for index, seed in enumerate(SEEDS):
        c4_tokens, c4_rows = select_document_calibration_windows(
            toks["base"], c4_dataset, samples=WINDOWS, sequence_length=LENGTH,
            seed=seed + 1000, excluded_document_rows=sorted(used_c4),
            excluded_token_sha256=sorted(used_c4_hashes))
        assert len({r["row"] for r in c4_rows}) == WINDOWS
        used_c4.update(int(r["row"]) for r in c4_rows)
        used_c4_hashes.update(str(r["token_sha256"]) for r in c4_rows)
        wiki_tokens = wiki_full[index * WINDOWS:(index + 1) * WINDOWS]
        fine_tokens, fine_source = fetch_remote(toks["base"], "fineweb_edu", index)
        code_tokens, code_source = fetch_remote(toks["base"], "codeparrot", index)
        pools = {"c4": c4_tokens, "wikitext": wiki_tokens, "fineweb_edu": fine_tokens, "codeparrot": code_tokens}
        order = []
        rng = random.Random(seed + 1901)
        for window in range(WINDOWS):
            domains = list(pools); rng.shuffle(domains)
            order.extend((domain, window) for domain in domains)
        token_batch = torch.stack([pools[domain][window] for domain, window in order]).long().contiguous()
        assert tuple(token_batch.shape) == (512, LENGTH)
        source = {"c4": {"arrow": str(C4), "rows": list(c4_rows)},
                  "wikitext": {"arrow": str(WIKI), "stream_rows": wiki_rows,
                               "slice_window_start": index * WINDOWS},
                  "fineweb_edu": fine_source, "codeparrot": code_source}
        for model_name, tokenizer in toks.items():
            # A separate model-bound manifest accompanies the shared token IDs.
            # The tokenizer-file identity plus this behavior probe is recorded.
            probe = tokenizer("Mistral v0.1 tokenizer parity.", add_special_tokens=True)["input_ids"]
            assert probe == toks["base"]("Mistral v0.1 tokenizer parity.", add_special_tokens=True)["input_ids"]
            path = DATA / f"{model_name}_mix_seed{seed}.pt"
            torch.save({"tokens": token_batch, "seed": seed,
                        "token_sha256": tensor_sha256(token_batch)}, path)
            model_runs[model_name].append({"seed": seed, "steps": 512, "tokens": str(path),
                                           "tokens_file_sha256": sha(path),
                                           "token_sha256": tensor_sha256(token_batch),
                                           "ordered_sources": [{"step": step + 1, "domain": domain, "window": window,
                                                                "token_sha256": tensor_sha256(pools[domain][window:window+1])}
                                                               for step, (domain, window) in enumerate(order)],
                                           "sources": source})
        print(json.dumps({"seed": seed, "mix_token_sha256": tensor_sha256(token_batch),
                          "remote_rows": {"fineweb": len(fine_source["used_rows"]),
                                          "codeparrot": len(code_source["used_rows"])} }), flush=True)
    for model_name, target in zip(MODEL, targets):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"status": "frozen_before_gpu", "model": str(MODEL[model_name]),
                                      "config_sha256": sha(MODEL[model_name] / "config.json"),
                                      "tokenizer_sha256": sha(MODEL[model_name] / "tokenizer.json"),
                                      "schedule": "Mix", "steps": 512, "sequence_length": LENGTH,
                                      "windows_per_domain": WINDOWS, "runs": model_runs[model_name]},
                                     ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
