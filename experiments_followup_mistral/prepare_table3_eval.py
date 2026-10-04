#!/usr/bin/env python3
"""Freeze all 1,350 Mistral-native LongBench-7 prompts before GPU evaluation."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from experiments_followup_mistral.train_mistral_v_only import sha
from release_config import MODEL_INSTRUCT, LONGBENCH_SOURCE, LONGBENCH_DATA


ROOT = Path(__file__).resolve().parents[1]
MODEL = MODEL_INSTRUCT
SOURCE = LONGBENCH_SOURCE
DATASET = LONGBENCH_DATA
TASKS = ("narrativeqa", "qasper", "multifieldqa_en", "qmsum", "musique", "triviaqa", "samsum")
COUNTS = (200, 200, 150, 200, 200, 200, 200)
LIMITS = {"narrativeqa": 128, "qasper": 64, "multifieldqa_en": 64, "qmsum": 512,
          "musique": 32, "triviaqa": 32, "samsum": 128}
TARGET = ROOT / "protocols_followup_mistral/table3_eval_manifest.json"
PROMPTS = ROOT / "experiments_followup_mistral/data/table3_instruct_inputs.jsonl.gz"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    if TARGET.exists() or PROMPTS.exists():
        raise FileExistsError("Table 3 frozen inputs already exist")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    templates = json.loads((SOURCE / "config/dataset2prompt.json").read_text())
    official_limits = json.loads((SOURCE / "config/dataset2maxlen.json").read_text())
    sources = {}; count = 0; truncations = {}
    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(PROMPTS, "wt", encoding="utf-8") as stream:
        for task, expected in zip(TASKS, COUNTS):
            path = DATASET / f"{task}.jsonl"
            sources[task] = {"path": str(path), "sha256": sha(path),
                             "expected_rows": expected,
                             "official_max_new_tokens": official_limits[task],
                             "frozen_max_new_tokens": LIMITS[task]}
            actual = 0; truncated_count = 0
            with path.open(encoding="utf-8") as source:
                for index, line in enumerate(source):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    prompt = templates[task].format(**row)
                    input_limit = 8192 - LIMITS[task] - 64
                    plain_ids = tokenizer.encode(prompt, add_special_tokens=False)
                    shortened = len(plain_ids) > input_limit
                    if shortened:
                        half = input_limit // 2
                        prompt = tokenizer.decode(plain_ids[:half], skip_special_tokens=True) + tokenizer.decode(
                            plain_ids[-half:], skip_special_tokens=True)
                        truncated_count += 1
                    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                              tokenize=False, add_generation_prompt=True)
                    ids = tokenizer.encode(rendered, add_special_tokens=False)
                    if len(ids) + LIMITS[task] > 8192:
                        raise RuntimeError(f"native total overflow: {task}/{index}/{len(ids)}")
                    record = {"global_index": count, "task": task, "row": index,
                              "source_line_sha256": hash_text(line.rstrip("\n")),
                              "input_ids": ids, "input_tokens": len(ids),
                              "token_sha256": hashlib.sha256(bytes().join(
                                  int(v).to_bytes(4, "little") for v in ids)).hexdigest(),
                              "middle_truncated": shortened, "answers": row["answers"],
                              "all_classes": row.get("all_classes"), "source_length": row.get("length"),
                              "max_new_tokens": LIMITS[task], "stop_token_ids": [int(tokenizer.eos_token_id),
                                  *([int(tokenizer.encode("\n", add_special_tokens=False)[-1])] if task == "samsum" else [])],
                              "prohibit_first_eos": task == "samsum"}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    actual += 1; count += 1
            if actual != expected:
                raise RuntimeError(f"{task} has {actual} rows, expected {expected}")
            truncations[task] = truncated_count
            print(json.dumps({"task": task, "rows": actual, "middle_truncated": truncated_count}), flush=True)
    if count != 1350:
        raise RuntimeError(f"unexpected Table 3 total {count}")
    TARGET.write_text(json.dumps({"status": "frozen_before_gpu", "model": str(MODEL),
                                  "config_sha256": sha(MODEL / "config.json"),
                                  "tokenizer_sha256": sha(MODEL / "tokenizer.json"),
                                  "tokenizer_config_sha256": sha(MODEL / "tokenizer_config.json"),
                                  "chat_template": "Mistral-7B-Instruct-v0.1 native user template for all seven tasks",
                                  "template_sha256": hash_text(tokenizer.chat_template),
                                  "prompt_template_source": {"path": str(SOURCE / "config/dataset2prompt.json"),
                                                              "sha256": sha(SOURCE / "config/dataset2prompt.json")},
                                  "scorer_source": {"path": str(SOURCE / "metrics.py"),
                                                    "sha256": sha(SOURCE / "metrics.py")},
                                  "tasks": list(TASKS), "task_counts": dict(zip(TASKS, COUNTS)),
                                  "task_max_new_tokens": LIMITS, "native_total_limit": 8192,
                                  "middle_truncation_counts": truncations,
                                  "prompt_records": str(PROMPTS), "prompt_records_sha256": sha(PROMPTS),
                                  "source_files": sources, "examples": count,
                                  "generation": "greedy; EOS; SAMSum newline and first-EOS handling",
                                  "score_handling": "QA max reference F1, QMSum/SAMSum official ROUGE, TriviaQA/SAMSum first output line"},
                                 ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
