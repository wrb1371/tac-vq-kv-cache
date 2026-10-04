"""Local input locations for the source-only research release.

Paths may be overridden by environment variables. No model weights, parent
codebooks, or benchmark data are shipped in this repository.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "external_inputs"


def location(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable, str(default))).expanduser().resolve()


MODEL_ROOT = location("TACVQ_MODEL_ROOT", INPUTS / "models")
MODEL_BASE = location("TACVQ_MODEL_BASE", MODEL_ROOT / "Mistral-7B-v0.1")
MODEL_INSTRUCT = location("TACVQ_MODEL_INSTRUCT", MODEL_ROOT / "Mistral-7B-Instruct-v0.1")
LLAMA3_BASE = location("TACVQ_LLAMA3_BASE", MODEL_ROOT / "Meta-Llama-3-8B")
C4_TRAIN_ARROW_0 = location("TACVQ_C4_TRAIN_ARROW_0", INPUTS / "datasets/c4/c4-train-00000-of-00002.arrow")
WIKITEXT_TRAIN_ARROW = location("TACVQ_WIKITEXT_TRAIN_ARROW", INPUTS / "datasets/wikitext/wikitext-train.arrow")
HF_DATASETS_ROOT = location("TACVQ_HF_DATASETS_ROOT", INPUTS / "datasets")
LONGBENCH_SOURCE = location("TACVQ_LONGBENCH_SOURCE", INPUTS / "LongBench/LongBench")
LONGBENCH_DATA = location("TACVQ_LONGBENCH_DATA", INPUTS / "longbench/data")
PARENT_ROOT = location("TACVQ_PARENT_ROOT", INPUTS / "parents")
