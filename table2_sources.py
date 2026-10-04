"""Public, path-configurable text construction for the seven-domain PPL panel."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset

from release_config import INPUTS


LOCAL = {
    name: Path(os.environ.get(f"TACVQ_{name.upper()}_ARROW", str(INPUTS / "datasets" / name / f"{name}.arrow")))
    for name in ("c4", "wikitext", "arc", "hellaswag", "openbookqa")
}
ARC_EASY = Path(os.environ.get("TACVQ_ARC_EASY_ARROW", str(INPUTS / "datasets/arc/arc_easy.arrow")))
REMOTE = {
    "fineweb_edu": {"dataset": "HuggingFaceFW/fineweb-edu", "config": "default", "split": "train", "offset": 40000, "field": "text"},
    "codeparrot": {"dataset": "codeparrot/codeparrot-clean", "config": "default", "split": "train", "offset": 40000, "field": "content"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_texts(domain: str) -> tuple[Iterable[str], dict[str, Any]]:
    path = LOCAL[domain]
    dataset = Dataset.from_file(str(path))
    if domain in ("c4", "wikitext"):
        texts = (str(row["text"]) for row in dataset)
    elif domain == "arc":
        easy = Dataset.from_file(str(ARC_EASY))
        texts = (
            str(row["question"]) + "\n" + "\n".join(str(x) for x in row["choices"]["text"])
            for local in (dataset, easy) for row in local
        )
    elif domain == "hellaswag":
        texts = (str(row["ctx"]) + "\n" + "\n".join(str(x) for x in row["endings"]) for row in dataset)
    elif domain == "openbookqa":
        texts = (str(row["question_stem"]) + "\n" + "\n".join(str(x) for x in row["choices"]["text"]) for row in dataset)
    else:
        raise ValueError(domain)
    if domain == "arc":
        return texts, {"kind": "local_arrow_read_only_concatenated_splits",
                       "paths": [{"path": str(p), "sha256": sha256_file(p), "rows": len(d), "subset": label}
                                 for p, d, label in ((path, dataset, "ARC-Challenge"), (ARC_EASY, easy, "ARC-Easy"))],
                       "order": ["ARC-Challenge", "ARC-Easy"]}
    return texts, {"kind": "local_arrow_read_only", "path": str(path),
                   "sha256": sha256_file(path), "rows": len(dataset)}
