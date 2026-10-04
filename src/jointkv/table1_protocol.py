"""Deterministic data protocol for the legacy section-5.16 PPL matrix.

The old project evaluates C4 and WikiText-2 as independent 8192-token
windows.  This module reproduces that construction without importing or
modifying the old project.  It also exposes the closely related calibration
window selectors so training and evaluation provenance can share one audited
implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import random
from typing import Any, Protocol, Sequence

import torch


TABLE1_SEQUENCE_LENGTH = 8192
TABLE1_C4_WINDOWS = 256


class TextRows(Protocol):
    """Small interface shared by Hugging Face datasets and test doubles."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TokenWindow:
    window: int
    token_start: int
    token_sha256: str
    row: int | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class EvalTokenBatch:
    """Contiguous token storage plus an auditable window manifest."""

    token_ids: torch.Tensor
    dataset: str
    split: str
    sequence_length: int
    available_windows: int
    selected_windows: int
    construction: str
    windows: tuple[TokenWindow, ...]

    @property
    def token_sha256(self) -> str:
        return tensor_sha256(self.token_ids)

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "format": "jointkv_table1_eval_tokens_v1",
            "dataset": self.dataset,
            "split": self.split,
            "sequence_length": self.sequence_length,
            "available_windows": self.available_windows,
            "selected_windows": self.selected_windows,
            "construction": self.construction,
            "token_sha256": self.token_sha256,
            "windows": [window.to_dict() for window in self.windows],
        }


def tensor_sha256(tensor: torch.Tensor) -> str:
    local = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(local.numpy().tobytes()).hexdigest()


def _tokenize(tokenizer: Any, text: str) -> torch.Tensor:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    )
    token_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if token_ids.ndim != 2 or token_ids.shape[0] != 1:
        raise ValueError("tokenizer must return one [1,T] input_ids row")
    return token_ids.to(dtype=torch.long).contiguous()


def _window_hashes(
    token_ids: torch.Tensor,
    sequence_length: int,
    *,
    row_starts: Sequence[tuple[int | None, int]] | None = None,
) -> tuple[TokenWindow, ...]:
    if token_ids.ndim != 2 or token_ids.shape[0] != 1:
        raise ValueError("evaluation tokens must have [1,total_tokens] layout")
    if token_ids.shape[1] % sequence_length:
        raise ValueError("evaluation token storage is not window aligned")
    count = token_ids.shape[1] // sequence_length
    if row_starts is not None and len(row_starts) != count:
        raise ValueError("row/start provenance differs from the window count")
    result = []
    for window in range(count):
        segment = token_ids[
            :, window * sequence_length : (window + 1) * sequence_length
        ]
        row, start = (
            row_starts[window] if row_starts is not None else (None, window * sequence_length)
        )
        result.append(
            TokenWindow(
                window=window,
                row=row,
                token_start=int(start),
                token_sha256=tensor_sha256(segment),
            )
        )
    return tuple(result)


def construct_wikitext2_eval_tokens(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    sequence_length: int = TABLE1_SEQUENCE_LENGTH,
    max_windows: int = 0,
) -> EvalTokenBatch:
    """Join the test split with blank lines, then take contiguous windows."""

    if sequence_length <= 0 or max_windows < 0:
        raise ValueError("sequence length and max windows are invalid")
    encoded = _tokenize(tokenizer, "\n\n".join(texts))
    available = encoded.numel() // sequence_length
    selected = available if max_windows == 0 else min(available, max_windows)
    token_ids = encoded[:, : selected * sequence_length].contiguous()
    return EvalTokenBatch(
        token_ids=token_ids,
        dataset="wikitext2",
        split="test",
        sequence_length=sequence_length,
        available_windows=available,
        selected_windows=selected,
        construction="contiguous_blank_line_join",
        windows=_window_hashes(token_ids, sequence_length),
    )


def construct_c4_eval_tokens(
    tokenizer: Any,
    rows: TextRows,
    *,
    sequence_length: int = TABLE1_SEQUENCE_LENGTH,
    max_windows: int = 0,
    seed: int = 0,
    formal_windows: int = TABLE1_C4_WINDOWS,
) -> EvalTokenBatch:
    """Reproduce the seeded random-document C4 validation construction.

    This intentionally matches the legacy evaluator's slightly unusual upper
    bound ``len(tokens) - sequence_length - 1`` for the random start.
    Documents may repeat, exactly as in the frozen protocol.
    """

    if sequence_length <= 0 or max_windows < 0 or formal_windows <= 0:
        raise ValueError("C4 evaluation geometry is invalid")
    target = formal_windows if max_windows == 0 else min(max_windows, formal_windows)
    generator = random.Random(seed)
    windows: list[torch.Tensor] = []
    provenance: list[tuple[int | None, int]] = []
    while len(windows) < target:
        row = generator.randint(0, len(rows) - 1)
        encoded = _tokenize(tokenizer, rows[row]["text"])
        if encoded.shape[1] < sequence_length:
            continue
        start = generator.randint(
            0,
            max(0, encoded.shape[1] - sequence_length - 1),
        )
        windows.append(encoded[:, start : start + sequence_length])
        provenance.append((row, start))
    token_ids = torch.hstack(windows).contiguous()
    return EvalTokenBatch(
        token_ids=token_ids,
        dataset="c4",
        split="validation",
        sequence_length=sequence_length,
        available_windows=formal_windows,
        selected_windows=target,
        construction="seeded_random_document_windows_with_replacement",
        windows=_window_hashes(
            token_ids,
            sequence_length,
            row_starts=provenance,
        ),
    )


def select_document_calibration_windows(
    tokenizer: Any,
    rows: TextRows,
    *,
    samples: int,
    sequence_length: int,
    seed: int,
    excluded_document_rows: Sequence[int] = (),
    excluded_token_sha256: Sequence[str] = (),
) -> tuple[torch.Tensor, list[dict[str, int | str]]]:
    """Match the old C4 calibration selector (unique sampled documents).

    Optional exclusions are applied *after* an eligible candidate has consumed
    its deterministic row and start RNG draws.  Thus filtering a conflicting
    candidate means taking the next candidate from the same seeded stream,
    rather than perturbing all subsequent identities.  Empty exclusions retain
    the historical behavior exactly.
    """

    if samples <= 0 or sequence_length <= 0:
        raise ValueError("calibration geometry must be positive")
    generator = random.Random(seed)
    windows: list[torch.Tensor] = []
    manifest: list[dict[str, int | str]] = []
    attempted: set[int] = set()
    excluded_rows = {int(row) for row in excluded_document_rows}
    excluded_hashes = {str(digest) for digest in excluded_token_sha256}
    while len(windows) < samples:
        if len(attempted) == len(rows):
            raise ValueError("not enough eligible unique documents")
        row = generator.randrange(len(rows))
        if row in attempted:
            continue
        attempted.add(row)
        encoded = _tokenize(tokenizer, rows[row]["text"])
        if encoded.shape[1] < sequence_length:
            continue
        start = generator.randint(0, encoded.shape[1] - sequence_length)
        window = encoded[:, start : start + sequence_length].contiguous()
        digest = tensor_sha256(window)
        if row in excluded_rows or digest in excluded_hashes:
            continue
        windows.append(window)
        manifest.append(
            {
                "row": row,
                "token_start": start,
                "tokens": sequence_length,
                "token_sha256": digest,
            }
        )
    return torch.cat(windows, dim=0), manifest


def select_wikitext2_calibration_windows(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    samples: int,
    sequence_length: int,
    seed: int,
) -> tuple[torch.Tensor, list[dict[str, int | str]]]:
    """Match the old seeded random starts on contiguous WikiText-2 train."""

    if samples <= 0 or sequence_length <= 0:
        raise ValueError("calibration geometry must be positive")
    encoded = _tokenize(tokenizer, "\n\n".join(texts))
    maximum_start = encoded.shape[1] - sequence_length
    if maximum_start < 0 or samples > maximum_start + 1:
        raise ValueError("requested calibration windows do not fit the corpus")
    starts = random.Random(seed).sample(range(maximum_start + 1), samples)
    windows = []
    manifest = []
    for start in starts:
        window = encoded[:, start : start + sequence_length].contiguous()
        windows.append(window)
        manifest.append(
            {
                "token_start": start,
                "tokens": sequence_length,
                "token_sha256": tensor_sha256(window),
            }
        )
    return torch.cat(windows, dim=0), manifest


def resolve_local_arrow(dataset_root: str | Path, dataset: str, split: str) -> Path:
    """Resolve the exact cached Arrow file used by the legacy project."""

    root = Path(dataset_root).resolve()
    if dataset == "wikitext2":
        relative = (
            Path("Salesforce___wikitext")
            / "wikitext-2-raw-v1"
            / "0.0.0"
            / "b08601e04326c79dfdd32d625aee71d232d685c3"
            / f"wikitext-{split}.arrow"
        )
    elif dataset == "c4" and split == "validation":
        relative = (
            Path("allenai___c4")
            / "default-c7bc8b0aefc5e48f"
            / "0.0.0"
            / "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
            / "c4-validation.arrow"
        )
    else:
        raise ValueError(f"unsupported dataset/split: {dataset}/{split}")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_table1_eval_tokens(
    tokenizer: Any,
    dataset_root: str | Path,
    dataset: str,
    *,
    sequence_length: int = TABLE1_SEQUENCE_LENGTH,
    max_windows: int = 0,
) -> tuple[EvalTokenBatch, Path]:
    """Load local Arrow data and construct the frozen section-5.16 windows."""

    from datasets import Dataset

    split = "test" if dataset == "wikitext2" else "validation"
    path = resolve_local_arrow(dataset_root, dataset, split)
    rows = Dataset.from_file(str(path))
    if dataset == "wikitext2":
        batch = construct_wikitext2_eval_tokens(
            tokenizer,
            rows["text"],
            sequence_length=sequence_length,
            max_windows=max_windows,
        )
    elif dataset == "c4":
        batch = construct_c4_eval_tokens(
            tokenizer,
            rows,
            sequence_length=sequence_length,
            max_windows=max_windows,
        )
    else:
        raise ValueError(dataset)
    return batch, path
