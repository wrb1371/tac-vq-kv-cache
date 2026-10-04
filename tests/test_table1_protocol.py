from types import SimpleNamespace

import torch

from jointkv.table1_protocol import (
    construct_c4_eval_tokens,
    construct_wikitext2_eval_tokens,
    select_document_calibration_windows,
    select_wikitext2_calibration_windows,
    tensor_sha256,
)


class ToyTokenizer:
    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        assert add_special_tokens is True
        values = [999] + [ord(character) for character in text]
        return SimpleNamespace(input_ids=torch.tensor([values], dtype=torch.long))


class ToyRows:
    def __init__(self, texts):
        self.texts = list(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return {"text": self.texts[index]}


def test_wikitext_eval_is_contiguous_and_window_aligned():
    tokenizer = ToyTokenizer()
    result = construct_wikitext2_eval_tokens(
        tokenizer,
        ["abcd", "efgh", "ijkl"],
        sequence_length=4,
        max_windows=2,
    )
    expected = tokenizer(
        "abcd\n\nefgh\n\nijkl", return_tensors="pt", add_special_tokens=True
    ).input_ids[:, :8]
    assert torch.equal(result.token_ids, expected)
    assert result.available_windows == 4
    assert result.selected_windows == 2
    assert result.windows[1].token_start == 4
    assert result.token_sha256 == tensor_sha256(expected)


def test_c4_eval_repeats_the_frozen_seeded_construction():
    tokenizer = ToyTokenizer()
    rows = ToyRows(["a" * 15, "b" * 20, "c" * 25])
    first = construct_c4_eval_tokens(
        tokenizer,
        rows,
        sequence_length=8,
        max_windows=5,
        seed=0,
        formal_windows=9,
    )
    second = construct_c4_eval_tokens(
        tokenizer,
        rows,
        sequence_length=8,
        max_windows=5,
        seed=0,
        formal_windows=9,
    )
    assert torch.equal(first.token_ids, second.token_ids)
    assert first.windows == second.windows
    assert first.available_windows == 9
    assert first.selected_windows == 5
    assert all(window.row is not None for window in first.windows)


def test_document_calibration_uses_unique_rows_and_hashes():
    tokens, manifest = select_document_calibration_windows(
        ToyTokenizer(),
        ToyRows(["a" * 20, "b" * 20, "c" * 20, "d" * 20]),
        samples=3,
        sequence_length=8,
        seed=29,
    )
    assert tokens.shape == (3, 8)
    assert len({row["row"] for row in manifest}) == 3
    for window, row in zip(tokens, manifest):
        assert row["token_sha256"] == tensor_sha256(window.unsqueeze(0))


def test_document_calibration_exclusion_takes_next_seeded_candidate():
    rows = ToyRows([character * 20 for character in "abcdefgh"])
    baseline_tokens, baseline_manifest = select_document_calibration_windows(
        ToyTokenizer(), rows, samples=4, sequence_length=8, seed=29
    )
    excluded_tokens, excluded_manifest = select_document_calibration_windows(
        ToyTokenizer(),
        rows,
        samples=3,
        sequence_length=8,
        seed=29,
        excluded_document_rows=[baseline_manifest[1]["row"]],
    )
    assert excluded_manifest == [
        baseline_manifest[0],
        baseline_manifest[2],
        baseline_manifest[3],
    ]
    assert torch.equal(excluded_tokens, baseline_tokens[[0, 2, 3]])


def test_wikitext_calibration_is_seeded_and_unique():
    kwargs = dict(samples=4, sequence_length=8, seed=29)
    first, first_manifest = select_wikitext2_calibration_windows(
        ToyTokenizer(), ["abcdefghijklmnopqrstuvwxyz"], **kwargs
    )
    second, second_manifest = select_wikitext2_calibration_windows(
        ToyTokenizer(), ["abcdefghijklmnopqrstuvwxyz"], **kwargs
    )
    assert torch.equal(first, second)
    assert first_manifest == second_manifest
    assert len({row["token_start"] for row in first_manifest}) == 4
