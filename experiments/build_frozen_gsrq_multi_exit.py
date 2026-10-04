#!/usr/bin/env python3
"""Build parent-preserving multi-exit artifacts from frozen GSRQ-family parents.

The rate-specific parent is copied byte-exact at the uniform exit.  Optional
higher stages are fitted only on the residual produced by that same frozen
parent.  SCFW parents preserve their FP32 key assignment whitener; published
GSRQ parents can instead opt into an explicitly audited Euclidean assignment
path.  Parent codewords are never optimized and no evaluation-domain data is
consumed.

All writes are confined to the requested output directory.  The legacy GSRQ
artifacts and the reusable calibration captures are read-only inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from jointkv.nested_vq import factorized_residual_vq_reconstructions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-codebook-dir", required=True)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--uniform-depth", type=int, required=True)
    parser.add_argument(
        "--action-depths",
        required=True,
        help="three sorted positive depths containing --uniform-depth",
    )
    parser.add_argument("--effective-bits", type=float, required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kmeans-iterations", type=int, default=10)
    parser.add_argument("--token-chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=9301)
    parser.add_argument(
        "--euclidean-key-assignment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "accept only a parent whose provenance records key_transform=None, "
            "fit/search K in raw Euclidean coordinates, and serialize an exact "
            "FP32 identity transform for downstream compatibility"
        ),
    )
    parser.add_argument(
        "--gradient-weight-dir",
        default=None,
        help=(
            "optional formal CE projection-gradient directory; when set, "
            "the extension GSKM M-step uses paper-style log-gradient weights"
        ),
    )
    parser.add_argument("--gradient-weight-target-median", type=float, default=1.0)
    parser.add_argument(
        "--gradient-weight-norm-scope",
        choices=("per_head", "whole_token_shared"),
        default="per_head",
    )
    parser.add_argument(
        "--fit-sample-indices",
        default="all",
        help=(
            "capture rows used to fit extension stages; use e.g. 0,2,4,6 "
            "to keep the complementary rows clean for validation"
        ),
    )
    parser.add_argument(
        "--validation-sample-indices",
        default="",
        help=(
            "optional disjoint capture rows recorded as untouched validation; "
            "they are never selected into the extension fit tensors"
        ),
    )
    parser.add_argument(
        "--extension-fit-mode",
        choices=("gskm", "lloyd"),
        default="gskm",
        help=(
            "fit new stages with the parent's gain--shape geometry (default); "
            "plain Lloyd is retained only as a diagnostic"
        ),
    )
    parser.add_argument(
        "--extension-gain-cap-multiplier",
        type=float,
        default=1.0,
        help=(
            "cap each new GSKM gain by this multiple of the same head's "
            "largest gain in the frozen parent's final stages"
        ),
    )
    parser.add_argument(
        "--extension-gain-cap-tail-stages",
        type=int,
        default=4,
        help="number of frozen parent tail stages defining the gain envelope",
    )
    parser.add_argument(
        "--extension-zero-codeword",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "reserve index zero of every new stage as an exact no-op; this "
            "makes deployed assignment distortion non-increasing"
        ),
    )
    parser.add_argument(
        "--require-capture-sequence-length",
        type=int,
        default=None,
        help="fail unless each capture uses this context length",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_actions(spec: str, uniform_depth: int) -> tuple[int, int, int]:
    raw = tuple(int(item) for item in spec.split(",") if item.strip())
    if (
        len(raw) != 3
        or tuple(sorted(set(raw))) != raw
        or raw[0] <= 0
        or uniform_depth not in raw
    ):
        raise ValueError("actions must be three sorted positive depths containing uniform")
    return raw  # type: ignore[return-value]


def parse_layers(spec: str, available: list[int]) -> list[int]:
    if spec == "all":
        return available
    selected: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, stop = (int(value) for value in item.split("-", 1))
            selected.update(range(start, stop + 1))
        else:
            selected.add(int(item))
    result = sorted(selected)
    if not result or not set(result).issubset(available):
        raise ValueError("selected layers are outside the parent")
    return result


def parse_sample_indices(
    spec: str,
    *,
    samples: int,
    name: str,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if spec.strip() == "all":
        return tuple(range(samples))
    values = tuple(int(item.strip()) for item in spec.split(",") if item.strip())
    if (
        (not values and not allow_empty)
        or tuple(sorted(set(values))) != values
        or (values and (values[0] < 0 or values[-1] >= samples))
    ):
        raise ValueError(
            f"{name} must be sorted unique capture rows inside [0,{samples})"
        )
    return values


def _sample_core(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop capture-only domain labels before exact token-window comparison."""

    fields = ("row", "token_start", "tokens", "token_sha256")
    return [
        {field: row[field] for field in fields if field in row}
        for row in rows
    ]


def _log_gradient_weights(
    norms: torch.Tensor, *, target_median: float
) -> tuple[torch.Tensor, dict[str, float | str]]:
    raw = norms.detach().float().reshape(-1)
    if (
        raw.numel() == 0
        or target_median <= 0
        or not bool(torch.isfinite(raw).all())
        or bool((raw < 0).any())
    ):
        raise ValueError("gradient norms and target median must be finite and positive")
    median = raw.median()
    if not bool(median > 0):
        raise ValueError("gradient-norm median must be positive")
    scale = float(target_median) / (median + 1e-12)
    weights = torch.log1p(raw * scale)
    if not bool(torch.isfinite(weights).all()) or not bool(weights.sum() > 0):
        raise ValueError("log-gradient weights are invalid")
    return weights, {
        "formula": "log1p(tau * gradient_norm / (median + epsilon))",
        "target_median": float(target_median),
        "raw_median": float(median),
        "lambda": float(scale),
        "weight_min": float(weights.min()),
        "weight_median": float(weights.median()),
        "weight_max": float(weights.max()),
    }


def load_ce_gradient_weights(
    directory: Path,
    *,
    layer: int,
    kind: str,
    capture: dict[str, Any],
    sample_indices: tuple[int, ...],
    model: str,
    expected_heads: int,
    target_median: float,
    norm_scope: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load exact formal CE norms and return fitted-row `[H,B*T]` weights.

    Median normalization is deliberately applied *after* selecting the rows
    consumed by this fit.  This is the same per-layer/head/kind normalization
    scope as the formal SCFW-GSRQ calibrator and prevents held-out rows from
    leaking into the extension codebook even through one scalar statistic.
    """

    if kind not in {"key", "value"}:
        raise ValueError("gradient kind must be key or value")
    manifest_path = directory / "gradient_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_manifest = list(capture.get("sample_manifest", ()))
    samples = int(capture["raw_key"].shape[0])
    sequence_length = int(capture["raw_key"].shape[2])
    selected_layers = [int(value) for value in manifest.get("selected_layers", ())]
    checks = {
        "format": manifest.get("format") == "gsrq_final_ce_projection_gradient_norms_v2",
        "status": manifest.get("status") == "complete",
        "model": os.path.realpath(str(manifest.get("model", ""))) == model,
        "samples": int(manifest.get("samples", -1)) == samples,
        "sequence_length": int(manifest.get("sequence_length", -1)) == sequence_length,
        "kv_heads": int(manifest.get("kv_heads", -1)) == expected_heads,
        "sample_manifest": _sample_core(list(manifest.get("sample_manifest", ())))
        == _sample_core(sample_manifest),
        "forward_replay": float(
            manifest.get("forward_replay_max_abs_logit_error", float("nan"))
        )
        == 0.0,
        "saved_gradient_scale": float(
            manifest.get("saved_gradient_scale", float("nan"))
        )
        == 1.0,
        "layer": layer in selected_layers,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"CE gradient provenance differs in {failed}")
    stored_layer = selected_layers.index(layer)
    rows = []
    for sample in sample_indices:
        expected = sample_manifest[sample]
        path = directory / "windows" / f"window_{sample:03d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        values = payload.get(f"{kind}_gradient_norms")
        if (
            payload.get("format") != manifest["format"]
            or payload.get("token_sha256") != expected["token_sha256"]
            or [int(value) for value in payload.get("selected_layers", ())]
            != selected_layers
            or float(payload.get("loss_scale", float("nan")))
            != float(manifest["loss_scale"])
            or not isinstance(values, torch.Tensor)
            or tuple(values.shape) != (
                len(selected_layers), sequence_length, expected_heads
            )
        ):
            raise ValueError(f"CE gradient window {sample} differs")
        rows.append(values[stored_layer].float())
    norms = torch.stack(rows)  # [B_fit,T,H]
    if norm_scope == "whole_token_shared":
        norms = (
            norms.square()
            .sum(dim=-1, keepdim=True)
            .sqrt()
            .expand(-1, -1, expected_heads)
        )
    weights = []
    diagnostics = []
    for head in range(expected_heads):
        local, row = _log_gradient_weights(
            norms[:, :, head], target_median=target_median
        )
        weights.append(local)
        diagnostics.append({"head": head, **row})
    return torch.stack(weights), {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "loss": manifest["loss"],
        "gradient_surface": manifest["gradient_surface"],
        "norm_scope": norm_scope,
        "fit_sample_indices": list(sample_indices),
        "fit_samples": len(sample_indices),
        "per_head": diagnostics,
    }


def direct_part(tables: torch.Tensor) -> dict[str, Any]:
    heads, stages, cardinality, dimension = map(int, tables.shape)
    return {
        "shared_codebooks": [
            tables[:, stage].contiguous() for stage in range(stages)
        ],
        "entry_factors": [
            torch.empty((heads, cardinality, 0), dtype=tables.dtype)
            for _ in range(stages)
        ],
        "dimension_factors": [
            torch.empty((heads, 0, dimension), dtype=tables.dtype)
            for _ in range(stages)
        ],
        "residual_mse_by_stage": [],
        "representation": "per_head",
    }


@torch.no_grad()
def nearest_indices(
    states: torch.Tensor,
    centroids: torch.Tensor,
    *,
    assignment_whiteners: torch.Tensor | None,
    token_chunk_size: int,
) -> torch.Tensor:
    heads, tokens, dimension = states.shape
    if centroids.shape[0] != heads or centroids.shape[2] != dimension:
        raise ValueError("state/centroid geometry differs")
    if assignment_whiteners is None:
        selector_states, selector_centroids = states, centroids
    else:
        selector_states = torch.matmul(
            states, assignment_whiteners.transpose(-1, -2)
        )
        selector_centroids = torch.matmul(
            centroids, assignment_whiteners.transpose(-1, -2)
        )
    norms = selector_centroids.square().sum(dim=-1)
    result = torch.empty((heads, tokens), dtype=torch.long, device=states.device)
    for start in range(0, tokens, token_chunk_size):
        stop = min(tokens, start + token_chunk_size)
        local = selector_states[:, start:stop]
        distances = (
            local.square().sum(dim=-1, keepdim=True)
            + norms[:, None, :]
            - 2.0 * torch.matmul(local, selector_centroids.transpose(-1, -2))
        )
        result[:, start:stop] = distances.argmin(dim=-1)
    return result


@torch.no_grad()
def fit_residual_stages(
    residual: torch.Tensor,
    *,
    stages: int,
    assignment_whiteners: torch.Tensor | None,
    initial_selector_residual: torch.Tensor | None,
    sample_weights: torch.Tensor | None = None,
    kmeans_iterations: int,
    token_chunk_size: int,
    seed: int,
    fit_mode: str,
    gain_caps: torch.Tensor | None,
    reserve_zero_codeword: bool = True,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    """Fit direct per-head residual stages in the parent's deployed metric.

    ``residual`` is the raw-coordinate residual at the frozen parent exit.
    For SCFW K, initialize its selector-space image once and recursively
    subtract each transformed extension word.  This is the exact arithmetic
    contract used by GSRQ inference and avoids per-stage remapping drift.

    The default update is GSRQ's gain--shape k-means: assignments use the
    deployed selector metric, while every raw centroid is constrained to one
    non-negative gain times a unit mean direction.  An unconstrained Euclidean
    Lloyd mean can dedicate a centroid to a single activation outlier and
    create enormous extension codewords; that was the source of the first M30
    failure and is intentionally kept behind the explicit ``lloyd`` ablation.
    """

    if stages <= 0:
        return torch.empty(
            residual.shape[0], 0, 256, residual.shape[-1], dtype=torch.float16
        ), []
    heads, tokens, _dimension = residual.shape
    if sample_weights is not None:
        sample_weights = sample_weights.to(
            device=residual.device, dtype=torch.float32
        )
        if (
            tuple(sample_weights.shape) != (heads, tokens)
            or not bool(torch.isfinite(sample_weights).all())
            or bool((sample_weights < 0).any())
            or bool((sample_weights.sum(dim=1) <= 0).any())
        ):
            raise ValueError(
                "sample weights must be finite non-negative [H,N] with "
                "positive mass for every head"
            )
    if fit_mode not in {"gskm", "lloyd"}:
        raise ValueError("extension fit mode must be gskm or lloyd")
    if assignment_whiteners is None:
        if initial_selector_residual is not None:
            raise ValueError("selector residual requires assignment whiteners")
        selector_residual = None
    else:
        if initial_selector_residual is None:
            raise ValueError("whitened extension requires the exact parent selector residual")
        if initial_selector_residual.shape != residual.shape:
            raise ValueError("raw/selector parent residual geometry differs")
        selector_residual = initial_selector_residual.clone()
    if gain_caps is not None:
        gain_caps = gain_caps.to(device=residual.device, dtype=torch.float32)
        if (
            tuple(gain_caps.shape) != (heads,)
            or not bool(torch.isfinite(gain_caps).all())
            or bool((gain_caps <= 0).any())
        ):
            raise ValueError("extension gain caps must be finite positive [H]")
    generator = torch.Generator(device=residual.device).manual_seed(int(seed))
    tables = []
    history: list[dict[str, float | int]] = []
    for stage in range(stages):
        before = float(residual.square().mean())
        working = residual if selector_residual is None else selector_residual
        seeds = []
        seed_gains = []
        learned_entries = 255 if reserve_zero_codeword else 256
        for head in range(heads):
            chosen = torch.randperm(
                tokens, generator=generator, device=residual.device
            )[:learned_entries]
            sampled = working[head].index_select(0, chosen)
            local_seeds = (
                F.normalize(sampled, dim=-1, eps=1e-8)
                if fit_mode == "gskm"
                else residual[head].index_select(0, chosen)
            )
            local_gains = torch.linalg.vector_norm(sampled, dim=-1)
            if reserve_zero_codeword:
                local_seeds = torch.cat(
                    (torch.zeros_like(local_seeds[:1]), local_seeds), dim=0
                )
                local_gains = torch.cat(
                    (torch.zeros_like(local_gains[:1]), local_gains), dim=0
                )
            seeds.append(local_seeds)
            seed_gains.append(local_gains)
        if fit_mode == "gskm":
            shapes = torch.stack(seeds)
            # A gain--shape seed must represent the sampled residual itself.
            # Initializing every gain to one leaves hundreds of out-of-scale,
            # unoccupied codewords after the first low-energy residual stage;
            # those stale entries can be selected under distribution/trajectory
            # shift and cause a catastrophic reconstruction jump.
            gains = torch.stack(seed_gains)
            if gain_caps is not None:
                gains = torch.minimum(gains, gain_caps[:, None])
        else:
            shapes = None
            gains = None
            centroids = torch.stack(seeds)

        def deployed_tables() -> tuple[torch.Tensor, torch.Tensor]:
            """Return FP16-roundtripped raw and selector codewords."""

            if fit_mode == "gskm":
                assert shapes is not None and gains is not None
                desired_selector = gains[..., None] * shapes
                if assignment_whiteners is None:
                    raw = desired_selector
                else:
                    raw = torch.linalg.solve_triangular(
                        assignment_whiteners,
                        desired_selector.transpose(-1, -2),
                        upper=True,
                    ).transpose(-1, -2)
                raw = raw.half().float()
            else:
                raw = centroids.half().float()
            selector = (
                raw
                if assignment_whiteners is None
                else torch.matmul(
                    raw, assignment_whiteners.transpose(-1, -2)
                )
            )
            return raw, selector

        for _iteration in range(kmeans_iterations):
            raw_centroids, selector_centroids = deployed_tables()
            indices = nearest_indices(
                working,
                selector_centroids,
                assignment_whiteners=None,
                token_chunk_size=token_chunk_size,
            )
            for head in range(heads):
                counts = torch.bincount(indices[head], minlength=256)
                occupied = counts > 0
                if fit_mode == "gskm":
                    assert shapes is not None and gains is not None
                    # This is GSRQ in its actual assignment coordinate system:
                    # K is updated in frozen FP32 selector space and only then
                    # inverse-mapped/FP16-rounded for raw-cache decoding.
                    normalized = F.normalize(working[head], dim=-1, eps=1e-8)
                    shape_sums = torch.zeros_like(shapes[head])
                    raw_sums = torch.zeros_like(shapes[head])
                    if sample_weights is None:
                        local_weights = None
                        weighted_normalized = normalized
                        weighted_working = working[head]
                        denominators = counts.to(working.dtype)
                    else:
                        local_weights = sample_weights[head]
                        weighted_normalized = normalized * local_weights[:, None]
                        weighted_working = working[head] * local_weights[:, None]
                        denominators = torch.zeros(
                            256, device=residual.device, dtype=torch.float32
                        )
                        denominators.index_add_(
                            0, indices[head], local_weights
                        )
                    shape_sums.index_add_(
                        0, indices[head], weighted_normalized
                    )
                    raw_sums.index_add_(
                        0, indices[head], weighted_working
                    )
                    shape_norms = torch.linalg.vector_norm(shape_sums, dim=-1)
                    valid = denominators.gt(0) & shape_norms.gt(1e-8)
                    if reserve_zero_codeword:
                        valid[0] = False
                    shapes[head, valid] = (
                        shape_sums[valid] / shape_norms[valid, None]
                    )
                    projected = (
                        raw_sums[valid] * shapes[head, valid]
                    ).sum(dim=-1)
                    gains[head, valid] = (
                        projected / denominators[valid].to(projected.dtype)
                    ).clamp_min_(0.0)
                    if gain_caps is not None:
                        gains[head].clamp_(max=float(gain_caps[head]))
                    if reserve_zero_codeword:
                        shapes[head, 0].zero_()
                        gains[head, 0] = 0
                else:
                    updated = centroids[head].clone()
                    sums = torch.zeros_like(centroids[head])
                    sums.index_add_(0, indices[head], residual[head])
                    if reserve_zero_codeword:
                        occupied[0] = False
                    updated[occupied] = (
                        sums[occupied] / counts[occupied, None].to(sums.dtype)
                    )
                    if reserve_zero_codeword:
                        updated[0].zero_()
                    centroids[head] = updated
            # ``deployed_tables`` applies FP16 roundtrip before every assignment,
            # matching the actual cache decoder instead of a float32 surrogate.
        raw_centroids, selector_centroids = deployed_tables()
        indices = nearest_indices(
            working,
            selector_centroids,
            assignment_whiteners=None,
            token_chunk_size=token_chunk_size,
        )
        selected = torch.stack(
            [raw_centroids[head].index_select(0, indices[head]) for head in range(heads)]
        )
        residual.sub_(selected)
        if selector_residual is not None:
            selector_selected = torch.stack(
                [
                    selector_centroids[head].index_select(0, indices[head])
                    for head in range(heads)
                ]
            )
            selector_residual.sub_(selector_selected)
        after = float(residual.square().mean())
        tables.append(raw_centroids.half().cpu())
        history.append(
            {
                "stage": stage,
                "residual_element_mse_before": before,
                "residual_element_mse_after": after,
                "occupied_clusters_min": int(
                    min(torch.unique(indices[head]).numel() for head in range(heads))
                ),
                "fit_mode": fit_mode,
                "maximum_codeword_norm": float(
                    torch.linalg.vector_norm(raw_centroids, dim=-1).max()
                ),
                "maximum_selector_codeword_norm": float(
                    torch.linalg.vector_norm(selector_centroids, dim=-1).max()
                ),
                "gain_cap_min": (
                    float(gain_caps.min()) if gain_caps is not None else -1.0
                ),
                "gain_cap_max": (
                    float(gain_caps.max()) if gain_caps is not None else -1.0
                ),
                "zero_codeword_reserved": int(reserve_zero_codeword),
                "zero_assignment_fraction": float(indices.eq(0).float().mean()),
                "gradient_weighted": int(sample_weights is not None),
            }
        )
    return torch.stack(tables, dim=1), history


@torch.no_grad()
def parent_residuals_and_reconstruction(
    states: torch.Tensor,
    tables: torch.Tensor,
    whitener: torch.Tensor | None,
    *,
    depth: int,
    token_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Return exact deployed raw/selector residuals at the frozen exit."""

    batch, heads, tokens, dimension = states.shape
    raw_residual = (
        states.permute(1, 0, 2, 3)
        .reshape(heads, batch * tokens, dimension)
        .float()
        .clone()
    )
    decoded = torch.zeros_like(raw_residual)
    local_whitener = (
        None
        if whitener is None
        else whitener.to(device=states.device, dtype=torch.float32)
    )
    selector_residual = None
    selector_tables = None
    if local_whitener is not None:
        selector_residual = (
            torch.einsum("hde,bhte->bhtd", local_whitener, states.float())
            .permute(1, 0, 2, 3)
            .reshape(heads, batch * tokens, dimension)
            .float()
            .clone()
        )
        selector_tables = torch.einsum(
            "hrke,hde->hrkd",
            tables[:, :depth].to(device=states.device, dtype=torch.float32),
            local_whitener,
        )
    for stage in range(depth):
        raw_table = tables[:, stage].to(device=states.device, dtype=torch.float32)
        selector_table = (
            raw_table
            if selector_tables is None
            else selector_tables[:, stage]
        )
        indices = nearest_indices(
            raw_residual if selector_residual is None else selector_residual,
            selector_table,
            assignment_whiteners=None,
            # The legacy full-head GSRQ path evaluates the whole context in
            # one bmm.  Preserve that layout/reduction order for frozen K.
            token_chunk_size=(
                token_chunk_size
                if selector_residual is None
                else selector_residual.shape[1]
            ),
        )
        selected_raw = torch.gather(
            raw_table,
            1,
            indices[..., None].expand(-1, -1, dimension),
        )
        decoded.add_(selected_raw)
        raw_residual.sub_(selected_raw)
        if selector_residual is not None:
            selected_selector = torch.gather(
                selector_table,
                1,
                indices[..., None].expand(-1, -1, dimension),
            )
            selector_residual.sub_(selected_selector)
    reconstruction = (
        decoded.reshape(heads, batch, tokens, dimension)
        .permute(1, 0, 2, 3)
        .to(states.dtype)
        .contiguous()
    )
    return raw_residual, selector_residual, reconstruction


def main() -> None:
    args = parse_args()
    started = time.time()
    if (
        args.uniform_depth <= 0
        or args.kmeans_iterations <= 0
        or args.extension_gain_cap_multiplier <= 0
        or args.extension_gain_cap_tail_stages <= 0
        or args.gradient_weight_target_median <= 0
    ):
        raise ValueError("depth, iterations, gain cap, and weight median must be positive")
    if args.gradient_weight_dir is not None and args.extension_fit_mode != "gskm":
        raise ValueError("CE gradient weighting is defined only for GSKM extension fitting")
    actions = parse_actions(args.action_depths, args.uniform_depth)
    maximum = max(actions)
    parent_dir = Path(args.parent_codebook_dir).resolve()
    capture_dir = Path(args.capture_dir).resolve()
    gradient_weight_dir = (
        None
        if args.gradient_weight_dir is None
        else Path(args.gradient_weight_dir).resolve()
    )
    if gradient_weight_dir is not None and not (
        gradient_weight_dir / "gradient_manifest.json"
    ).is_file():
        raise FileNotFoundError(
            gradient_weight_dir / "gradient_manifest.json"
        )
    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / "training_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(manifest_path)
    parent_files = sorted(parent_dir.glob("layer_*.pt"))
    if not parent_files:
        raise FileNotFoundError(f"no parent layers in {parent_dir}")
    available = [int(path.stem.split("_")[-1]) for path in parent_files]
    selected_layers = parse_layers(args.layers, available)
    model = os.path.realpath(args.model)
    codebook_files: dict[str, dict[str, Any]] = {}
    layer_diagnostics: dict[str, Any] = {}
    for layer in selected_layers:
        parent_path = parent_dir / f"layer_{layer:02d}.pt"
        capture_path = capture_dir / f"layer_{layer:02d}.pt"
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        capture = torch.load(capture_path, map_location="cpu", weights_only=False)
        provenance = parent.get("provenance", {})
        key = parent.get("codebooks", {}).get("key")
        value = parent.get("codebooks", {}).get("value")
        parent_whitener = parent.get("key_assignment_whiteners")
        euclidean_parent = bool(args.euclidean_key_assignment)
        if euclidean_parent:
            if provenance.get("key_transform") is not None:
                raise ValueError(
                    f"parent layer {layer} is not a raw-Euclidean GSRQ artifact"
                )
            whitener = None
            serialized_whitener = torch.eye(128, dtype=torch.float32).repeat(8, 1, 1)
        else:
            whitener = parent_whitener
            serialized_whitener = parent_whitener
        if (
            os.path.realpath(str(provenance.get("model", ""))) != model
            or provenance.get("surface") != "pre_rope_key_and_value_projection"
            or int(provenance.get("layer", -1)) != layer
            or not isinstance(key, torch.Tensor)
            or not isinstance(value, torch.Tensor)
            or key.shape != value.shape
            or tuple(key.shape[:1] + key.shape[2:]) != (8, 256, 128)
            or key.shape[1] < args.uniform_depth
            or (
                not euclidean_parent
                and (
                    not isinstance(whitener, torch.Tensor)
                    or tuple(whitener.shape) != (8, 128, 128)
                    or whitener.dtype != torch.float32
                )
            )
        ):
            raise ValueError(f"parent layer {layer} failed the frozen contract")
        if any(name not in capture for name in ("raw_key", "value")):
            raise ValueError(f"capture layer {layer} lacks KV tensors")
        capture_key, capture_value = capture["raw_key"], capture["value"]
        if (
            not isinstance(capture_key, torch.Tensor)
            or not isinstance(capture_value, torch.Tensor)
            or capture_key.shape != capture_value.shape
            or capture_key.ndim != 4
            or tuple(capture_key.shape[1::2]) != (8, 128)
            or (
                capture.get("model") is not None
                and os.path.realpath(str(capture["model"])) != model
            )
            or (
                capture.get("surface") is not None
                and capture["surface"] != "pre_rope_key_and_value_projection"
            )
            or (
                args.require_capture_sequence_length is not None
                and int(capture_key.shape[2])
                != args.require_capture_sequence_length
            )
        ):
            raise ValueError(f"capture layer {layer} failed geometry/provenance checks")

        fit_indices = parse_sample_indices(
            args.fit_sample_indices,
            samples=int(capture_key.shape[0]),
            name="fit sample indices",
        )
        validation_indices = parse_sample_indices(
            args.validation_sample_indices,
            samples=int(capture_key.shape[0]),
            name="validation sample indices",
            allow_empty=True,
        )
        if set(fit_indices) & set(validation_indices):
            raise ValueError("fit and validation capture rows overlap")
        if validation_indices and set(fit_indices) | set(validation_indices) != set(
            range(int(capture_key.shape[0]))
        ):
            raise ValueError("fit/validation rows must partition the capture")
        fit_index_tensor = torch.tensor(fit_indices, dtype=torch.long)

        key_weights = value_weights = None
        key_weight_diagnostics = value_weight_diagnostics = None
        if gradient_weight_dir is not None:
            key_weights, key_weight_diagnostics = load_ce_gradient_weights(
                gradient_weight_dir,
                layer=layer,
                kind="key",
                capture=capture,
                sample_indices=fit_indices,
                model=model,
                expected_heads=int(capture_key.shape[1]),
                target_median=args.gradient_weight_target_median,
                norm_scope=args.gradient_weight_norm_scope,
            )
            value_weights, value_weight_diagnostics = load_ce_gradient_weights(
                gradient_weight_dir,
                layer=layer,
                kind="value",
                capture=capture,
                sample_indices=fit_indices,
                model=model,
                expected_heads=int(capture_value.shape[1]),
                target_median=args.gradient_weight_target_median,
                norm_scope=args.gradient_weight_norm_scope,
            )

        parent_key = key[:, : args.uniform_depth].contiguous()
        parent_value = value[:, : args.uniform_depth].contiguous()
        raw_key = capture_key.index_select(0, fit_index_tensor).to(args.device)
        raw_value = capture_value.index_select(0, fit_index_tensor).to(args.device)
        local_whitener = (
            None
            if whitener is None
            else whitener.to(args.device, dtype=torch.float32)
        )
        key_residual, key_selector_residual, key_parent_recon = (
            parent_residuals_and_reconstruction(
                raw_key,
                parent_key,
                local_whitener,
                depth=args.uniform_depth,
                token_chunk_size=args.token_chunk_size,
            )
        )
        value_residual, _unused_value_selector, value_parent_recon = (
            parent_residuals_and_reconstruction(
                raw_value,
                parent_value,
                None,
                depth=args.uniform_depth,
                token_chunk_size=args.token_chunk_size,
            )
        )
        extra_stages = maximum - args.uniform_depth
        tail_start = max(
            0, args.uniform_depth - args.extension_gain_cap_tail_stages
        )
        key_parent_tail = parent_key[:, tail_start : args.uniform_depth].to(
            args.device, dtype=torch.float32
        )
        key_selector_tail = (
            key_parent_tail
            if local_whitener is None
            else torch.einsum(
                "hske,hde->hskd", key_parent_tail, local_whitener
            )
        )
        key_gain_caps = (
            torch.linalg.vector_norm(key_selector_tail, dim=-1)
            .amax(dim=(1, 2))
            .mul(args.extension_gain_cap_multiplier)
        )
        value_gain_caps = (
            torch.linalg.vector_norm(
                parent_value[:, tail_start : args.uniform_depth].to(
                    args.device, dtype=torch.float32
                ),
                dim=-1,
            )
            .amax(dim=(1, 2))
            .mul(args.extension_gain_cap_multiplier)
        )
        key_extra, key_history = fit_residual_stages(
            key_residual,
            stages=extra_stages,
            assignment_whiteners=local_whitener,
            initial_selector_residual=key_selector_residual,
            sample_weights=(
                None if key_weights is None else key_weights.to(args.device)
            ),
            kmeans_iterations=args.kmeans_iterations,
            token_chunk_size=args.token_chunk_size,
            seed=args.seed + layer * 1009,
            fit_mode=args.extension_fit_mode,
            gain_caps=(
                key_gain_caps if args.extension_fit_mode == "gskm" else None
            ),
            reserve_zero_codeword=args.extension_zero_codeword,
        )
        value_extra, value_history = fit_residual_stages(
            value_residual,
            stages=extra_stages,
            assignment_whiteners=None,
            initial_selector_residual=None,
            sample_weights=(
                None if value_weights is None else value_weights.to(args.device)
            ),
            kmeans_iterations=args.kmeans_iterations,
            token_chunk_size=args.token_chunk_size,
            seed=args.seed + layer * 1009 + 503,
            fit_mode=args.extension_fit_mode,
            gain_caps=(
                value_gain_caps if args.extension_fit_mode == "gskm" else None
            ),
            reserve_zero_codeword=args.extension_zero_codeword,
        )
        full_key = torch.cat((parent_key, key_extra), dim=1).contiguous()
        full_value = torch.cat((parent_value, value_extra), dim=1).contiguous()
        wrapped = {
            "key": direct_part(full_key),
            "value": direct_part(full_value),
        }
        tensor_bytes = (
            full_key.numel() * full_key.element_size()
            + full_value.numel() * full_value.element_size()
            + serialized_whitener.numel() * serialized_whitener.element_size()
        )
        target = output_dir / "codebooks" / f"layer_{layer:02d}.pt"
        artifact = {
            "format": "jointkv_factorized_nested_vq_v1",
            "stage_bits": [8] * maximum,
            "stage_cardinalities": [256] * maximum,
            "factor_ranks": [0] * maximum,
            "factorization_mode": "per_head_unfactorized",
            "action_depths": list(actions),
            "action_pair_payload_bits": {
                str(depth): depth * 16 for depth in actions
            },
            "value_encoder": {"mode": "greedy_residual"},
            "factorized_codebooks": wrapped,
            # SCFW preserves its learned FP32 transform.  Raw GSRQ stores an
            # exact FP32 identity solely because the shared evaluator expects
            # a tensor-valued transform; it introduces no unpublished metric.
            "key_assignment_whiteners": serialized_whitener.clone(),
            "provenance": {
                **provenance,
                "initialization": (
                    "rate-specific GSRQ-family parent is frozen through the uniform "
                    "exit; higher stages are direct per-head deployed-metric "
                    f"{args.extension_fit_mode} refinements fitted on that exact "
                    "parent's residual"
                ),
                "parent_codebook_path": str(parent_path),
                "parent_codebook_sha256": sha256_file(parent_path),
                "parent_uniform_depth": args.uniform_depth,
                "parent_prefix_frozen": True,
                "key_assignment_geometry": (
                    "raw_euclidean_gsrq"
                    if euclidean_parent
                    else "parent_fp32_whitener"
                ),
                "parent_whitener_dtype": (
                    None if whitener is None else str(whitener.dtype)
                ),
                "serialized_assignment_transform": (
                    "exact_fp32_identity"
                    if euclidean_parent
                    else "byte_exact_parent_fp32_whitener"
                ),
                "capture_path": str(capture_path),
                "capture_sha256": sha256_file(capture_path),
                "extension_stages": extra_stages,
                "extension_training_domain": (
                    str(capture.get("dataset", "train-only captures"))
                    + " train captures only; no validation/test data"
                ),
                "extension_sequence_length": int(capture_key.shape[2]),
                "extension_samples": int(capture_key.shape[0]),
                "extension_fit_samples": len(fit_indices),
                "extension_fit_sample_indices": list(fit_indices),
                "extension_validation_samples_untouched": len(validation_indices),
                "extension_validation_sample_indices": list(validation_indices),
                "extension_objective": (
                    f"deployed-metric residual {args.extension_fit_mode}; "
                    "no local-attention M-step; optional exact no-op codeword"
                ),
                "gradient_weighting": (
                    None
                    if gradient_weight_dir is None
                    else {
                        "directory": str(gradient_weight_dir),
                        "formula": (
                            "log1p(tau * final_CE_projection_gradient_norm "
                            "/ (fit-row median + epsilon))"
                        ),
                        "target_median": args.gradient_weight_target_median,
                        "norm_scope": args.gradient_weight_norm_scope,
                        "key": key_weight_diagnostics,
                        "value": value_weight_diagnostics,
                    }
                ),
            },
            "training": {
                "parent_updated": False,
                "key_extension_history": key_history,
                "value_extension_history": value_history,
            },
        }
        atomic_torch_save(artifact, target)
        # The uniform prefix must remain tensor-identical after wrapping.
        saved = torch.load(target, map_location="cpu", weights_only=False)
        for kind, expected in (("key", parent_key), ("value", parent_value)):
            actual = torch.stack(
                saved["factorized_codebooks"][kind]["shared_codebooks"][
                    : args.uniform_depth
                ],
                dim=1,
            )
            if not torch.equal(actual, expected):
                raise RuntimeError(f"layer {layer} {kind} parent prefix changed")
        if not torch.equal(
            saved["key_assignment_whiteners"], serialized_whitener
        ):
            raise RuntimeError(f"layer {layer} FP32 assignment transform changed")
        codebook_files[str(layer)] = {
            "path": str(target),
            "sha256": sha256_file(target),
            "factorized_static_bytes": tensor_bytes,
            "materialized_equivalent_static_bytes": tensor_bytes,
            "parent": str(parent_path),
            "parent_sha256": sha256_file(parent_path),
        }
        layer_diagnostics[str(layer)] = {
            "key_parent_residual_mse": float((raw_key - key_parent_recon).float().square().mean()),
            "value_parent_residual_mse": float((raw_value - value_parent_recon).float().square().mean()),
            "key_final_residual_mse": (
                key_history[-1]["residual_element_mse_after"]
                if key_history
                else float((raw_key - key_parent_recon).float().square().mean())
            ),
            "value_final_residual_mse": (
                value_history[-1]["residual_element_mse_after"]
                if value_history
                else float((raw_value - value_parent_recon).float().square().mean())
            ),
            "ce_gradient_weighted": gradient_weight_dir is not None,
        }
        print(
            json.dumps(
                {"event": "frozen_parent_layer_complete", "layer": layer},
                ensure_ascii=False,
            ),
            flush=True,
        )
        del (
            parent,
            capture,
            raw_key,
            raw_value,
            fit_index_tensor,
            key_parent_recon,
            value_parent_recon,
            key_residual,
            value_residual,
            key_extra,
            value_extra,
            key_weights,
            value_weights,
            full_key,
            full_value,
            artifact,
            saved,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = {
        "status": "complete",
        "format": "jointkv_factorized_nested_vq_training_manifest_v1",
        "scope": (
            "parent-preserving frozen-GSRQ precision-depth artifact"
            if args.euclidean_key_assignment
            else "parent-preserving frozen-SCFW-GSRQ precision-depth artifact"
        ),
        "model": model,
        "selected_layers": selected_layers,
        "stage_bits": [8] * maximum,
        "stage_cardinalities": [256] * maximum,
        "factor_ranks": [0] * maximum,
        "factorization_mode": "per_head_unfactorized",
        "action_depths": list(actions),
        "action_pair_payload_bits": {
            str(depth): depth * 16 for depth in actions
        },
        "effective_bits": args.effective_bits,
        "rate_profile": f"frozen_gsrq_parent_{args.effective_bits:g}bit",
        "allocation_scope": "test-time oracle allocator is evaluated separately",
        "inference_action_zero_enabled": False,
        "training_hyperparameters": {
            "uniform_anchor_depth": args.uniform_depth,
            "value_encoder": {"mode": "greedy_residual"},
            "parent_prefix_frozen": True,
            "kmeans_iterations": args.kmeans_iterations,
            "extension_fit_mode": args.extension_fit_mode,
            "extension_gain_cap_multiplier": args.extension_gain_cap_multiplier,
            "extension_gain_cap_tail_stages": args.extension_gain_cap_tail_stages,
            "extension_zero_codeword": args.extension_zero_codeword,
            "seed": args.seed,
            "euclidean_key_assignment": args.euclidean_key_assignment,
            "required_capture_sequence_length": args.require_capture_sequence_length,
            "fit_sample_indices": list(fit_indices),
            "validation_sample_indices_untouched": list(validation_indices),
            "gradient_weight_dir": (
                None if gradient_weight_dir is None else str(gradient_weight_dir)
            ),
            "gradient_weight_formula": (
                None
                if gradient_weight_dir is None
                else "log1p(tau * gradient_norm / (fit-row median + epsilon))"
            ),
            "gradient_weight_target_median": args.gradient_weight_target_median,
            "gradient_weight_norm_scope": args.gradient_weight_norm_scope,
        },
        "parent_codebook_dir": str(parent_dir),
        "capture_dir": str(capture_dir),
        "codebook_files": codebook_files,
        "model_static_bytes": {
            "factorized": sum(
                row["factorized_static_bytes"] for row in codebook_files.values()
            ),
            "materialized_equivalent": sum(
                row["materialized_equivalent_static_bytes"]
                for row in codebook_files.values()
            ),
        },
        "identity_contract": {
            "parent_codebooks_tensor_equal_through_uniform_exit": True,
            "key_assignment_geometry": (
                "raw_euclidean_gsrq"
                if args.euclidean_key_assignment
                else "parent_fp32_whitener"
            ),
            "key_assignment_transform_tensor_contract": (
                "exact_fp32_identity"
                if args.euclidean_key_assignment
                else "byte_exact_parent_fp32_whitener"
            ),
            "greedy_residual_encoder": True,
            "surface": "pre_rope_key_and_value_projection",
        },
        "layer_diagnostics": layer_diagnostics,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json_save(manifest, manifest_path)
    print(json.dumps({"status": "complete", "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
