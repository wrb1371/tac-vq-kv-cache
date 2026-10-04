#!/usr/bin/env python3
"""Line A0: selector-free, stage-17-free uniform-16 closed-loop training.

Only the first sixteen stages of the frozen-GSRQ 17-stage initializer are
imported.  The backbone is frozen, every forward uses the deployable hard U16
trajectory, and held-out U16 NLL is the sole checkpoint-selection criterion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from datasets import Dataset
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from jointkv.closed_loop_vq import (
    QuantizedKV,
    anchor_effect_residual_vq_exits,
    hard_residual_vq_exits,
)
from jointkv.table1_protocol import select_document_calibration_windows, tensor_sha256
from release_config import LLAMA3_BASE, C4_TRAIN_ARROW_0

try:
    from build_frozen_gsrq_multi_exit import atomic_json_save, atomic_torch_save, sha256_file
    from train_full_model_closed_loop_vq import (
        ClosedLoopRunContext,
        evaluate,
        install_closed_loop_attention,
        run_loss,
    )
except ModuleNotFoundError:
    from experiments.build_frozen_gsrq_multi_exit import (
        atomic_json_save,
        atomic_torch_save,
        sha256_file,
    )
    from experiments.train_full_model_closed_loop_vq import (
        ClosedLoopRunContext,
        evaluate,
        install_closed_loop_attention,
        run_loss,
    )


DEFAULT_MODEL = str(LLAMA3_BASE)
DEFAULT_C4_TRAIN = str(C4_TRAIN_ARROW_0)
UNIFORM_DEPTH = 16


def expected_checkpoint_validation_steps(
    *, train_samples: int, epochs: int, interval_steps: int, microsteps: int = 1
) -> list[int]:
    """Return the checkpoint opportunities produced by the training loop."""

    if train_samples <= 0 or epochs <= 0 or interval_steps < 0 or microsteps <= 0:
        raise ValueError("invalid checkpoint geometry")
    total = train_samples * epochs * microsteps
    if interval_steps == 0:
        return [train_samples * microsteps * (epoch + 1) for epoch in range(epochs)]
    steps = list(range(interval_steps, total + 1, interval_steps))
    if not steps or steps[-1] != total:
        steps.append(total)
    return steps


def assignment_soft_backward_epoch_configuration(
    epoch: int,
    *,
    epochs: int,
    topk: int,
    temperature_start: float,
    temperature_end: float,
    beta: float,
    hard_tail_epochs: int,
) -> dict[str, Any]:
    """Return the preregistered A2 soft-to-hard schedule for one epoch."""

    if not 0 <= epoch < epochs:
        raise ValueError("epoch lies outside the training schedule")
    soft_epochs = epochs - hard_tail_epochs
    if epoch >= soft_epochs or beta == 0:
        return {"topk": 1, "temperature": 0.0, "beta": 0.0, "enabled": False}
    fraction = 1.0 if soft_epochs == 1 else epoch / (soft_epochs - 1)
    # Geometric interpolation respects the scale of Gibbs temperature and
    # gives the documented exact endpoints (for example 1.0 -> 0.05).
    temperature = temperature_start * (
        temperature_end / temperature_start
    ) ** fraction
    return {
        "topk": int(topk),
        "temperature": float(temperature),
        "beta": float(beta),
        "enabled": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--c4-train-arrow", default=DEFAULT_C4_TRAIN)
    parser.add_argument("--input-artifact-dir", required=True)
    parser.add_argument("--comparison-artifact-dir", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument("--validation-samples", type=int, default=4)
    parser.add_argument("--train-seed", type=int, default=113)
    parser.add_argument("--validation-seed", type=int, default=211)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--checkpoint-validation-interval-steps",
        type=int,
        default=0,
        help=(
            "validate/checkpoint every N optimizer steps; 0 preserves the "
            "historical once-per-epoch behavior"
        ),
    )
    parser.add_argument(
        "--train-excluded-document-row",
        type=int,
        action="append",
        default=[],
        help="skip a train candidate row after consuming its deterministic RNG draws",
    )
    parser.add_argument(
        "--train-excluded-token-sha256",
        action="append",
        default=[],
        help="skip a train candidate token hash after consuming its deterministic RNG draws",
    )
    parser.add_argument(
        "--validation-excluded-document-row",
        type=int,
        action="append",
        default=[],
        help="skip a validation candidate row after consuming its deterministic RNG draws",
    )
    parser.add_argument(
        "--validation-excluded-token-sha256",
        action="append",
        default=[],
        help="skip a validation candidate token hash after consuming its deterministic RNG draws",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--assignment-microsteps",
        type=int,
        default=1,
        help=(
            "repeat each sampled train sequence for this many optimizer updates; "
            "A-R3 uses the same value in matched control and candidate arms"
        ),
    )
    parser.add_argument(
        "--frozen-assignment-mstep",
        action="store_true",
        help=(
            "A-R3: make the repeated updates a frozen-index M-step after one "
            "literal hard E-step, then exact-hard accept/reject the whole cycle"
        ),
    )
    parser.add_argument(
        "--frozen-assignment-exact-train-nll-tolerance",
        type=float,
        default=0.0,
        help=(
            "A-R3 allowed post-refresh increase in exact current-batch hard NLL "
            "before restoring every codebook table in the M-step cycle"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--codebook-optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--gradient-clip-mode",
        choices=("global", "per_layer_kv", "key_global_value_per_layer"),
        default="global",
        help=(
            "historical global cap, A-R2a independent (layer,K/V) caps, or "
            "one shared K cap plus independent per-layer V caps"
        ),
    )
    parser.add_argument(
        "--trainable-sides",
        choices=("both", "value_only"),
        default="both",
        help=(
            "train both K/V codebooks or freeze every K table and train only V; "
            "the hard K forward remains the imported GSRQ U16 codec"
        ),
    )
    parser.add_argument(
        "--trainable-stages",
        default="all",
        help=(
            "Stage-selective TAC: comma-separated U16 stage indices to update; "
            "all preserves the historical optimizer contract."
        ),
    )
    parser.add_argument(
        "--stage-learning-rate-decay",
        type=float,
        default=1.0,
        help="Stage-selective TAC multiplier: gradient at stage s is scaled by decay**s.",
    )
    parser.add_argument(
        "--key-update-geometry",
        choices=("raw", "whitened"),
        default="raw",
        help=(
            "A-GC: update K codebooks in the assignment-whitened coordinate "
            "system, implemented as the exact raw-space W^-1 gradient"
        ),
    )
    parser.add_argument(
        "--relative-update-cap",
        type=float,
        default=0.0,
        help=(
            "A-R2b: maximum per-(layer,K/V) SGD table move as a fraction of "
            "the table Frobenius norm; 0 disables the trust region"
        ),
    )
    parser.add_argument(
        "--sentinel-samples",
        type=int,
        default=0,
        help="A-R2c: independent no-grad hard-U16 sentinel rows; 0 disables it",
    )
    parser.add_argument("--sentinel-seed", type=int, default=0)
    parser.add_argument(
        "--sentinel-excluded-document-row",
        type=int,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--sentinel-excluded-token-sha256",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--sentinel-flip-rate-max",
        type=float,
        default=0.0,
        help="A-R2c maximum literal hard-index flip fraction per layer; 0 disables backtracking",
    )
    parser.add_argument(
        "--sentinel-mean-nll-tolerance",
        type=float,
        default=0.0,
        help="A-R2c allowed exact sentinel mean-NLL increase before full rollback",
    )
    parser.add_argument(
        "--sentinel-relative-margin-drop-max",
        type=float,
        default=0.0,
        help=(
            "A-R2c maximum per-side fractional drop in fixed-raw mean top-2 "
            "squared-distance margin before that layer is rolled back"
        ),
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=("heldout_best", "fixed_final"),
        default="heldout_best",
        help="state exported after training; validation checkpoints are always recorded",
    )
    parser.add_argument("--token-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--trajectory-match-weight",
        type=float,
        default=0.0,
        help="mean per-layer same-trajectory raw-vs-U16 attention loss weight",
    )
    parser.add_argument("--trajectory-match-epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--assignment-soft-topk",
        type=int,
        default=1,
        help="A2 backward-only shortlist size; 1 keeps the historical hard baseline",
    )
    parser.add_argument(
        "--assignment-soft-temperature-start", type=float, default=0.0
    )
    parser.add_argument(
        "--assignment-soft-temperature-end", type=float, default=0.0
    )
    parser.add_argument(
        "--assignment-soft-beta",
        type=float,
        default=0.0,
        help="A2 soft-assignment surrogate strength; forward remains literal hard NN",
    )
    parser.add_argument(
        "--assignment-soft-hard-tail-epochs",
        type=int,
        default=0,
        help="final epochs using the historical pure-hard backward",
    )
    parser.add_argument(
        "--anchor-effect",
        action="store_true",
        help="AE-0: freeze GSRQ anchor/search tables and train full effect/decode tables",
    )
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--smoke-layers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_cuda_device_text(device: str) -> str:
    """Accept an explicit physical CUDA device without changing run semantics."""

    if not device.startswith("cuda:"):
        raise ValueError("Line-A run requires an explicit cuda:N device")
    suffix = device.removeprefix("cuda:")
    if not suffix.isdigit():
        raise ValueError("Line-A CUDA device must use the cuda:N form")
    return device


def stack_direct_tables(part: dict[str, Any]) -> torch.Tensor:
    shared = part.get("shared_codebooks")
    entries = part.get("entry_factors")
    projections = part.get("dimension_factors")
    if not isinstance(shared, list) or not shared:
        raise ValueError("pure-U16 initializer lacks direct codebooks")
    if not isinstance(entries, list) or not isinstance(projections, list):
        raise ValueError("pure-U16 initializer lacks factor lists")
    if not (len(shared) == len(entries) == len(projections)):
        raise ValueError("initializer stage lists differ")
    if any(entry.shape[-1] or projection.shape[-2] for entry, projection in zip(entries, projections)):
        raise ValueError("pure-U16 baseline requires direct per-head tables")
    return torch.stack(shared, dim=1).contiguous()


def direct_part(tables: torch.Tensor) -> dict[str, Any]:
    heads, stages, cardinality, dimension = map(int, tables.shape)
    if stages != UNIFORM_DEPTH:
        raise ValueError("pure-U16 export must contain exactly sixteen stages")
    local = tables.detach().half().cpu()
    return {
        "shared_codebooks": [local[:, stage].contiguous() for stage in range(stages)],
        "entry_factors": [
            torch.empty((heads, cardinality, 0), dtype=torch.float16)
            for _ in range(stages)
        ],
        "dimension_factors": [
            torch.empty((heads, 0, dimension), dtype=torch.float16)
            for _ in range(stages)
        ],
        "residual_mse_by_stage": [],
        "representation": "per_head",
    }


def _update_hash(digest: Any, label: str, tensor: torch.Tensor) -> None:
    local = tensor.detach().contiguous().cpu()
    digest.update(label.encode())
    digest.update(str(tuple(local.shape)).encode())
    digest.update(str(local.dtype).encode())
    digest.update(local.numpy().tobytes())


def logical_codebook_sha256(
    key_tables: Sequence[torch.Tensor], value_tables: Sequence[torch.Tensor]
) -> str:
    digest = hashlib.sha256()
    for layer, (key, value) in enumerate(zip(key_tables, value_tables)):
        _update_hash(digest, f"layer{layer:02d}/key", key.detach().half())
        _update_hash(digest, f"layer{layer:02d}/value", value.detach().half())
    return digest.hexdigest()


def frozen_assignment_u16_reconstruction(
    states: torch.Tensor,
    tables: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode literal U16 indices while retaining the normal identity STE.

    A-R3's M-step intentionally freezes the E-step indices, but must still
    expose the selected FP16-roundtripped codewords to the CE gradient and
    preserve the identity path to the incoming transformer trajectory.  This
    function has no nearest-neighbour operation: it is only valid after an
    exact hard E-step has supplied all sixteen indices.
    """

    if states.ndim != 4 or tables.ndim != 4 or indices.ndim != 4:
        raise ValueError("frozen assignment geometry must be [B,H,T,D]/[H,S,K,D]/[B,H,T,S]")
    batch, heads, tokens, dimension = map(int, states.shape)
    if tuple(tables.shape[:2]) != (heads, UNIFORM_DEPTH) or int(tables.shape[-1]) != dimension:
        raise ValueError("frozen assignment table geometry differs")
    if tuple(indices.shape) != (batch, heads, tokens, UNIFORM_DEPTH):
        raise ValueError("frozen assignment indices differ from live raw geometry")
    device_indices = indices.to(device=states.device, dtype=torch.long)
    cardinality = int(tables.shape[2])
    if int(device_indices.min()) < 0 or int(device_indices.max()) >= cardinality:
        raise ValueError("frozen assignment index lies outside codebook")
    deployed_tables = tables + (tables.half().float() - tables).detach()
    flat = (
        device_indices.permute(1, 0, 2, 3)
        .reshape(heads, batch * tokens, UNIFORM_DEPTH)
        .contiguous()
    )
    decoded = torch.zeros(
        (heads, batch * tokens, dimension), device=states.device, dtype=torch.float32
    )
    for stage in range(UNIFORM_DEPTH):
        selected = torch.gather(
            deployed_tables[:, stage].float(),
            1,
            flat[..., stage, None].expand(-1, -1, dimension),
        )
        decoded = decoded + selected
    reconstructed = (
        decoded.reshape(heads, batch, tokens, dimension)
        .permute(1, 0, 2, 3)
        .contiguous()
        .to(states.dtype)
    )
    return reconstructed + (states - states.detach()), device_indices.to(torch.uint8)


class PureUniform16Codec(nn.Module):
    """A codec with no selector and no parameter or forward path for stage 17."""

    def __init__(
        self,
        key_tables: Sequence[torch.Tensor],
        value_tables: Sequence[torch.Tensor],
        key_whiteners: Sequence[torch.Tensor | None],
        *,
        head_dim: int = 128,
        token_chunk_size: int = 1024,
        soft_backward_topk: int = 1,
        soft_backward_temperature: float = 0.0,
        soft_backward_beta: float = 0.0,
    ) -> None:
        super().__init__()
        if not key_tables or not (
            len(key_tables) == len(value_tables) == len(key_whiteners)
        ):
            raise ValueError("pure-U16 layer payloads differ")
        geometry = tuple(key_tables[0].shape)
        if len(geometry) != 4 or geometry[1] != UNIFORM_DEPTH:
            raise ValueError("pure-U16 tables must have exactly sixteen stages")
        if geometry[-1] != head_dim:
            raise ValueError("codebook dimension differs from head_dim")
        if any(tuple(table.shape) != geometry for table in (*key_tables, *value_tables)):
            raise ValueError("pure-U16 layer geometries differ")
        self.key_tables = nn.ParameterList(
            [nn.Parameter(table.detach().float().contiguous()) for table in key_tables]
        )
        self.value_tables = nn.ParameterList(
            [nn.Parameter(table.detach().float().contiguous()) for table in value_tables]
        )
        for layer, whitener in enumerate(key_whiteners):
            self.register_buffer(
                f"key_whitener_{layer:02d}",
                torch.empty(0, dtype=torch.float32)
                if whitener is None
                else whitener.detach().float().contiguous(),
                persistent=True,
            )
        self.head_dim = int(head_dim)
        self.token_chunk_size = int(token_chunk_size)
        self._assignment_capture: dict[int, dict[str, torch.Tensor]] | None = None
        self._assignment_capture_raw = True
        self._local_reconstruction_capture: (
            dict[int, dict[str, torch.Tensor]] | None
        ) = None
        self._frozen_assignments: dict[int, dict[str, torch.Tensor]] | None = None
        self.configure_soft_backward(
            topk=soft_backward_topk,
            temperature=soft_backward_temperature,
            beta=soft_backward_beta,
        )

    def configure_soft_backward(
        self, *, topk: int, temperature: float, beta: float
    ) -> None:
        """Set an ephemeral training-only A2 backward configuration."""

        cardinality = int(self.key_tables[0].shape[2])
        if topk <= 0 or topk > cardinality:
            raise ValueError("soft-backward top-k lies outside the codebook")
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("soft-backward beta must be finite and non-negative")
        if beta > 0 and topk <= 1:
            raise ValueError("positive soft-backward beta requires top-k > 1")
        if beta > 0 and (not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("positive soft-backward beta requires positive temperature")
        self.soft_backward_topk = int(topk)
        self.soft_backward_temperature = float(temperature)
        self.soft_backward_beta = float(beta)

    def soft_backward_configuration(self) -> dict[str, Any]:
        return {
            "topk": self.soft_backward_topk,
            "temperature": self.soft_backward_temperature,
            "beta": self.soft_backward_beta,
            "enabled": self.soft_backward_beta > 0,
        }

    def whitener(self, layer: int) -> torch.Tensor | None:
        value = getattr(self, f"key_whitener_{layer:02d}")
        return None if value.numel() == 0 else value

    def begin_assignment_capture(self, *, include_raw_surfaces: bool = True) -> None:
        """Capture literal hard indices for one no-grad uniform forward.

        The capture is deliberately unavailable during a differentiable
        training forward: A-R2c uses it only on its separate sentinel stream.
        Stored indices are compact CPU int16 tensors (all 16-stage entries are
        in `[0,255]`) and do not alter the forward path.  A-R3's E-step only
        needs indices and therefore disables the much larger raw-surface
        capture used by A-R2c's fixed-raw boundary diagnostic.
        """

        if self._assignment_capture is not None:
            raise RuntimeError("assignment capture is already active")
        self._assignment_capture = {}
        self._assignment_capture_raw = bool(include_raw_surfaces)

    def end_assignment_capture(self) -> dict[int, dict[str, torch.Tensor]]:
        if self._assignment_capture is None:
            raise RuntimeError("assignment capture is not active")
        result = self._assignment_capture
        self._assignment_capture = None
        self._assignment_capture_raw = True
        if set(result) != set(range(len(self.key_tables))):
            raise RuntimeError("assignment capture did not observe every codec layer")
        return result

    def begin_local_reconstruction_gradient_capture(self) -> None:
        """Capture the literal fixed-assignment local V reconstruction gradient.

        This diagnostic/control path is intentionally outside autograd.  For
        every layer it evaluates, on the *same* hard index stream used by the
        deployed forward,

            sum_i 0.5 * ||residual_i - c[index_i]||^2

        stage by stage and stores ``dR/dc`` plus exact assignment occupancy on
        CPU.  It cannot change the hard forward value or its task-adjoint
        graph.  M1 uses the captured tensor as an alternative optimizer
        direction after separately measuring the task-gradient norm.
        """

        if self._local_reconstruction_capture is not None:
            raise RuntimeError("local reconstruction capture is already active")
        self._local_reconstruction_capture = {}

    def end_local_reconstruction_gradient_capture(
        self,
    ) -> dict[int, dict[str, torch.Tensor]]:
        if self._local_reconstruction_capture is None:
            raise RuntimeError("local reconstruction capture is not active")
        result = self._local_reconstruction_capture
        self._local_reconstruction_capture = None
        if set(result) != set(range(len(self.value_tables))):
            raise RuntimeError("local reconstruction capture missed codec layers")
        return result

    def _capture_local_reconstruction_gradient(
        self,
        layer: int,
        raw_value: torch.Tensor,
        value_indices: torch.Tensor,
    ) -> None:
        capture = self._local_reconstruction_capture
        if capture is None:
            return
        if layer in capture:
            raise RuntimeError("local reconstruction capture observed a layer twice")
        if raw_value.ndim != 4 or value_indices.ndim != 4:
            raise ValueError("local reconstruction capture geometry is invalid")
        batch, heads, tokens, dimension = map(int, raw_value.shape)
        stages = int(self.value_tables[layer].shape[1])
        cardinality = int(self.value_tables[layer].shape[2])
        if tuple(value_indices.shape) != (batch, heads, tokens, stages):
            raise ValueError("local reconstruction assignments differ from raw V")
        with torch.no_grad():
            # Match hard_residual_vq_exits exactly: both assignment search and
            # deployed reconstruction use the FP16-roundtripped table, while
            # the quantization-aware master has identity gradient.
            table = self.value_tables[layer].detach().half().float()
            residual = (
                raw_value.detach()
                .float()
                .permute(1, 0, 2, 3)
                .reshape(heads, batch * tokens, dimension)
                .contiguous()
                .clone()
            )
            gradient = torch.zeros_like(table)
            occupancy = torch.zeros(
                (heads, stages, cardinality),
                dtype=torch.int32,
                device=table.device,
            )
            head_index = torch.arange(heads, device=table.device).reshape(heads, 1)
            for stage in range(stages):
                index = (
                    value_indices[..., stage]
                    .long()
                    .permute(1, 0, 2)
                    .reshape(heads, batch * tokens)
                )
                chosen = table[head_index, stage, index]
                error = chosen - residual
                gradient[:, stage].scatter_add_(
                    1, index[..., None].expand(-1, -1, dimension), error
                )
                occupancy[:, stage].scatter_add_(
                    1, index, torch.ones_like(index, dtype=torch.int32)
                )
                residual.sub_(chosen)
            capture[layer] = {
                "gradient": gradient.cpu().contiguous(),
                "occupancy": occupancy.cpu().contiguous(),
            }

    def begin_frozen_assignment_mstep(
        self, assignments: Mapping[int, Mapping[str, torch.Tensor]]
    ) -> None:
        """Freeze a complete exact E-step index stream for one A-R3 M-step."""

        if self._assignment_capture is not None or self._frozen_assignments is not None:
            raise RuntimeError("assignment capture/freeze phase is already active")
        if set(assignments) != set(range(len(self.key_tables))):
            raise ValueError("frozen assignments do not cover every codec layer")
        frozen: dict[int, dict[str, torch.Tensor]] = {}
        for layer in range(len(self.key_tables)):
            frozen[layer] = {}
            for kind in ("key", "value"):
                value = assignments[layer].get(f"{kind}_indices")
                if not isinstance(value, torch.Tensor) or value.dtype not in (torch.uint8, torch.int16, torch.int32, torch.int64):
                    raise ValueError("frozen assignment indices must be integer tensors")
                frozen[layer][kind] = value.detach().to(device="cpu", dtype=torch.int16).contiguous()
        self._frozen_assignments = frozen

    def end_frozen_assignment_mstep(self) -> None:
        if self._frozen_assignments is None:
            raise RuntimeError("frozen assignment M-step is not active")
        self._frozen_assignments = None

    def forward(
        self,
        layer: int,
        raw_key: torch.Tensor,
        raw_value: torch.Tensor,
        *,
        mode: str = "uniform",
    ) -> QuantizedKV:
        if mode != "uniform":
            raise ValueError("pure-U16 codec exposes only uniform mode")
        if self._frozen_assignments is not None:
            if self._assignment_capture is not None:
                raise RuntimeError("cannot capture and freeze assignments in one forward")
            frozen = self._frozen_assignments.get(layer)
            if frozen is None:
                raise RuntimeError("frozen assignment M-step omitted a codec layer")
            key, key_indices = frozen_assignment_u16_reconstruction(
                raw_key, self.key_tables[layer], frozen["key"]
            )
            value, value_indices = frozen_assignment_u16_reconstruction(
                raw_value, self.value_tables[layer], frozen["value"]
            )
            self._capture_local_reconstruction_gradient(
                layer, raw_value, value_indices
            )
            return QuantizedKV(
                key=key,
                value=value,
                key_indices=key_indices,
                value_indices=value_indices,
                projected_depths=None,
            )
        key_exits, key_indices = hard_residual_vq_exits(
            raw_key,
            self.key_tables[layer],
            prefixes=(UNIFORM_DEPTH,),
            assignment_whiteners=self.whitener(layer),
            token_chunk_size=self.token_chunk_size,
            soft_backward_topk=self.soft_backward_topk,
            soft_backward_temperature=self.soft_backward_temperature,
            soft_backward_beta=self.soft_backward_beta,
        )
        value_exits, value_indices = hard_residual_vq_exits(
            raw_value,
            self.value_tables[layer],
            prefixes=(UNIFORM_DEPTH,),
            assignment_whiteners=None,
            token_chunk_size=self.token_chunk_size,
            soft_backward_topk=self.soft_backward_topk,
            soft_backward_temperature=self.soft_backward_temperature,
            soft_backward_beta=self.soft_backward_beta,
        )
        self._capture_local_reconstruction_gradient(layer, raw_value, value_indices)
        if self._assignment_capture is not None:
            if layer in self._assignment_capture:
                raise RuntimeError("assignment capture observed a layer twice")
            capture = {
                "key_indices": key_indices.detach()
                .to(device="cpu", dtype=torch.int16)
                .contiguous(),
                "value_indices": value_indices.detach()
                .to(device="cpu", dtype=torch.int16)
                .contiguous(),
            }
            if self._assignment_capture_raw:
                capture.update(
                    {
                        "raw_key": raw_key.detach().to(device="cpu", dtype=torch.float16).contiguous(),
                        "raw_value": raw_value.detach().to(device="cpu", dtype=torch.float16).contiguous(),
                    }
                )
            self._assignment_capture[layer] = capture
        return QuantizedKV(
            key=key_exits[UNIFORM_DEPTH],
            value=value_exits[UNIFORM_DEPTH],
            key_indices=key_indices,
            value_indices=value_indices,
            projected_depths=None,
        )


class AnchorEffectUniform16Codec(nn.Module):
    """AE-0 codec: frozen GSRQ partitions and trainable full decode effects."""

    def __init__(
        self,
        key_tables: Sequence[torch.Tensor],
        value_tables: Sequence[torch.Tensor],
        key_whiteners: Sequence[torch.Tensor | None],
        *,
        head_dim: int = 128,
        token_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        if not key_tables or not (
            len(key_tables) == len(value_tables) == len(key_whiteners)
        ):
            raise ValueError("AE-0 layer payloads differ")
        geometry = tuple(key_tables[0].shape)
        if len(geometry) != 4 or geometry[1] != UNIFORM_DEPTH:
            raise ValueError("AE-0 tables must have exactly sixteen stages")
        if geometry[-1] != head_dim:
            raise ValueError("codebook dimension differs from head_dim")
        if any(tuple(table.shape) != geometry for table in (*key_tables, *value_tables)):
            raise ValueError("AE-0 layer geometries differ")
        for layer, (key, value, whitener) in enumerate(
            zip(key_tables, value_tables, key_whiteners)
        ):
            # Both branches start from the exact persistent FP16 GSRQ table.
            # Anchors are buffers, not frozen Parameters, so they cannot enter
            # an optimizer accidentally.
            self.register_buffer(
                f"anchor_key_{layer:02d}",
                key.detach().half().contiguous(),
                persistent=True,
            )
            self.register_buffer(
                f"anchor_value_{layer:02d}",
                value.detach().half().contiguous(),
                persistent=True,
            )
            self.register_buffer(
                f"key_whitener_{layer:02d}",
                torch.empty(0, dtype=torch.float32)
                if whitener is None
                else whitener.detach().float().contiguous(),
                persistent=True,
            )
        self.effect_key_tables = nn.ParameterList(
            [nn.Parameter(table.detach().half().float().contiguous()) for table in key_tables]
        )
        self.effect_value_tables = nn.ParameterList(
            [
                nn.Parameter(table.detach().half().float().contiguous())
                for table in value_tables
            ]
        )
        self.head_dim = int(head_dim)
        self.token_chunk_size = int(token_chunk_size)

    @property
    def key_tables(self) -> nn.ParameterList:
        """Trainable decode tables; compatibility with training diagnostics."""

        return self.effect_key_tables

    @property
    def value_tables(self) -> nn.ParameterList:
        return self.effect_value_tables

    def anchor_key(self, layer: int) -> torch.Tensor:
        return getattr(self, f"anchor_key_{layer:02d}")

    def anchor_value(self, layer: int) -> torch.Tensor:
        return getattr(self, f"anchor_value_{layer:02d}")

    def whitener(self, layer: int) -> torch.Tensor | None:
        value = getattr(self, f"key_whitener_{layer:02d}")
        return None if value.numel() == 0 else value

    def configure_soft_backward(
        self, *, topk: int, temperature: float, beta: float
    ) -> None:
        if topk != 1 or temperature != 0 or beta != 0:
            raise ValueError("AE-0 excludes the failed A2 assignment surrogate")

    def forward(
        self,
        layer: int,
        raw_key: torch.Tensor,
        raw_value: torch.Tensor,
        *,
        mode: str = "uniform",
    ) -> QuantizedKV:
        if mode != "uniform":
            raise ValueError("AE-0 exposes only uniform mode")
        key_exits, key_indices = anchor_effect_residual_vq_exits(
            raw_key,
            self.anchor_key(layer),
            self.effect_key_tables[layer],
            prefixes=(UNIFORM_DEPTH,),
            assignment_whiteners=self.whitener(layer),
            token_chunk_size=self.token_chunk_size,
        )
        value_exits, value_indices = anchor_effect_residual_vq_exits(
            raw_value,
            self.anchor_value(layer),
            self.effect_value_tables[layer],
            prefixes=(UNIFORM_DEPTH,),
            assignment_whiteners=None,
            token_chunk_size=self.token_chunk_size,
        )
        return QuantizedKV(
            key=key_exits[UNIFORM_DEPTH],
            value=value_exits[UNIFORM_DEPTH],
            key_indices=key_indices,
            value_indices=value_indices,
            projected_depths=None,
        )


def anchor_logical_codebook_sha256(codec: AnchorEffectUniform16Codec) -> str:
    return logical_codebook_sha256(
        [codec.anchor_key(layer) for layer in range(len(codec.key_tables))],
        [codec.anchor_value(layer) for layer in range(len(codec.value_tables))],
    )


def audit_pure_uniform_contract(codec: PureUniform16Codec) -> dict[str, Any]:
    names = [name for name, _ in codec.named_parameters()]
    selector_names = [name for name in names if "selector" in name]
    stages = {int(table.shape[1]) for table in (*codec.key_tables, *codec.value_tables)}
    if stages != {UNIFORM_DEPTH} or selector_names or hasattr(codec, "selector"):
        raise RuntimeError("pure-U16 structural contract failed")
    codebooks_only = all(
        name.startswith(("key_tables.", "value_tables.")) for name in names
    )
    if not codebooks_only:
        raise RuntimeError("pure-U16 has a non-codebook trainable parameter")
    return {
        "uniform_depth": UNIFORM_DEPTH,
        "trainable_parameter_names_are_codebooks_only": codebooks_only,
        "selector_parameter_names": selector_names,
        "selector_parameter_count": 0,
        "stage17_parameter_count": 0,
        "stage17_forward_calls_by_construction": 0,
        "mixed_forward_available": False,
        "number_of_codebook_parameters": sum(p.numel() for p in codec.parameters()),
    }


def audit_anchor_effect_contract(
    codec: AnchorEffectUniform16Codec,
) -> dict[str, Any]:
    parameter_names = [name for name, _ in codec.named_parameters()]
    trainable_names = [
        name for name, parameter in codec.named_parameters() if parameter.requires_grad
    ]
    buffer_names = [name for name, _ in codec.named_buffers()]
    expected_trainable = all(
        name.startswith(("effect_key_tables.", "effect_value_tables."))
        for name in trainable_names
    )
    anchor_parameter_names = [name for name in parameter_names if "anchor" in name]
    anchor_buffer_names = [name for name in buffer_names if name.startswith("anchor_")]
    expected_anchor_buffers = 2 * len(codec.key_tables)
    tied_start_bitwise = all(
        torch.equal(
            codec.key_tables[layer].detach().half(), codec.anchor_key(layer)
        )
        and torch.equal(
            codec.value_tables[layer].detach().half(), codec.anchor_value(layer)
        )
        for layer in range(len(codec.key_tables))
    )
    if (
        not expected_trainable
        or not trainable_names
        or anchor_parameter_names
        or len(anchor_buffer_names) != expected_anchor_buffers
        or not tied_start_bitwise
    ):
        raise RuntimeError("AE-0 trainable/frozen partition contract failed")
    return {
        "method": "AE-0 full effect tables",
        "uniform_depth": UNIFORM_DEPTH,
        "anchor_storage": "persistent FP16 buffers",
        "effect_storage": "FP32 masters with FP16-roundtrip forward",
        "trainable_parameter_names_are_effect_only": expected_trainable,
        "trainable_parameter_names": trainable_names,
        "anchor_parameter_names": anchor_parameter_names,
        "anchor_buffer_count": len(anchor_buffer_names),
        "anchor_logical_sha256": anchor_logical_codebook_sha256(codec),
        "effect_initially_bitwise_equal_to_anchor": tied_start_bitwise,
        "initial_forward_equivalence": (
            "standard hard U16 by identical FP16 tables and shared tested recursion"
        ),
        "current_layer_search_dependency": (
            "frozen anchor residual only; effect never enters index recursion"
        ),
        "selector_parameter_count": 0,
        "stage17_parameter_count": 0,
        "mixed_forward_available": False,
        "trainable_effect_parameter_count": sum(
            parameter.numel() for parameter in codec.parameters()
        ),
        "anchor_static_elements": sum(
            codec.anchor_key(layer).numel() + codec.anchor_value(layer).numel()
            for layer in range(len(codec.key_tables))
        ),
    }


def load_uniform16_initializer(
    artifact_dir: Path,
    *,
    model_path: Path,
    device: torch.device,
    anchor_effect: bool = False,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    PureUniform16Codec | AnchorEffectUniform16Codec,
    dict[str, Any],
]:
    manifest_path = artifact_dir / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("initializer manifest is incomplete")
    if os.path.realpath(manifest["model"]) != os.path.realpath(model_path):
        raise ValueError("initializer/model paths differ")
    bits = tuple(int(bit) for bit in manifest.get("stage_bits", ()))
    actions = tuple(int(depth) for depth in manifest.get("action_depths", ()))
    if len(bits) < UNIFORM_DEPTH or bits[:UNIFORM_DEPTH] != (8,) * UNIFORM_DEPTH:
        raise ValueError("initializer lacks sixteen 8-bit prefix stages")
    if UNIFORM_DEPTH not in actions:
        raise ValueError("initializer does not expose U16")
    layers = [int(layer) for layer in manifest["selected_layers"]]
    if layers != list(range(len(layers))):
        raise ValueError("initializer must cover a contiguous full model")

    payloads, keys, values, whiteners = [], [], [], []
    source_files: dict[str, Any] = {}
    stage17_digest = hashlib.sha256()
    for layer in layers:
        row = manifest["codebook_files"][str(layer)]
        path = Path(row["path"])
        actual_sha = sha256_file(path)
        if row.get("sha256") is not None and actual_sha != row["sha256"]:
            raise ValueError(f"initializer layer {layer} failed SHA256")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("value_encoder", {}).get("mode") != "greedy_residual":
            raise ValueError(f"layer {layer} is not greedy residual VQ")
        full_key = stack_direct_tables(payload["factorized_codebooks"]["key"])
        full_value = stack_direct_tables(payload["factorized_codebooks"]["value"])
        if full_key.shape[1] < UNIFORM_DEPTH or full_value.shape[1] < UNIFORM_DEPTH:
            raise ValueError(f"layer {layer} lacks U16")
        if full_key.shape[1] > UNIFORM_DEPTH:
            _update_hash(stage17_digest, f"layer{layer:02d}/key17", full_key[:, 16])
            _update_hash(stage17_digest, f"layer{layer:02d}/value17", full_value[:, 16])
        payloads.append(payload)
        keys.append(full_key[:, :UNIFORM_DEPTH].contiguous())
        values.append(full_value[:, :UNIFORM_DEPTH].contiguous())
        geometry = payload.get("provenance", {}).get("key_assignment_geometry")
        whiteners.append(
            None if geometry == "raw_euclidean_gsrq" else payload["key_assignment_whiteners"]
        )
        source_files[str(layer)] = {"path": str(path.resolve()), "sha256": actual_sha}
    imported_hash = logical_codebook_sha256(keys, values)
    codec_class = AnchorEffectUniform16Codec if anchor_effect else PureUniform16Codec
    codec = codec_class(keys, values, whiteners).to(device)
    lineage = {
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_actions": list(actions),
        "source_stage_count": len(bits),
        "imported_prefix_depth": UNIFORM_DEPTH,
        "imported_first16_logical_sha256": imported_hash,
        "excluded_stage17_logical_sha256": (
            stage17_digest.hexdigest() if len(bits) >= 17 else None
        ),
        "stage17_excluded_before_codec_construction": True,
        "codec_construction": "anchor_effect_ae0" if anchor_effect else "standard_rq",
        "source_codebook_files": source_files,
    }
    return manifest, payloads, codec, lineage


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def gradient_diagnostics(
    codec: PureUniform16Codec | AnchorEffectUniform16Codec,
) -> dict[str, Any]:
    per_layer = []
    per_stage_squared = [0.0] * UNIFORM_DEPTH
    non_null = 0
    for key, value in zip(codec.key_tables, codec.value_tables):
        squared = 0.0
        for parameter in (key, value):
            if parameter.grad is None:
                continue
            non_null += 1
            local = parameter.grad.detach().float()
            squared += float(local.square().sum())
            for stage in range(UNIFORM_DEPTH):
                per_stage_squared[stage] += float(local[:, stage].square().sum())
        per_layer.append(math.sqrt(squared))
    return {
        "non_null_codebook_gradient_tensors": non_null,
        "per_layer_norms": per_layer,
        "per_stage_norms": [math.sqrt(value) for value in per_stage_squared],
        "selector_gradient_tensors": 0,
        "stage17_gradient_tensors": 0,
    }


@torch.no_grad()
def precondition_key_codebook_gradients(
    codec: PureUniform16Codec,
    *,
    mode: str,
    allow_absent_frozen: bool = False,
) -> dict[str, Any]:
    """Apply the exact assignment-metric natural coordinate change to K.

    The deployed search uses ``u = L c`` and ``W = L^T L``.  Plain SGD in
    ``u`` therefore corresponds to ``delta c = -eta W^-1 grad_c``.  This
    routine transforms only the stored raw-space gradient; the hard forward,
    persistent FP16 table, indices, and decoder remain unchanged.
    """

    if mode not in {"raw", "whitened"}:
        raise ValueError("unknown key update geometry")
    rows = []
    for layer, parameter in enumerate(codec.key_tables):
        gradient = parameter.grad
        whitener = codec.whitener(layer)
        if gradient is None:
            if allow_absent_frozen and mode == "raw" and not parameter.requires_grad:
                rows.append(
                    {
                        "layer": layer,
                        "whitener_present": whitener is not None,
                        "gradient_present": False,
                        "parameter_frozen": True,
                        "norm_before": None,
                        "norm_after": None,
                        "condition_number_max": None,
                        "identity": True,
                    }
                )
                continue
            raise RuntimeError(f"layer {layer} K gradient is absent")
        before = float(torch.linalg.vector_norm(gradient.detach().float()))
        if mode == "raw":
            rows.append(
                {
                    "layer": layer,
                    "whitener_present": whitener is not None,
                    "gradient_present": True,
                    "parameter_frozen": False,
                    "norm_before": before,
                    "norm_after": before,
                    "condition_number_max": None,
                    "identity": True,
                }
            )
            continue
        if whitener is None:
            raise ValueError("whitened K update requires every layer whitener")
        local = whitener.detach().float()
        metric = torch.matmul(local.transpose(-1, -2), local)
        flat = gradient.detach().float().reshape(
            gradient.shape[0], -1, gradient.shape[-1]
        )
        transformed = torch.linalg.solve(metric, flat.transpose(-1, -2)).transpose(-1, -2)
        if not bool(torch.isfinite(transformed).all()):
            raise RuntimeError("whitened K gradient preconditioner produced non-finite values")
        gradient.copy_(transformed.reshape_as(gradient).to(dtype=gradient.dtype))
        rows.append(
            {
                "layer": layer,
                "whitener_present": True,
                "gradient_present": True,
                "parameter_frozen": False,
                "norm_before": before,
                "norm_after": float(torch.linalg.vector_norm(gradient.detach().float())),
                "condition_number_max": float(torch.linalg.cond(metric).max()),
                "identity": False,
            }
        )
    return {
        "mode": mode,
        "formula": "raw K gradient <- (L^T L)^-1 raw K gradient",
        "layers": rows,
        "forward_changed": False,
        "value_gradient_changed": False,
    }


def clip_codebook_gradients(
    codec: PureUniform16Codec | AnchorEffectUniform16Codec,
    *,
    maximum_norm: float,
    mode: str,
) -> dict[str, Any]:
    """Clip codebook gradients without ever amplifying a small group.

    ``global`` is the historical behavior.  ``per_layer_kv`` is the isolated
    A-R2a intervention: every key table and value table is one independent
    group, giving exactly 64 groups for a 32-layer model.
    """

    if not math.isfinite(maximum_norm) or maximum_norm <= 0:
        raise ValueError("gradient cap must be finite and positive")
    groups = [
        (layer, kind, parameter)
        for kind, tables in (("key", codec.key_tables), ("value", codec.value_tables))
        for layer, parameter in enumerate(tables)
    ]
    if mode == "global":
        parameters = [
            parameter
            for _layer, _kind, parameter in groups
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("gradient clipping received no trainable codebooks")
        total_before = float(torch.nn.utils.clip_grad_norm_(parameters, maximum_norm))
        scale = min(1.0, maximum_norm / (total_before + 1e-12))
        return {
            "mode": mode,
            "maximum_norm": float(maximum_norm),
            "group_count": len(groups),
            "trainable_group_count": len(parameters),
            "total_norm_before_clip": total_before,
            "total_norm_after_clip": min(total_before, maximum_norm),
            "global_scale": scale,
            "per_group": None,
        }
    if mode not in {"per_layer_kv", "key_global_value_per_layer"}:
        raise ValueError(f"unknown gradient clip mode: {mode}")

    # A-P1 V2: K remains one historical competition group while every V table
    # receives its own cap.  Record per-table before/after values even for the
    # shared K group so downstream energy audits retain layer resolution.
    if mode == "key_global_value_per_layer":
        key_rows = [row for row in groups if row[1] == "key"]
        value_rows = [row for row in groups if row[1] == "value"]
        key_parameters = [row[2] for row in key_rows if row[2].requires_grad]
        key_before_rows = {
            layer: (
                0.0
                if parameter.grad is None
                else float(torch.linalg.vector_norm(parameter.grad.detach().float()))
            )
            for layer, _kind, parameter in key_rows
        }
        key_total_before = math.sqrt(sum(value * value for value in key_before_rows.values()))
        if key_parameters:
            torch.nn.utils.clip_grad_norm_(key_parameters, maximum_norm)
        rows: list[dict[str, Any]] = []
        squared_before = 0.0
        squared_after = 0.0
        for layer, kind, parameter in key_rows:
            before = key_before_rows[layer]
            after = (
                0.0
                if parameter.grad is None
                else float(torch.linalg.vector_norm(parameter.grad.detach().float()))
            )
            squared_before += before * before
            squared_after += after * after
            rows.append(
                {
                    "layer": layer,
                    "kind": kind,
                    "norm_before_clip": before,
                    "norm_after_clip": after,
                    "scale": 1.0 if before == 0.0 else after / before,
                    "clip_activated": key_total_before > maximum_norm,
                    "shared_group": "all_key_tables",
                }
            )
        for layer, kind, parameter in value_rows:
            if parameter.grad is None:
                before = after = 0.0
            else:
                before = float(torch.linalg.vector_norm(parameter.grad.detach().float()))
                torch.nn.utils.clip_grad_norm_([parameter], maximum_norm)
                after = float(torch.linalg.vector_norm(parameter.grad.detach().float()))
            squared_before += before * before
            squared_after += after * after
            rows.append(
                {
                    "layer": layer,
                    "kind": kind,
                    "norm_before_clip": before,
                    "norm_after_clip": after,
                    "scale": 1.0 if before == 0.0 else after / before,
                    "clip_activated": before > maximum_norm,
                    "shared_group": None,
                }
            )
        return {
            "mode": mode,
            "maximum_norm": float(maximum_norm),
            "group_count": 1 + len(value_rows),
            "trainable_group_count": int(bool(key_parameters))
            + sum(parameter.requires_grad for _layer, _kind, parameter in value_rows),
            "total_norm_before_clip": math.sqrt(squared_before),
            "total_norm_after_clip": math.sqrt(squared_after),
            "global_scale": None,
            "key_global_norm_before_clip": key_total_before,
            "per_group": rows,
        }

    rows: list[dict[str, Any]] = []
    squared_before = 0.0
    squared_after = 0.0
    for layer, kind, parameter in groups:
        if parameter.grad is None:
            before = 0.0
            after = 0.0
            scale = 1.0
        else:
            before = float(torch.linalg.vector_norm(parameter.grad.detach().float()))
            torch.nn.utils.clip_grad_norm_([parameter], maximum_norm)
            after = float(torch.linalg.vector_norm(parameter.grad.detach().float()))
            scale = 1.0 if before == 0.0 else after / before
        squared_before += before * before
        squared_after += after * after
        rows.append(
            {
                "layer": layer,
                "kind": kind,
                "norm_before_clip": before,
                "norm_after_clip": after,
                "scale": scale,
                "clip_activated": before > maximum_norm,
            }
        )
    return {
        "mode": mode,
        "maximum_norm": float(maximum_norm),
        "group_count": len(groups),
        "trainable_group_count": sum(
            parameter.requires_grad for _layer, _kind, parameter in groups
        ),
        "total_norm_before_clip": math.sqrt(squared_before),
        "total_norm_after_clip": math.sqrt(squared_after),
        "global_scale": None,
        "per_group": rows,
    }


def cap_relative_codebook_updates(
    codec: PureUniform16Codec | AnchorEffectUniform16Codec,
    *,
    learning_rate: float,
    maximum_relative_update: float,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Apply an exact per-table SGD trust region by scaling post-clip gradients.

    The A-R2b intervention is deliberately limited to plain SGD with no weight
    decay.  Under that contract the proposed table delta is exactly
    ``-learning_rate * grad``; scaling the gradient before ``optimizer.step``
    therefore gives a literal, auditable bound on every `(layer,K/V)` table
    update.  As with clipping, a small table is never amplified.
    """

    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("trust-region learning rate must be finite and positive")
    if not math.isfinite(maximum_relative_update) or maximum_relative_update < 0:
        raise ValueError("relative update cap must be finite and non-negative")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("trust-region epsilon must be finite and positive")

    groups = [
        (layer, kind, parameter)
        for kind, tables in (("key", codec.key_tables), ("value", codec.value_tables))
        for layer, parameter in enumerate(tables)
    ]
    if maximum_relative_update == 0:
        return {
            "enabled": False,
            "maximum_relative_update": 0.0,
            "epsilon": float(epsilon),
            "group_count": len(groups),
            "capped_group_count": 0,
            "per_group": None,
        }

    rows: list[dict[str, Any]] = []
    capped = 0
    for layer, kind, parameter in groups:
        table_norm = float(torch.linalg.vector_norm(parameter.detach().float()))
        if parameter.grad is None:
            gradient_norm = 0.0
            proposed = 0.0
            allowed = maximum_relative_update * (table_norm + epsilon)
            scale = 1.0
        else:
            gradient_norm = float(torch.linalg.vector_norm(parameter.grad.detach().float()))
            proposed = learning_rate * gradient_norm
            allowed = maximum_relative_update * (table_norm + epsilon)
            scale = 1.0 if proposed <= allowed else allowed / proposed
            if scale < 1.0:
                parameter.grad.mul_(scale)
                capped += 1
        applied = proposed * scale
        rows.append(
            {
                "layer": layer,
                "kind": kind,
                "table_frobenius_norm_before_update": table_norm,
                "gradient_norm_after_clip": gradient_norm,
                "proposed_update_frobenius_norm": proposed,
                "allowed_update_frobenius_norm": allowed,
                "applied_update_frobenius_norm": applied,
                "relative_update_proposed": proposed / (table_norm + epsilon),
                "relative_update_applied": applied / (table_norm + epsilon),
                "scale": scale,
                "trust_region_activated": scale < 1.0,
            }
        )
    return {
        "enabled": True,
        "maximum_relative_update": float(maximum_relative_update),
        "epsilon": float(epsilon),
        "group_count": len(groups),
        "capped_group_count": capped,
        "per_group": rows,
    }


@torch.no_grad()
def exact_sentinel_state(
    model: nn.Module,
    context: ClosedLoopRunContext,
    codec: PureUniform16Codec,
    tokens: torch.Tensor,
) -> dict[str, Any]:
    """Evaluate the isolated A-R2c sentinel and capture literal hard indices."""

    if tokens.ndim != 2 or tokens.shape[0] <= 0:
        raise ValueError("sentinel tokens must be a nonempty [batch,time] tensor")
    total_nll = 0.0
    total_tokens = 0
    assignments: dict[int, dict[str, list[torch.Tensor]]] = {
        layer: {"key": [], "value": []} for layer in range(len(codec.key_tables))
    }
    raw_surfaces: dict[int, dict[str, list[torch.Tensor]]] = {
        layer: {"key": [], "value": []} for layer in range(len(codec.key_tables))
    }
    for row in range(tokens.shape[0]):
        codec.begin_assignment_capture()
        loss, rate_records = run_loss(
            model, context, tokens[row : row + 1], mode="uniform"
        )
        captured = codec.end_assignment_capture()
        if rate_records:
            raise RuntimeError("pure-U16 sentinel unexpectedly emitted allocator records")
        count = int(tokens.shape[1] - 1)
        total_nll += float(loss) * count
        total_tokens += count
        for layer, record in captured.items():
            assignments[layer]["key"].append(record["key_indices"])
            assignments[layer]["value"].append(record["value_indices"])
            raw_surfaces[layer]["key"].append(record["raw_key"])
            raw_surfaces[layer]["value"].append(record["raw_value"])
    return {
        "mean_nll": total_nll / total_tokens,
        "perplexity": math.exp(total_nll / total_tokens),
        "tokens": total_tokens,
        "assignments": {
            layer: {
                kind: torch.cat(parts, dim=0).contiguous()
                for kind, parts in side.items()
            }
            for layer, side in assignments.items()
        },
        "raw_surfaces": {
            layer: {
                kind: torch.cat(parts, dim=0).contiguous()
                for kind, parts in side.items()
            }
            for layer, side in raw_surfaces.items()
        },
    }


@torch.no_grad()
def sentinel_boundary_assignments(
    codec: PureUniform16Codec,
    raw_surfaces: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[int, dict[str, torch.Tensor]]:
    """Reassign fixed raw sentinel K/V against the current codebook tables.

    This is the A-R2c boundary statistic.  It intentionally does *not* use
    post-update closed-loop raw states; those would mix a valid upstream
    trajectory change into every later layer's discrete-boundary diagnostic.
    Exact closed-loop sentinel NLL remains a separate accept/reject check.
    """

    if set(raw_surfaces) != set(range(len(codec.key_tables))):
        raise ValueError("sentinel raw surfaces do not cover every layer")
    result: dict[int, dict[str, torch.Tensor]] = {}
    for layer in range(len(codec.key_tables)):
        row = raw_surfaces[layer]
        key_raw = row["key"].to(codec.key_tables[layer].device)
        value_raw = row["value"].to(codec.value_tables[layer].device)
        _key_exits, key_indices = hard_residual_vq_exits(
            key_raw,
            codec.key_tables[layer],
            prefixes=(UNIFORM_DEPTH,),
            assignment_whiteners=codec.whitener(layer),
            token_chunk_size=codec.token_chunk_size,
        )
        _value_exits, value_indices = hard_residual_vq_exits(
            value_raw,
            codec.value_tables[layer],
            prefixes=(UNIFORM_DEPTH,),
            token_chunk_size=codec.token_chunk_size,
        )
        result[layer] = {
            "key": key_indices.detach().to(device="cpu", dtype=torch.int16).contiguous(),
            "value": value_indices.detach().to(device="cpu", dtype=torch.int16).contiguous(),
        }
    return result


@torch.no_grad()
def sentinel_boundary_statistics(
    codec: PureUniform16Codec,
    raw_surfaces: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[int, dict[str, dict[str, float]]]:
    """Measure fixed-raw U16 boundary margin and residual energy per layer.

    Assignment churn alone does not distinguish a small codeword adjustment
    from a collapse of the local nearest-neighbour margin.  This diagnostic
    intentionally follows the deployed FP16-roundtripped hard recursion on
    the *same* captured raw K/V surfaces used for the flip statistic.  It is
    therefore a codebook-boundary measurement, not a post-update
    closed-loop-trajectory measurement.
    """

    if set(raw_surfaces) != set(range(len(codec.key_tables))):
        raise ValueError("sentinel raw surfaces do not cover every layer")

    def side_statistics(
        raw: torch.Tensor,
        tables: torch.Tensor,
        whitener: torch.Tensor | None,
    ) -> dict[str, float]:
        if raw.ndim != 4 or tables.ndim != 4:
            raise ValueError("sentinel statistic geometry is invalid")
        batch, heads, tokens, dimension = map(int, raw.shape)
        if tuple(tables.shape[:2]) != (heads, UNIFORM_DEPTH) or int(tables.shape[-1]) != dimension:
            raise ValueError("sentinel statistic table geometry is invalid")
        physical_tables = tables.detach().half().float()
        physical_residual = (
            raw.to(tables.device, dtype=torch.float32)
            .permute(1, 0, 2, 3)
            .reshape(heads, batch * tokens, dimension)
            .contiguous()
        )
        if whitener is None:
            search_tables = physical_tables
            search_residual = physical_residual.clone()
        else:
            transform = whitener.detach().float()
            search_tables = torch.einsum("hrke,hde->hrkd", physical_tables, transform)
            search_residual = (
                torch.einsum(
                    "hde,bhte->bhtd",
                    transform,
                    raw.to(tables.device, dtype=torch.float32),
                )
                .permute(1, 0, 2, 3)
                .reshape(heads, batch * tokens, dimension)
                .contiguous()
            )
        margin_sum, margin_count = 0.0, 0
        for stage in range(UNIFORM_DEPTH):
            table = search_tables[:, stage]
            table_norm = table.square().sum(dim=-1)
            selected_parts: list[torch.Tensor] = []
            for start in range(0, search_residual.shape[1], codec.token_chunk_size):
                residual = search_residual[:, start : start + codec.token_chunk_size]
                distance = (
                    residual.square().sum(dim=-1, keepdim=True)
                    + table_norm[:, None, :]
                    - 2.0 * torch.bmm(residual, table.transpose(-1, -2))
                )
                top2 = torch.topk(distance, k=2, dim=-1, largest=False, sorted=True).values
                margin_sum += float((top2[..., 1] - top2[..., 0]).sum())
                margin_count += int(top2[..., 0].numel())
                selected_parts.append(distance.argmin(dim=-1))
            selected = torch.cat(selected_parts, dim=1)
            gather_index = selected[..., None].expand(-1, -1, dimension)
            chosen_physical = torch.gather(physical_tables[:, stage], 1, gather_index)
            chosen_search = torch.gather(search_tables[:, stage], 1, gather_index)
            physical_residual = physical_residual - chosen_physical
            search_residual = search_residual - chosen_search
        return {
            "mean_top2_squared_distance_margin": margin_sum / margin_count,
            "mean_final_physical_residual_energy": float(physical_residual.square().mean()),
            "assignment_positions_per_stage": int(margin_count // UNIFORM_DEPTH),
        }

    result: dict[int, dict[str, dict[str, float]]] = {}
    for layer in range(len(codec.key_tables)):
        result[layer] = {
            "key": side_statistics(
                raw_surfaces[layer]["key"], codec.key_tables[layer], codec.whitener(layer)
            ),
            "value": side_statistics(
                raw_surfaces[layer]["value"], codec.value_tables[layer], None
            ),
        }
    return result


def sentinel_margin_drop_rates(
    before: Mapping[int, Mapping[str, Mapping[str, float]]],
    after: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> list[dict[str, Any]]:
    """Return per-layer fractional fixed-raw margin drops (positive is worse)."""

    if set(before) != set(after):
        raise ValueError("sentinel margin layer sets differ")
    rows: list[dict[str, Any]] = []
    for layer in sorted(before):
        sides: dict[str, Any] = {}
        for kind in ("key", "value"):
            baseline = float(before[layer][kind]["mean_top2_squared_distance_margin"])
            candidate = float(after[layer][kind]["mean_top2_squared_distance_margin"])
            if baseline < 0 or candidate < 0:
                raise ValueError("sentinel margins must be non-negative")
            drop = 0.0 if baseline == 0 else max(0.0, 1.0 - candidate / baseline)
            sides[kind] = {
                "before_mean_top2_squared_distance_margin": baseline,
                "after_mean_top2_squared_distance_margin": candidate,
                "relative_margin_drop": drop,
                "before_mean_final_physical_residual_energy": float(
                    before[layer][kind]["mean_final_physical_residual_energy"]
                ),
                "after_mean_final_physical_residual_energy": float(
                    after[layer][kind]["mean_final_physical_residual_energy"]
                ),
            }
        rows.append(
            {
                "layer": layer,
                "key": sides["key"],
                "value": sides["value"],
                "maximum_side_relative_margin_drop": max(
                    sides["key"]["relative_margin_drop"],
                    sides["value"]["relative_margin_drop"],
                ),
            }
        )
    return rows


def sentinel_assignment_flip_rates(
    before: Mapping[int, Mapping[str, torch.Tensor]],
    after: Mapping[int, Mapping[str, torch.Tensor]],
) -> list[dict[str, Any]]:
    """Return exact per-layer hard-index churn for the accepted-state check."""

    if set(before) != set(after):
        raise ValueError("sentinel assignment layer sets differ")
    rows = []
    for layer in sorted(before):
        sides: dict[str, Any] = {}
        for kind in ("key", "value"):
            lhs, rhs = before[layer][kind], after[layer][kind]
            if lhs.dtype != torch.int16 or rhs.dtype != torch.int16 or lhs.shape != rhs.shape:
                raise ValueError("sentinel hard-index tensors differ")
            flips = int((lhs != rhs).sum())
            total = int(lhs.numel())
            sides[kind] = {"flips": flips, "total": total, "flip_rate": flips / total}
        rows.append(
            {
                "layer": layer,
                "key": sides["key"],
                "value": sides["value"],
                "maximum_side_flip_rate": max(
                    sides["key"]["flip_rate"], sides["value"]["flip_rate"]
                ),
            }
        )
    return rows


def assert_sentinel_boundary_matches_live(
    live: Mapping[int, Mapping[str, torch.Tensor]],
    boundary: Mapping[int, Mapping[str, torch.Tensor]],
) -> None:
    if set(live) != set(boundary):
        raise RuntimeError("sentinel live and fixed-raw layer sets differ")
    for layer in live:
        for kind in ("key", "value"):
            if not torch.equal(live[layer][kind], boundary[layer][kind]):
                raise RuntimeError(
                    "fixed-raw sentinel assignment differs from the live hard forward"
                )


def snapshot_codebook_layers(
    codec: PureUniform16Codec,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Take an exact in-device table snapshot for a possible A-R2c rollback."""

    return [
        (key.detach().clone(), value.detach().clone())
        for key, value in zip(codec.key_tables, codec.value_tables)
    ]


@torch.no_grad()
def restore_codebook_layers(
    codec: PureUniform16Codec,
    snapshot: Sequence[tuple[torch.Tensor, torch.Tensor]],
    layers: Iterable[int],
) -> None:
    for layer in layers:
        codec.key_tables[layer].copy_(snapshot[layer][0])
        codec.value_tables[layer].copy_(snapshot[layer][1])


def parameter_change_diagnostics(
    source_payloads: Sequence[dict[str, Any]],
    codec: PureUniform16Codec | AnchorEffectUniform16Codec,
) -> dict[str, Any]:
    changed, total, maximum = 0, 0, 0.0
    per_stage_changed = [0] * UNIFORM_DEPTH
    per_stage_total = [0] * UNIFORM_DEPTH
    for layer, payload in enumerate(source_payloads):
        source_key = stack_direct_tables(payload["factorized_codebooks"]["key"])[
            :, :UNIFORM_DEPTH
        ].half()
        source_value = stack_direct_tables(payload["factorized_codebooks"]["value"])[
            :, :UNIFORM_DEPTH
        ].half()
        for source, trained in (
            (source_key, codec.key_tables[layer].detach().half().cpu()),
            (source_value, codec.value_tables[layer].detach().half().cpu()),
        ):
            delta = (trained.float() - source.float()).abs()
            maximum = max(maximum, float(delta.max()))
            different = trained != source
            changed += int(different.sum())
            total += different.numel()
            for stage in range(UNIFORM_DEPTH):
                per_stage_changed[stage] += int(different[:, stage].sum())
                per_stage_total[stage] += different[:, stage].numel()
    return {
        "fp16_changed_elements": changed,
        "fp16_total_elements": total,
        "fp16_changed_fraction": changed / total,
        "maximum_absolute_fp16_change": maximum,
        "per_stage_changed_fraction": [
            count / size for count, size in zip(per_stage_changed, per_stage_total)
        ],
        "stage17_changed_elements": 0,
    }


def calibration_hashes(
    source_manifest: dict[str, Any], source_payloads: Sequence[dict[str, Any]]
) -> set[str]:
    rows: Iterable[dict[str, Any]] = source_manifest.get("provenance", {}).get(
        "sample_manifest", ()
    )
    payload_rows = source_payloads[0].get("provenance", {}).get("sample_manifest", ())
    if payload_rows:
        rows = payload_rows
    return {str(row.get("token_sha256")) for row in rows if row.get("token_sha256")}


def calibration_document_rows(
    source_manifest: dict[str, Any], source_payloads: Sequence[dict[str, Any]]
) -> set[int]:
    rows: Iterable[dict[str, Any]] = source_manifest.get("provenance", {}).get(
        "sample_manifest", ()
    )
    payload_rows = source_payloads[0].get("provenance", {}).get("sample_manifest", ())
    if payload_rows:
        rows = payload_rows
    return {int(row["row"]) for row in rows if row.get("row") is not None}


def save_artifact(
    output_dir: Path,
    source_artifact_dir: Path,
    source_manifest: dict[str, Any],
    source_payloads: Sequence[dict[str, Any]],
    codec: PureUniform16Codec | AnchorEffectUniform16Codec,
    *,
    training: dict[str, Any],
    lineage: dict[str, Any],
) -> Path:
    anchor_effect = isinstance(codec, AnchorEffectUniform16Codec)
    codebook_files: dict[str, Any] = {}
    for layer, source in enumerate(source_payloads):
        payload = copy.deepcopy(source)
        if anchor_effect:
            # Remove the ambiguous standard-RQ field so a legacy evaluator
            # fails closed instead of silently using anchors as decode effects.
            payload.pop("factorized_codebooks", None)
            payload["anchor_effect_codebooks"] = {
                "format": "jointkv_anchor_effect_direct_v1",
                "key": {
                    "anchor": direct_part(codec.anchor_key(layer)),
                    "effect": direct_part(codec.key_tables[layer]),
                },
                "value": {
                    "anchor": direct_part(codec.anchor_value(layer)),
                    "effect": direct_part(codec.value_tables[layer]),
                },
                "encoder_search_role": "anchor",
                "decoder_gather_role": "effect",
                "shared_index_stream": True,
            }
        else:
            payload["factorized_codebooks"] = {
                "key": direct_part(codec.key_tables[layer]),
                "value": direct_part(codec.value_tables[layer]),
            }
        payload.setdefault("training", {})["line_a_pure_uniform16"] = training
        payload.setdefault("provenance", {})["line_a_pure_uniform16"] = {
            "objective": training["objective"],
            "imported_prefix_depth": UNIFORM_DEPTH,
            "stage17_excluded": True,
            "selector_excluded": True,
            "anchor_effect": anchor_effect,
            "source_layer_file": lineage["source_codebook_files"][str(layer)],
        }
        target = output_dir / "codebooks" / f"layer_{layer:02d}.pt"
        atomic_torch_save(payload, target)
        codebook_files[str(layer)] = {
            "path": str(target),
            "sha256": sha256_file(target),
            "source": lineage["source_codebook_files"][str(layer)]["path"],
            "source_sha256": lineage["source_codebook_files"][str(layer)]["sha256"],
        }
        if anchor_effect:
            codebook_files[str(layer)].update(
                {
                    "anchor_key_sha256": tensor_sha256(codec.anchor_key(layer)),
                    "anchor_value_sha256": tensor_sha256(codec.anchor_value(layer)),
                    "effect_key_sha256": tensor_sha256(
                        codec.key_tables[layer].detach().half()
                    ),
                    "effect_value_sha256": tensor_sha256(
                        codec.value_tables[layer].detach().half()
                    ),
                }
            )
    manifest = copy.deepcopy(source_manifest)
    for key in ("selector_file", "selector_sha256"):
        manifest.pop(key, None)
    manifest.update(
        {
            "status": "complete",
            "format": (
                "jointkv_line_a_anchor_effect_ae0_training_manifest_v1"
                if anchor_effect
                else "jointkv_line_a_pure_uniform16_training_manifest_v1"
            ),
            "scope": (
                "Line-A AE-0 frozen-anchor/full-effect uniform-16 codec"
                if anchor_effect
                else "Line-A pure uniform-16 full-model closed-loop codec"
            ),
            "stage_bits": [8] * UNIFORM_DEPTH,
            "stage_cardinalities": [256] * UNIFORM_DEPTH,
            "factor_ranks": [0] * UNIFORM_DEPTH,
            "action_depths": [UNIFORM_DEPTH],
            "action_pair_payload_bits": {str(UNIFORM_DEPTH): 256},
            "effective_bits": 1.0,
            "rate_profile": "pure_uniform16_1bpa",
            "allocation_scope": "uniform depth 16 only; no allocator or selector",
            "selector_static_bytes_fp32": 0,
            "anchor_effect_layout": (
                {
                    "enabled": True,
                    "encoder_search_tables": "anchor FP16",
                    "decoder_gather_tables": "effect FP16",
                    "shared_dynamic_index_payload": True,
                    "dynamic_rate_bpa": 1.0,
                    "anchor_static_bytes_fp16": sum(
                        codec.anchor_key(layer).numel()
                        + codec.anchor_value(layer).numel()
                        for layer in range(len(codec.key_tables))
                    )
                    * 2,
                    "effect_static_bytes_fp16": sum(
                        codec.key_tables[layer].numel()
                        + codec.value_tables[layer].numel()
                        for layer in range(len(codec.key_tables))
                    )
                    * 2,
                }
                if anchor_effect
                else None
            ),
            "codebook_files": codebook_files,
            "source_artifact_dir": str(source_artifact_dir),
            "lineage_audit": lineage,
            "training_hyperparameters": {
                **source_manifest.get("training_hyperparameters", {}),
                "parent_prefix_frozen": anchor_effect,
                "full_model_closed_loop_completed": True,
                **training,
            },
        }
    )
    manifest_path = output_dir / "training_manifest.json"
    atomic_json_save(manifest, manifest_path)
    return manifest_path


def main() -> None:
    args = parse_args()
    if args.trainable_stages.strip().lower() == "all":
        trainable_stages = tuple(range(UNIFORM_DEPTH))
    else:
        try:
            trainable_stages = tuple(sorted({int(item.strip()) for item in args.trainable_stages.split(",") if item.strip()}))
        except ValueError as error:
            raise ValueError("--trainable-stages must be 'all' or comma-separated integers") from error
        if not trainable_stages or any(stage < 0 or stage >= UNIFORM_DEPTH for stage in trainable_stages):
            raise ValueError(f"--trainable-stages must select stages in [0,{UNIFORM_DEPTH - 1}]")
    a2_soft_backward = args.assignment_soft_beta > 0
    sentinel_backtracking = args.sentinel_samples > 0
    frozen_assignment_mstep = bool(args.frozen_assignment_mstep)
    if (
        args.sequence_length <= 1
        or args.train_samples <= 0
        or args.validation_samples <= 0
        or args.epochs <= 0
        or args.checkpoint_validation_interval_steps < 0
        or args.learning_rate <= 0
        or args.gradient_clip <= 0
        or not math.isfinite(args.relative_update_cap)
        or args.relative_update_cap < 0
        or not 1 <= args.assignment_microsteps <= 4
        or not math.isfinite(args.frozen_assignment_exact_train_nll_tolerance)
        or args.frozen_assignment_exact_train_nll_tolerance < 0
        or args.sentinel_samples < 0
        or not math.isfinite(args.sentinel_flip_rate_max)
        or not 0 <= args.sentinel_flip_rate_max <= 1
        or not math.isfinite(args.sentinel_mean_nll_tolerance)
        or args.sentinel_mean_nll_tolerance < 0
        or not math.isfinite(args.sentinel_relative_margin_drop_max)
        or not 0 <= args.sentinel_relative_margin_drop_max < 1
        or args.token_chunk_size <= 0
        or args.trajectory_match_weight < 0
        or args.trajectory_match_epsilon <= 0
        or args.assignment_soft_topk <= 0
        or not math.isfinite(args.assignment_soft_beta)
        or args.assignment_soft_beta < 0
        or args.assignment_soft_hard_tail_epochs < 0
        or args.assignment_soft_hard_tail_epochs > args.epochs
        or not math.isfinite(args.stage_learning_rate_decay)
        or args.stage_learning_rate_decay <= 0
    ):
        raise ValueError("invalid pure-U16 hyperparameters")
    if a2_soft_backward:
        if args.trajectory_match_weight:
            raise ValueError("A2 soft backward must remain isolated from failed A1 TM")
        if args.assignment_soft_topk <= 1:
            raise ValueError("A2 soft backward requires top-k > 1")
        if not 0 < args.assignment_soft_beta <= 1:
            raise ValueError("A2 beta must lie in (0,1]")
        if (
            not math.isfinite(args.assignment_soft_temperature_start)
            or not math.isfinite(args.assignment_soft_temperature_end)
            or args.assignment_soft_temperature_start <= 0
            or args.assignment_soft_temperature_end <= 0
            or args.assignment_soft_temperature_end
            > args.assignment_soft_temperature_start
        ):
            raise ValueError("A2 temperature must be positive and non-increasing")
        if args.assignment_soft_hard_tail_epochs >= args.epochs:
            raise ValueError("A2 requires at least one soft-backward epoch")
    elif (
        args.assignment_soft_topk != 1
        or args.assignment_soft_temperature_start != 0
        or args.assignment_soft_temperature_end != 0
        or args.assignment_soft_hard_tail_epochs != 0
    ):
        raise ValueError("soft-backward schedule options require positive beta")
    if args.anchor_effect and (a2_soft_backward or args.trajectory_match_weight):
        raise ValueError("AE-0 excludes failed A1 TM and A2 soft backward")
    if args.key_update_geometry == "whitened" and (
        args.anchor_effect
        or a2_soft_backward
        or args.trajectory_match_weight
        or frozen_assignment_mstep
        or sentinel_backtracking
        or args.relative_update_cap
        or args.gradient_clip_mode != "per_layer_kv"
        or args.trainable_sides != "both"
        or args.codebook_optimizer != "sgd"
        or args.weight_decay != 0
    ):
        raise ValueError(
            "whitened K updates require isolated pure-hard per-layer/KV plain SGD"
        )
    if args.relative_update_cap:
        if args.gradient_clip_mode != "per_layer_kv":
            raise ValueError("relative update cap requires per_layer_kv clipping")
        if args.codebook_optimizer != "sgd" or args.weight_decay != 0:
            raise ValueError(
                "relative update cap requires exact plain SGD with zero weight decay"
            )
    if frozen_assignment_mstep:
        if (
            args.assignment_microsteps < 2
            or args.anchor_effect
            or a2_soft_backward
            or args.trajectory_match_weight
            or sentinel_backtracking
            or args.relative_update_cap != 0
            or args.gradient_clip_mode != "per_layer_kv"
            or args.trainable_sides != "both"
            or args.codebook_optimizer != "sgd"
            or args.weight_decay != 0
            or args.smoke_layers
        ):
            raise ValueError(
                "A-R3 frozen-assignment M-step requires 2-4 microsteps, pure hard "
                "U16, per_layer_kv plain SGD, no trust/sentinel/soft/TM/anchor/smoke"
            )
        if (
            args.checkpoint_validation_interval_steps
            and args.checkpoint_validation_interval_steps % args.assignment_microsteps
        ):
            raise ValueError("A-R3 checkpoint interval must end an M-step cycle")
    elif args.frozen_assignment_exact_train_nll_tolerance != 0:
        raise ValueError("frozen-assignment tolerance requires frozen-assignment M-step")
    if sentinel_backtracking:
        if (
            args.anchor_effect
            or a2_soft_backward
            or args.trajectory_match_weight
            or args.gradient_clip_mode != "per_layer_kv"
            or args.trainable_sides != "both"
            or args.relative_update_cap <= 0
            or args.sentinel_flip_rate_max <= 0
            or args.sentinel_relative_margin_drop_max <= 0
            or args.smoke_layers
        ):
            raise ValueError(
                "sentinel backtracking requires pure hard U16, per_layer_kv, and positive trust/flip/margin caps"
            )
    elif (
        args.sentinel_flip_rate_max != 0
        or args.sentinel_mean_nll_tolerance != 0
        or args.sentinel_relative_margin_drop_max != 0
        or args.sentinel_seed != 0
        or args.sentinel_excluded_document_row
        or args.sentinel_excluded_token_sha256
    ):
        raise ValueError("sentinel options require positive sentinel_samples")
    validate_cuda_device_text(args.device)

    random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    started = time.time()
    model_path = Path(args.model).resolve()
    arrow_path = Path(args.c4_train_arrow).resolve()
    input_dir = Path(args.input_artifact_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if (output_dir / "training_manifest.json").exists() and not args.overwrite:
        raise FileExistsError(output_dir / "training_manifest.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    # PyTorch 2.5.1 rejects reset_peak_memory_stats before lazy CUDA init.
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
    rows = Dataset.from_file(str(arrow_path))
    train_tokens, train_rows = select_document_calibration_windows(
        tokenizer,
        rows,
        samples=args.train_samples,
        sequence_length=args.sequence_length,
        seed=args.train_seed,
        excluded_document_rows=args.train_excluded_document_row,
        excluded_token_sha256=args.train_excluded_token_sha256,
    )
    validation_tokens, validation_rows = select_document_calibration_windows(
        tokenizer,
        rows,
        samples=args.validation_samples,
        sequence_length=args.sequence_length,
        seed=args.validation_seed,
        excluded_document_rows=args.validation_excluded_document_row,
        excluded_token_sha256=args.validation_excluded_token_sha256,
    )
    train_hashes = {str(row["token_sha256"]) for row in train_rows}
    validation_hashes = {str(row["token_sha256"]) for row in validation_rows}
    train_document_rows = {int(row["row"]) for row in train_rows}
    validation_document_rows = {int(row["row"]) for row in validation_rows}
    if train_hashes & validation_hashes or train_document_rows & validation_document_rows:
        raise ValueError("training and validation document rows or token hashes overlap")
    sentinel_tokens = None
    sentinel_rows: list[dict[str, Any]] = []
    sentinel_hashes: set[str] = set()
    sentinel_document_rows: set[int] = set()
    if sentinel_backtracking:
        sentinel_tokens, sentinel_rows = select_document_calibration_windows(
            tokenizer,
            rows,
            samples=args.sentinel_samples,
            sequence_length=args.sequence_length,
            seed=args.sentinel_seed,
            excluded_document_rows=args.sentinel_excluded_document_row,
            excluded_token_sha256=args.sentinel_excluded_token_sha256,
        )
        sentinel_hashes = {str(row["token_sha256"]) for row in sentinel_rows}
        sentinel_document_rows = {int(row["row"]) for row in sentinel_rows}
        if (
            sentinel_hashes & (train_hashes | validation_hashes)
            or sentinel_document_rows & (train_document_rows | validation_document_rows)
            or len(sentinel_hashes) != args.sentinel_samples
            or len(sentinel_document_rows) != args.sentinel_samples
        ):
            raise ValueError("sentinel overlaps train/validation or is not unique")

    source_manifest, source_payloads, codec, lineage = load_uniform16_initializer(
        input_dir,
        model_path=model_path,
        device=device,
        anchor_effect=args.anchor_effect,
    )
    initializer_hashes = calibration_hashes(source_manifest, source_payloads)
    initializer_document_rows = calibration_document_rows(source_manifest, source_payloads)
    overlap_gate = {
        "initializer_calibration_hash_count": len(initializer_hashes),
        "initializer_calibration_document_row_count": len(initializer_document_rows),
        "train_token_hash_overlap_count": len(initializer_hashes & train_hashes),
        "validation_token_hash_overlap_count": len(initializer_hashes & validation_hashes),
        "train_document_row_overlap_count": len(
            initializer_document_rows & train_document_rows
        ),
        "validation_document_row_overlap_count": len(
            initializer_document_rows & validation_document_rows
        ),
        "sentinel_token_hash_overlap_count": len(initializer_hashes & sentinel_hashes),
        "sentinel_document_row_overlap_count": len(
            initializer_document_rows & sentinel_document_rows
        ),
    }
    overlap_gate["passed"] = not any(
        value
        for key, value in overlap_gate.items()
        if key.endswith("_overlap_count")
    )
    if not overlap_gate["passed"]:
        raise ValueError("train or held-out validation overlaps initializer calibration")
    codec.token_chunk_size = args.token_chunk_size
    contract = (
        audit_anchor_effect_contract(codec)
        if isinstance(codec, AnchorEffectUniform16Codec)
        else audit_pure_uniform_contract(codec)
    )
    anchor_hash_before = (
        anchor_logical_codebook_sha256(codec)
        if isinstance(codec, AnchorEffectUniform16Codec)
        else None
    )
    if args.anchor_effect and anchor_hash_before != lineage["imported_first16_logical_sha256"]:
        raise RuntimeError("AE-0 anchor hash differs from imported GSRQ prefix")
    active_layers = len(codec.key_tables)
    if args.smoke_layers:
        if not 0 < args.smoke_layers <= active_layers:
            raise ValueError("smoke layer count is outside codec")
        active_layers = args.smoke_layers

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    backbone_audit = {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    if backbone_audit["trainable_parameter_count"]:
        raise RuntimeError("backbone is not frozen")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    context = ClosedLoopRunContext(codec, "uniform")
    context.trajectory_match_epsilon = args.trajectory_match_epsilon
    install_closed_loop_attention(model, context, active_layers=active_layers)
    if args.trainable_sides == "value_only":
        for parameter in codec.key_tables.parameters():
            parameter.requires_grad_(False)
    trainable_names = [
        name for name, parameter in codec.named_parameters() if parameter.requires_grad
    ]
    frozen_names = [
        name for name, parameter in codec.named_parameters() if not parameter.requires_grad
    ]
    if args.trainable_sides == "value_only" and (
        not trainable_names
        or not all(name.startswith("value_tables.") for name in trainable_names)
        or not all(name.startswith("key_tables.") for name in frozen_names)
    ):
        raise RuntimeError("value-only trainable/frozen codebook partition failed")
    contract["trainable_sides"] = args.trainable_sides
    contract["trainable_parameter_names"] = trainable_names
    contract["frozen_parameter_names"] = frozen_names
    contract["frozen_key_tables_while_hard_forward_remains_active"] = (
        args.trainable_sides == "value_only"
    )
    parameters = [parameter for parameter in codec.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("no trainable codec parameters remain")
    optimizer_type = torch.optim.SGD if args.codebook_optimizer == "sgd" else torch.optim.AdamW
    optimizer = optimizer_type(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    validation_device = validation_tokens.to(device)
    sentinel_device = None if sentinel_tokens is None else sentinel_tokens.to(device)
    validation_before = evaluate(
        model, context, validation_device, modes=("uniform",)
    )["uniform"]
    trajectory_validation_before = None
    if args.trajectory_match_weight:
        context.collect_trajectory_match = True
        layer_totals = [0.0] * active_layers
        row_totals = []
        with torch.no_grad():
            for row in range(validation_device.shape[0]):
                run_loss(
                    model,
                    context,
                    validation_device[row : row + 1],
                    mode="uniform",
                )
                local = [float(loss) for _layer, loss in context.trajectory_match_losses]
                if len(local) != active_layers:
                    raise RuntimeError("trajectory validation did not cover active layers")
                row_totals.append(sum(local) / active_layers)
                layer_totals = [left + right for left, right in zip(layer_totals, local)]
        trajectory_validation_before = {
            "mean_relative_attention_mse": sum(row_totals) / len(row_totals),
            "per_layer_relative_attention_mse": [
                value / len(row_totals) for value in layer_totals
            ],
        }
        context.collect_trajectory_match = False
    comparison_metrics: dict[str, Any] = {}
    for comparison_text in args.comparison_artifact_dir:
        comparison_dir = Path(comparison_text).resolve()
        _, comparison_payloads, comparison_codec, comparison_lineage = load_uniform16_initializer(
            comparison_dir, model_path=model_path, device=device
        )
        comparison_codec.token_chunk_size = args.token_chunk_size
        context.codec = comparison_codec
        local = evaluate(model, context, validation_device, modes=("uniform",))["uniform"]
        local["artifact_dir"] = str(comparison_dir)
        local["manifest_sha256"] = comparison_lineage["source_manifest_sha256"]
        local["uniform16_logical_sha256"] = comparison_lineage[
            "imported_first16_logical_sha256"
        ]
        comparison_metrics[comparison_dir.name] = local
        del comparison_codec, comparison_payloads
        torch.cuda.empty_cache()
    context.codec = codec
    sentinel_state = None
    sentinel_backtracking_history: list[dict[str, Any]] = []
    if sentinel_backtracking:
        if not isinstance(codec, PureUniform16Codec) or sentinel_device is None:
            raise RuntimeError("A-R2c sentinel requires the standard pure-U16 codec")
        sentinel_state = exact_sentinel_state(model, context, codec, sentinel_device)
        sentinel_state["boundary_assignments"] = sentinel_boundary_assignments(
            codec, sentinel_state["raw_surfaces"]
        )
        sentinel_state["boundary_statistics"] = sentinel_boundary_statistics(
            codec, sentinel_state["raw_surfaces"]
        )
        assert_sentinel_boundary_matches_live(
            sentinel_state["assignments"], sentinel_state["boundary_assignments"]
        )
        print(
            json.dumps(
                {
                    "event": "line_a_r2c_sentinel_before",
                    "mean_nll": sentinel_state["mean_nll"],
                    "perplexity": sentinel_state["perplexity"],
                    "tokens": sentinel_state["tokens"],
                }
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "event": "line_a_pure_u16_validation_before",
                "metrics": validation_before,
                "comparisons": comparison_metrics,
                "contract": contract,
            }
        ),
        flush=True,
    )

    history = []
    best_state = cpu_state_dict(codec)
    best_validation = float(validation_before["mean_nll"])
    best_epoch = -1
    best_step = 0
    step = 0
    checkpoint_history = []
    frozen_assignment_mstep_history: list[dict[str, Any]] = []
    active_frozen_assignment_cycle: dict[str, Any] | None = None

    def validate_checkpoint(epoch: int, source: str) -> None:
        nonlocal best_validation, best_epoch, best_step, best_state
        context.collect_trajectory_match = False
        # Heldout checkpoint selection always runs the literal historical hard
        # path without constructing any backward-only shortlist machinery.
        codec.configure_soft_backward(topk=1, temperature=0.0, beta=0.0)
        validation = evaluate(
            model, context, validation_device, modes=("uniform",)
        )["uniform"]
        accepted = validation["mean_nll"] < best_validation
        if accepted:
            best_validation = float(validation["mean_nll"])
            best_epoch = epoch
            best_step = step
            best_state = cpu_state_dict(codec)
        local_checkpoint = {
            "step": step,
            "epoch": epoch,
            "source": source,
            "accepted": accepted,
            "checkpoint_criterion": "heldout_uniform16_mean_nll_only",
            "metrics": validation,
        }
        checkpoint_history.append(local_checkpoint)
        print(
            json.dumps(
                {"event": "line_a_pure_u16_checkpoint_validation", **local_checkpoint}
            ),
            flush=True,
        )
    for epoch in range(args.epochs):
        assignment_configuration = assignment_soft_backward_epoch_configuration(
            epoch,
            epochs=args.epochs,
            topk=args.assignment_soft_topk,
            temperature_start=args.assignment_soft_temperature_start,
            temperature_end=args.assignment_soft_temperature_end,
            beta=args.assignment_soft_beta,
            hard_tail_epochs=args.assignment_soft_hard_tail_epochs,
        )
        codec.configure_soft_backward(
            topk=assignment_configuration["topk"],
            temperature=assignment_configuration["temperature"],
            beta=assignment_configuration["beta"],
        )
        base_order = list(range(args.train_samples))
        random.Random(args.train_seed + epoch).shuffle(base_order)
        order = [
            row
            for row in base_order
            for _microstep in range(args.assignment_microsteps)
        ]
        for row_position, row in enumerate(order):
            if args.max_steps and step >= args.max_steps:
                break
            microstep_index = row_position % args.assignment_microsteps
            train_batch = train_tokens[row : row + 1].to(device)
            if frozen_assignment_mstep and microstep_index == 0:
                if active_frozen_assignment_cycle is not None:
                    raise RuntimeError("A-R3 previous frozen M-step cycle was not closed")
                pre_cycle_snapshot = snapshot_codebook_layers(codec)
                codec.begin_assignment_capture(include_raw_surfaces=False)
                with torch.no_grad():
                    e_step_loss, e_step_rates = run_loss(
                        model, context, train_batch, mode="uniform"
                    )
                e_step_assignments = codec.end_assignment_capture()
                if e_step_rates:
                    raise RuntimeError("A-R3 E-step unexpectedly emitted allocator rate records")
                codec.begin_frozen_assignment_mstep(e_step_assignments)
                active_frozen_assignment_cycle = {
                    "epoch": epoch,
                    "sample": row,
                    "cycle": row_position // args.assignment_microsteps,
                    "before_exact_train_mean_nll": float(e_step_loss),
                    "snapshot": pre_cycle_snapshot,
                }
            optimizer.zero_grad(set_to_none=True)
            context.collect_trajectory_match = bool(args.trajectory_match_weight)
            loss, rate_records = run_loss(
                model, context, train_batch, mode="uniform"
            )
            # Non-reentrant checkpointing records the first forward graph, but
            # may execute attention again while backward materializes saved
            # tensors.  Freeze the exact first-forward side losses here so any
            # replay-only context appends cannot alter the objective or audit.
            trajectory_layer_losses = list(context.trajectory_match_losses)
            if (
                args.trajectory_match_weight
                and len(trajectory_layer_losses) != active_layers
            ):
                raise RuntimeError(
                    "trajectory training did not cover every active layer exactly once"
                )
            trajectory_loss = (
                torch.stack(
                    [local_loss for _layer, local_loss in trajectory_layer_losses]
                ).mean()
                if args.trajectory_match_weight
                else None
            )
            total_loss = loss + (
                args.trajectory_match_weight * trajectory_loss
                if trajectory_loss is not None
                else 0.0
            )
            total_loss.backward()
            # Stage-selective TAC keeps the full U16 forward and shared index
            # stream intact, while restricting the optimizer update to the
            # preregistered stage subset.  Masking happens before geometry and
            # clipping diagnostics, so the recorded gradient norms describe
            # exactly the optimized subspace.
            if len(trainable_stages) < UNIFORM_DEPTH:
                frozen_stage_mask = torch.ones(UNIFORM_DEPTH, dtype=torch.bool, device=codec.key_tables[0].device)
                frozen_stage_mask[list(trainable_stages)] = False
                for parameter in (*codec.key_tables, *codec.value_tables):
                    if parameter.grad is not None:
                        parameter.grad[:, frozen_stage_mask] = 0
            if args.stage_learning_rate_decay != 1.0:
                for parameter in (*codec.key_tables, *codec.value_tables):
                    if parameter.grad is not None:
                        scales = torch.as_tensor(
                            [args.stage_learning_rate_decay ** stage for stage in range(UNIFORM_DEPTH)],
                            dtype=parameter.grad.dtype,
                            device=parameter.grad.device,
                        )
                        parameter.grad.mul_(scales.view(1, -1, 1, 1))
            gradients = gradient_diagnostics(codec)
            key_update_geometry = precondition_key_codebook_gradients(
                codec,
                mode=args.key_update_geometry,
                allow_absent_frozen=args.trainable_sides == "value_only",
            )
            preconditioned_gradients = gradient_diagnostics(codec)
            clipping = clip_codebook_gradients(
                codec,
                maximum_norm=args.gradient_clip,
                mode=args.gradient_clip_mode,
            )
            trust_region = cap_relative_codebook_updates(
                codec,
                learning_rate=args.learning_rate,
                maximum_relative_update=args.relative_update_cap,
            )
            pre_update_snapshot = (
                snapshot_codebook_layers(codec) if sentinel_backtracking else None
            )
            optimizer.step()
            frozen_assignment_audit = None
            if frozen_assignment_mstep and microstep_index == args.assignment_microsteps - 1:
                if active_frozen_assignment_cycle is None:
                    raise RuntimeError("A-R3 final microstep lacks an E-step state")
                codec.end_frozen_assignment_mstep()
                with torch.no_grad():
                    post_refresh_loss, post_refresh_rates = run_loss(
                        model, context, train_batch, mode="uniform"
                    )
                if post_refresh_rates:
                    raise RuntimeError("A-R3 refresh unexpectedly emitted allocator rate records")
                accepted = float(post_refresh_loss) <= (
                    active_frozen_assignment_cycle["before_exact_train_mean_nll"]
                    + args.frozen_assignment_exact_train_nll_tolerance
                )
                if not accepted:
                    restore_codebook_layers(
                        codec,
                        active_frozen_assignment_cycle["snapshot"],
                        range(len(codec.key_tables)),
                    )
                frozen_assignment_audit = {
                    "enabled": True,
                    "cycle": active_frozen_assignment_cycle["cycle"],
                    "microsteps": args.assignment_microsteps,
                    "before_exact_train_mean_nll": active_frozen_assignment_cycle[
                        "before_exact_train_mean_nll"
                    ],
                    "post_refresh_exact_train_mean_nll": float(post_refresh_loss),
                    "exact_train_nll_tolerance": args.frozen_assignment_exact_train_nll_tolerance,
                    "accepted": accepted,
                    "global_codebook_rollback": not accepted,
                    "e_step_assignment_layer_count": len(e_step_assignments),
                }
                frozen_assignment_mstep_history.append(frozen_assignment_audit)
                active_frozen_assignment_cycle = None
            sentinel_audit = None
            if sentinel_backtracking:
                if sentinel_state is None or pre_update_snapshot is None:
                    raise RuntimeError("A-R2c sentinel state is absent")
                candidate_sentinel = exact_sentinel_state(
                    model, context, codec, sentinel_device
                )
                candidate_mean_nll = candidate_sentinel["mean_nll"]
                flips = sentinel_assignment_flip_rates(
                    sentinel_state["boundary_assignments"],
                    sentinel_boundary_assignments(codec, sentinel_state["raw_surfaces"]),
                )
                candidate_boundary_statistics = sentinel_boundary_statistics(
                    codec, sentinel_state["raw_surfaces"]
                )
                margin_drops = sentinel_margin_drop_rates(
                    sentinel_state["boundary_statistics"], candidate_boundary_statistics
                )
                layer_rollbacks = [
                    int(row["layer"])
                    for row, margin_row in zip(flips, margin_drops)
                    if (
                        row["maximum_side_flip_rate"] > args.sentinel_flip_rate_max
                        or margin_row["maximum_side_relative_margin_drop"]
                        > args.sentinel_relative_margin_drop_max
                    )
                ]
                if layer_rollbacks:
                    restore_codebook_layers(codec, pre_update_snapshot, layer_rollbacks)
                    candidate_sentinel = exact_sentinel_state(
                        model, context, codec, sentinel_device
                    )
                global_nll_rollback = (
                    candidate_sentinel["mean_nll"]
                    > sentinel_state["mean_nll"] + args.sentinel_mean_nll_tolerance
                )
                if global_nll_rollback:
                    restore_codebook_layers(
                        codec, pre_update_snapshot, range(len(codec.key_tables))
                    )
                    final_sentinel = sentinel_state
                else:
                    candidate_sentinel["boundary_assignments"] = sentinel_boundary_assignments(
                        codec, candidate_sentinel["raw_surfaces"]
                    )
                    candidate_sentinel["boundary_statistics"] = sentinel_boundary_statistics(
                        codec, candidate_sentinel["raw_surfaces"]
                    )
                    assert_sentinel_boundary_matches_live(
                        candidate_sentinel["assignments"],
                        candidate_sentinel["boundary_assignments"],
                    )
                    final_sentinel = candidate_sentinel
                sentinel_audit = {
                    "enabled": True,
                    "flip_rate_max": args.sentinel_flip_rate_max,
                    "relative_margin_drop_max": args.sentinel_relative_margin_drop_max,
                    "mean_nll_tolerance": args.sentinel_mean_nll_tolerance,
                    "before_mean_nll": sentinel_state["mean_nll"],
                    "candidate_mean_nll_before_layer_rollback": (
                        candidate_mean_nll
                    ),
                    "after_layer_rollback_mean_nll": candidate_sentinel["mean_nll"],
                    "final_mean_nll": final_sentinel["mean_nll"],
                    "layer_rollbacks": layer_rollbacks,
                    "global_nll_rollback": global_nll_rollback,
                    "per_layer_flip": flips,
                    "per_layer_margin": margin_drops,
                }
                sentinel_backtracking_history.append(sentinel_audit)
                sentinel_state = final_sentinel
            if rate_records:
                raise RuntimeError("pure-U16 unexpectedly emitted allocator rate records")
            if not all(torch.isfinite(parameter).all() for parameter in codec.parameters()):
                raise RuntimeError("pure-U16 update produced non-finite parameters")
            step += 1
            local = {
                "step": step,
                "epoch": epoch,
                "sample": row,
                "uniform16_loss": float(loss.detach()),
                "trajectory_match_loss": (
                    None if trajectory_loss is None else float(trajectory_loss.detach())
                ),
                "trajectory_match_weight": args.trajectory_match_weight,
                "assignment_soft_backward": assignment_configuration,
                "total_objective": float(total_loss.detach()),
                "per_layer_trajectory_match_loss": (
                    None
                    if trajectory_loss is None
                    else [
                        float(local_loss.detach())
                        for _layer, local_loss in trajectory_layer_losses
                    ]
                ),
                "total_codebook_gradient_norm": float(
                    clipping["total_norm_before_clip"]
                ),
                "gradient_diagnostics_before_clip": gradients,
                "key_update_geometry": key_update_geometry,
                "gradient_diagnostics_after_key_geometry_before_clip": preconditioned_gradients,
                "gradient_clipping": clipping,
                "relative_update_trust_region": trust_region,
                "sentinel_backtracking": sentinel_audit,
                "frozen_assignment_hard_em": frozen_assignment_audit,
                "selector_gradient_tensors": 0,
                "stage17_gradient_tensors": 0,
            }
            history.append(local)
            print(json.dumps({"event": "line_a_pure_u16_train_step", **local}), flush=True)
            if (
                args.checkpoint_validation_interval_steps
                and step % args.checkpoint_validation_interval_steps == 0
            ):
                validate_checkpoint(epoch, "fixed_step_interval")
                # Restore this epoch's training assignment configuration after
                # the literal-hard validation path.  This is a no-op for R1,
                # but preserves the generic trainer's A2 behavior.
                codec.configure_soft_backward(
                    topk=assignment_configuration["topk"],
                    temperature=assignment_configuration["temperature"],
                    beta=assignment_configuration["beta"],
                )
        if (
            not args.checkpoint_validation_interval_steps
            or not checkpoint_history
            or checkpoint_history[-1]["step"] != step
        ):
            validate_checkpoint(epoch, "epoch_end")
        if frozen_assignment_mstep and active_frozen_assignment_cycle is not None:
            raise RuntimeError("A-R3 epoch ended with a frozen assignment phase active")
        if args.max_steps and step >= args.max_steps:
            break

    if not args.max_steps:
        expected_validation_steps = expected_checkpoint_validation_steps(
            train_samples=args.train_samples,
            epochs=args.epochs,
            interval_steps=args.checkpoint_validation_interval_steps,
            microsteps=args.assignment_microsteps,
        )
        if [row["step"] for row in checkpoint_history] != expected_validation_steps:
            raise RuntimeError("checkpoint validation opportunities differ from geometry")
    codec.configure_soft_backward(topk=1, temperature=0.0, beta=0.0)
    context.collect_trajectory_match = False
    final_training_state = cpu_state_dict(codec)
    validation_fixed_final = evaluate(
        model, context, validation_device, modes=("uniform",)
    )["uniform"]
    codec.load_state_dict(best_state)
    validation_heldout_best = evaluate(
        model, context, validation_device, modes=("uniform",)
    )["uniform"]
    selected_step = best_step
    selected_epoch = best_epoch
    if args.checkpoint_selection == "fixed_final":
        codec.load_state_dict(final_training_state)
        validation_after = validation_fixed_final
        selected_step = step
        selected_epoch = args.epochs - 1
    else:
        validation_after = validation_heldout_best
    trajectory_validation_after = None
    if args.trajectory_match_weight:
        context.collect_trajectory_match = True
        layer_totals = [0.0] * active_layers
        row_totals = []
        with torch.no_grad():
            for row in range(validation_device.shape[0]):
                run_loss(
                    model,
                    context,
                    validation_device[row : row + 1],
                    mode="uniform",
                )
                local = [float(loss) for _layer, loss in context.trajectory_match_losses]
                row_totals.append(sum(local) / active_layers)
                layer_totals = [left + right for left, right in zip(layer_totals, local)]
        trajectory_validation_after = {
            "mean_relative_attention_mse": sum(row_totals) / len(row_totals),
            "per_layer_relative_attention_mse": [
                value / len(row_totals) for value in layer_totals
            ],
        }
        context.collect_trajectory_match = False
    final_hash = logical_codebook_sha256(codec.key_tables, codec.value_tables)
    anchor_hash_after = (
        anchor_logical_codebook_sha256(codec)
        if isinstance(codec, AnchorEffectUniform16Codec)
        else None
    )
    anchor_immutability_gate = (
        {
            "before_sha256": anchor_hash_before,
            "after_sha256": anchor_hash_after,
            "source_imported_first16_sha256": lineage[
                "imported_first16_logical_sha256"
            ],
            "passed": anchor_hash_before
            == anchor_hash_after
            == lineage["imported_first16_logical_sha256"],
        }
        if args.anchor_effect
        else None
    )
    if anchor_immutability_gate is not None and not anchor_immutability_gate["passed"]:
        raise RuntimeError("AE-0 frozen anchor changed during training")
    changes = parameter_change_diagnostics(source_payloads, codec)
    properties = torch.cuda.get_device_properties(device)
    runtime = {
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "elapsed_seconds": time.time() - started,
    }
    training = {
        "line": (
            "A-GC"
            if args.key_update_geometry == "whitened"
            else "AE-0"
            if args.anchor_effect
            else (
                "A-R3"
                if frozen_assignment_mstep
                else (
                "A2"
                if a2_soft_backward
                else (
                    "A1"
                    if args.trajectory_match_weight
                    else (
                        "A-R2c"
                        if sentinel_backtracking
                        else (
                            "A-R2b"
                            if args.relative_update_cap
                            else (
                                "A-P1"
                                if args.trainable_sides == "value_only"
                                or args.gradient_clip_mode == "key_global_value_per_layer"
                                else "A-R2a"
                                if args.gradient_clip_mode == "per_layer_kv"
                                else "A0"
                            )
                        )
                    )
                )
                )
            )
        ),
        "method": (
            "Pure-U16 + independent (layer,K/V) clipping + assignment-metric-consistent W^-1 K update"
            if args.key_update_geometry == "whitened"
            else "AE-0 frozen GSRQ anchor + full trainable effect tables"
            if args.anchor_effect
            else (
                "Pure-U16 + literal hard E-step/frozen-index M-step/exact hard refresh reject"
                if frozen_assignment_mstep
                else (
                    "Pure-U16 + hard-forward/top-k-soft-backward"
                    if a2_soft_backward
                    else (
                        "Pure-U16 + trajectory-matched attention"
                        if args.trajectory_match_weight
                        else (
                            (
                                "Pure-U16 + independent (layer,K/V) gradient clipping "
                                "+ relative per-table SGD trust region + exact sentinel "
                                "hard-index/loss backtracking"
                                if sentinel_backtracking
                                else (
                                    "Pure-U16 + independent (layer,K/V) gradient clipping "
                                    "+ relative per-table SGD trust region"
                                    if args.relative_update_cap
                                    else "Pure-U16 + independent (layer,K/V) gradient clipping"
                                )
                            )
                            if args.gradient_clip_mode == "per_layer_kv"
                            else (
                                "Pure-U16 + one global K cap + independent per-layer V caps"
                                if args.gradient_clip_mode == "key_global_value_per_layer"
                                else "Pure-U16"
                            )
                        )
                    )
                )
            )
        ),
        "objective": (
            "uniform16 full-model closed-loop next-token cross entropy only; "
            "fixed GSRQ anchor search and trainable full effect decode"
            if args.anchor_effect
            else (
                "uniform16 full-model closed-loop next-token cross entropy only; "
                "hard NN forward with backward-only top-k soft assignment"
                if a2_soft_backward
                else (
                    "uniform16 full-model closed-loop next-token cross entropy only; "
                    "one literal hard E-step plus fixed-index M microsteps and "
                    "post-cycle exact hard train-loss reject"
                    if frozen_assignment_mstep
                    else (
                    "uniform16 full-model closed-loop next-token cross entropy plus "
                    "mean per-layer same-trajectory raw-vs-U16 relative attention MSE"
                    if args.trajectory_match_weight
                    else "uniform16 full-model closed-loop next-token cross entropy only"
                    )
                )
            )
        ),
        "checkpoint_criterion": (
            "fixed_final_primary_with_heldout_dashboard_secondary"
            if args.checkpoint_selection == "fixed_final"
            else "heldout_uniform16_mean_nll_only"
        ),
        "discrete_forward": (
            "hard frozen-anchor nearest-neighbour search plus same-index effect "
            "gather, exactly 16 stages"
            if args.anchor_effect
            else "hard nearest-neighbour residual VQ, exactly 16 stages"
        ),
        "gradient_estimator": (
            "identity state STE; selected effect words only; anchor search detached"
            if args.anchor_effect
            else (
                "identity state STE plus hard-forward/top-k-soft-assignment backward"
                if a2_soft_backward
                else (
                    "identity state STE for frozen-index selected-codeword decode"
                    if frozen_assignment_mstep
                    else "identity STE for VQ state path"
                )
            )
        ),
        "selector_participation": "none",
        "mixed_depth_participation": "none",
        "stage17_participation": "excluded before codec construction",
        "model": str(model_path),
        "c4_train_arrow": str(arrow_path),
        "c4_train_arrow_sha256": sha256_file(arrow_path),
        "sequence_length": args.sequence_length,
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "train_seed": args.train_seed,
        "validation_seed": args.validation_seed,
        "train_manifest": train_rows,
        "validation_manifest": validation_rows,
        "train_token_sha256": tensor_sha256(train_tokens),
        "validation_token_sha256": tensor_sha256(validation_tokens),
        "sentinel_backtracking": {
            "enabled": sentinel_backtracking,
            "samples": args.sentinel_samples,
            "seed": args.sentinel_seed if sentinel_backtracking else None,
            "manifest": sentinel_rows if sentinel_backtracking else None,
            "token_sha256": (
                None if sentinel_tokens is None else tensor_sha256(sentinel_tokens)
            ),
            "candidate_exclusions": {
                "document_rows": sorted(set(args.sentinel_excluded_document_row)),
                "token_sha256": sorted(set(args.sentinel_excluded_token_sha256)),
                "application": "after eligible candidate consumes row/start RNG draws",
            },
            "flip_rate_max": args.sentinel_flip_rate_max,
            "relative_margin_drop_max": args.sentinel_relative_margin_drop_max,
            "mean_nll_tolerance": args.sentinel_mean_nll_tolerance,
            "history": sentinel_backtracking_history if sentinel_backtracking else [],
        },
        "train_candidate_exclusions": {
            "document_rows": sorted(set(args.train_excluded_document_row)),
            "token_sha256": sorted(set(args.train_excluded_token_sha256)),
            "application": "after eligible candidate consumes row/start RNG draws",
        },
        "validation_candidate_exclusions": {
            "document_rows": sorted(set(args.validation_excluded_document_row)),
            "token_sha256": sorted(set(args.validation_excluded_token_sha256)),
            "application": "after eligible candidate consumes row/start RNG draws",
        },
        "initializer_overlap_gate": overlap_gate,
        "epochs": args.epochs,
        "assignment_microsteps": args.assignment_microsteps,
        "completed_steps": step,
        "checkpoint_validation_interval_steps": args.checkpoint_validation_interval_steps,
        "checkpoint_validation_steps": [row["step"] for row in checkpoint_history],
        "checkpoint_history": checkpoint_history,
        "learning_rate": args.learning_rate,
        "codebook_optimizer": args.codebook_optimizer,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "gradient_clip_mode": args.gradient_clip_mode,
        "trainable_sides": args.trainable_sides,
        "trainable_stages": list(trainable_stages),
        "stage_learning_rate_decay": args.stage_learning_rate_decay,
        "stage_selective_tac": {
            "enabled": len(trainable_stages) < UNIFORM_DEPTH,
            "optimized_u16_stages": list(trainable_stages),
            "frozen_u16_stages": [stage for stage in range(UNIFORM_DEPTH) if stage not in trainable_stages],
            "forward_uses_all_u16_stages": True,
            "shared_index_stream_preserved": True,
            "gradient_mask_applied_before_geometry_and_clipping": True,
        },
        "frozen_key_codebooks": args.trainable_sides == "value_only",
        "key_update_geometry": {
            "mode": args.key_update_geometry,
            "key_formula": (
                "raw K gradient <- (L^T L)^-1 raw K gradient"
                if args.key_update_geometry == "whitened"
                else "identity"
            ),
            "value_formula": "identity",
            "forward_indices_or_rate_changed": False,
        },
        "gradient_clip_grouping": (
            "one independent group per (layer,K/V); no upward normalization"
            if args.gradient_clip_mode == "per_layer_kv"
            else (
                "one shared group for all K tables plus one independent group per V layer"
                if args.gradient_clip_mode == "key_global_value_per_layer"
                else "all trainable codebook parameters in one historical global group"
            )
        ),
        "relative_update_trust_region": {
            "enabled": bool(args.relative_update_cap),
            "maximum_relative_update": args.relative_update_cap,
            "grouping": "one independent group per (layer,K/V)",
            "optimizer_contract": (
                "plain SGD, zero weight decay; exact delta=-lr*post_clip_grad"
                if args.relative_update_cap
                else None
            ),
        },
        "frozen_assignment_hard_em": {
            "enabled": frozen_assignment_mstep,
            "assignment_microsteps": args.assignment_microsteps,
            "e_step": (
                "literal FP16-roundtripped hard 16-stage nearest-neighbour assignment; "
                "indices captured CPU int16 without raw-surface retention"
                if frozen_assignment_mstep
                else None
            ),
            "m_step": (
                "literal selected FP16-roundtripped codeword gather with fixed E-step "
                "indices; CE gradients update selected codewords and identity STE passes "
                "to the incoming state"
                if frozen_assignment_mstep
                else None
            ),
            "refresh": (
                "after every complete M-step cycle, exact normal hard U16 current-batch "
                "NLL is recomputed; all codebooks restore to pre-cycle snapshot when "
                "post NLL exceeds E-step NLL plus tolerance"
                if frozen_assignment_mstep
                else None
            ),
            "exact_train_nll_tolerance": args.frozen_assignment_exact_train_nll_tolerance,
            "history": frozen_assignment_mstep_history if frozen_assignment_mstep else [],
        },
        "checkpoint_selection": args.checkpoint_selection,
        "token_chunk_size": args.token_chunk_size,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpoint_side_loss_protocol": (
            "non-reentrant first-forward TM loss snapshot; replay appends ignored"
            if args.gradient_checkpointing and args.trajectory_match_weight
            else None
        ),
        "trajectory_match_weight": args.trajectory_match_weight,
        "trajectory_match_epsilon": args.trajectory_match_epsilon,
        "trajectory_reference_stop_gradient": bool(args.trajectory_match_weight),
        "trajectory_reference_input": (
            "same quantized incoming hidden trajectory; current layer raw K/V only"
            if args.trajectory_match_weight
            else None
        ),
        "assignment_soft_backward": {
            "enabled": a2_soft_backward,
            "topk": args.assignment_soft_topk,
            "temperature_start": args.assignment_soft_temperature_start,
            "temperature_end": args.assignment_soft_temperature_end,
            "temperature_interpolation": (
                "geometric" if a2_soft_backward else None
            ),
            "beta": args.assignment_soft_beta,
            "hard_tail_epochs": args.assignment_soft_hard_tail_epochs,
            "forward_contract": (
                "custom autograd forward returns literal hard gathers; indices, "
                "reconstruction, and bitstream are hard-baseline bitwise identical"
                if a2_soft_backward
                else "historical hard baseline"
            ),
        },
        "anchor_effect_ae0": (
            {
                "enabled": True,
                "anchor_trainable": False,
                "effect_parameterization": "full per-head per-stage per-codeword",
                "effect_initialization": "bitwise FP16 equal to anchor",
                "current_layer_index_dependency": "anchor search only",
                "shared_dynamic_index_stream": True,
                "dynamic_rate_bpa": 1.0,
                "anchor_immutability_gate": anchor_immutability_gate,
                "failed_components_excluded": [
                    "trajectory-matched attention loss",
                    "top-k soft backward",
                    "selector/mixed depth",
                    "stage17",
                ],
            }
            if args.anchor_effect
            else None
        ),
        "active_layers": active_layers,
        "smoke_layers": args.smoke_layers,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "selected_epoch": selected_epoch,
        "selected_step": selected_step,
        "validation_before": validation_before,
        "comparison_validation": comparison_metrics,
        "validation_fixed_final": validation_fixed_final,
        "validation_heldout_best_secondary": validation_heldout_best,
        "validation_after": validation_after,
        "trajectory_validation_before": trajectory_validation_before,
        "trajectory_validation_after": trajectory_validation_after,
        "optimizer_history": history,
        "contract_audit": contract,
        "backbone_audit": backbone_audit,
        "parameter_change_diagnostics": changes,
        "imported_first16_logical_sha256": lineage["imported_first16_logical_sha256"],
        "final_first16_logical_sha256": final_hash,
        "final_effect_logical_sha256": final_hash if args.anchor_effect else None,
        "anchor_immutability_gate": anchor_immutability_gate,
        "runtime": runtime,
    }
    manifest_path = save_artifact(
        output_dir,
        input_dir,
        source_manifest,
        source_payloads,
        codec,
        training=training,
        lineage=lineage,
    )
    result = {
        "format": (
            "jointkv_line_a_anchor_effect_ae0_result_v1"
            if args.anchor_effect
            else "jointkv_line_a_pure_uniform16_result_v1"
        ),
        "status": "complete",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "lineage": lineage,
        "contract_audit": contract,
        "initializer_overlap_gate": overlap_gate,
        "validation_before": validation_before,
        "comparison_validation": comparison_metrics,
        "checkpoint_selection": args.checkpoint_selection,
        "selected_step": selected_step,
        "validation_fixed_final": validation_fixed_final,
        "validation_heldout_best_secondary": validation_heldout_best,
        "validation_after": validation_after,
        "trajectory_validation_before": trajectory_validation_before,
        "trajectory_validation_after": trajectory_validation_after,
        "relative_ppl_change_vs_initializer_percent": 100.0
        * (validation_after["perplexity"] / validation_before["perplexity"] - 1.0),
        "parameter_change_diagnostics": changes,
        "anchor_immutability_gate": anchor_immutability_gate,
        "final_effect_logical_sha256": final_hash if args.anchor_effect else None,
        "runtime": runtime,
    }
    result_path = output_dir / "result.json"
    atomic_json_save(result, result_path)
    print(json.dumps({**result, "result": str(result_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
