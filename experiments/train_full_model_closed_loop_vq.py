#!/usr/bin/env python3
"""Full-model closed-loop training for nested KV VQ and precision allocation.

Every transformer layer is executed on the trajectory produced by quantized
preceding layers.  The backbone stays frozen; final next-token cross entropy
jointly updates all progressive codebooks and a deployable block selector.
The selector's forward actions are projected independently for every request
to a metadata-inclusive one-BPA cap.  No teacher cache, local attention proxy,
or eviction action participates in the training objective.
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
import time
import types
from typing import Any

from datasets import Dataset
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from jointkv.closed_loop_vq import ClosedLoopNestedVQ
from jointkv.nested_vq import validate_stage_bits
from jointkv.table1_protocol import select_document_calibration_windows, tensor_sha256
from release_config import LLAMA3_BASE, C4_TRAIN_ARROW_0

try:
    from build_frozen_gsrq_multi_exit import atomic_json_save, atomic_torch_save, sha256_file
except ModuleNotFoundError:
    from experiments.build_frozen_gsrq_multi_exit import (
        atomic_json_save,
        atomic_torch_save,
        sha256_file,
    )


DEFAULT_MODEL = str(LLAMA3_BASE)
DEFAULT_C4_TRAIN = str(C4_TRAIN_ARROW_0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--c4-train-arrow", default=DEFAULT_C4_TRAIN)
    parser.add_argument("--input-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument("--validation-samples", type=int, default=4)
    parser.add_argument("--train-seed", type=int, default=113)
    parser.add_argument("--validation-seed", type=int, default=211)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--codebook-optimizer",
        choices=("sgd", "adamw"),
        default="sgd",
        help="SGD respects absolute LM-gradient scale near FP16 VQ boundaries",
    )
    parser.add_argument("--selector-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--trust-region-fraction", type=float, default=0.1)
    parser.add_argument("--effective-bits", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--token-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--uniform-anchor-weight",
        type=float,
        default=0.25,
        help="closed-loop uniform-16 CE anchor evaluated on the same trajectory",
    )
    parser.add_argument(
        "--validation-uniform-tolerance",
        type=float,
        default=0.0,
        help="maximum relative held-out uniform-16 NLL regression for checkpoint acceptance",
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-layers", type=int, default=0)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stack_direct_tables(part: dict[str, Any]) -> torch.Tensor:
    shared = part.get("shared_codebooks")
    entries = part.get("entry_factors")
    projections = part.get("dimension_factors")
    if not isinstance(shared, list) or not shared:
        raise ValueError("closed-loop initializer lacks direct codebooks")
    if not isinstance(entries, list) or not isinstance(projections, list):
        raise ValueError("closed-loop initializer lacks factor lists")
    if not (len(shared) == len(entries) == len(projections)):
        raise ValueError("closed-loop initializer stage lists differ")
    if any(entry.shape[-1] or projection.shape[-2] for entry, projection in zip(entries, projections)):
        raise ValueError("closed-loop prototype currently requires direct per-head tables")
    return torch.stack(shared, dim=1).contiguous()


def direct_part(tables: torch.Tensor) -> dict[str, Any]:
    heads, stages, cardinality, dimension = map(int, tables.shape)
    local = tables.detach().half().cpu()
    return {
        "shared_codebooks": [local[:, stage].contiguous() for stage in range(stages)],
        "entry_factors": [torch.empty((heads, cardinality, 0), dtype=torch.float16) for _ in range(stages)],
        "dimension_factors": [torch.empty((heads, 0, dimension), dtype=torch.float16) for _ in range(stages)],
        "residual_mse_by_stage": [],
        "representation": "per_head",
    }


def load_initializer(
    artifact_dir: Path,
    *,
    model_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], ClosedLoopNestedVQ]:
    manifest_path = artifact_dir / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("initializer manifest is incomplete")
    if os.path.realpath(manifest["model"]) != os.path.realpath(model_path):
        raise ValueError("initializer/model paths differ")
    stage_bits = validate_stage_bits(manifest["stage_bits"])
    exits = tuple(int(depth) for depth in manifest["action_depths"])
    if exits != (15, 16, 17) or stage_bits != (8,) * 17:
        raise ValueError("full-model parent currently requires 17x8-bit {15,16,17}")
    layers = [int(layer) for layer in manifest["selected_layers"]]
    if layers != list(range(len(layers))):
        raise ValueError("closed-loop initializer must cover a contiguous full model")
    payloads = []
    key_tables = []
    value_tables = []
    whiteners = []
    for layer in layers:
        path = Path(manifest["codebook_files"][str(layer)]["path"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("value_encoder", {}).get("mode") != "greedy_residual":
            raise ValueError(f"layer {layer} is not a greedy residual codec")
        payloads.append(payload)
        key_tables.append(stack_direct_tables(payload["factorized_codebooks"]["key"]))
        value_tables.append(stack_direct_tables(payload["factorized_codebooks"]["value"]))
        geometry = payload.get("provenance", {}).get("key_assignment_geometry")
        whiteners.append(
            None
            if geometry == "raw_euclidean_gsrq"
            else payload["key_assignment_whiteners"]
        )
    codec = ClosedLoopNestedVQ(
        key_tables,
        value_tables,
        whiteners,
        stage_bits=stage_bits,
        exits=exits,
        block_size=128,  # overwritten after construction from CLI when needed
        head_dim=128,
        effective_bits=float(manifest["effective_bits"]),
        trust_region_fraction=0.1,
    ).to(device)
    return manifest, payloads, codec


class ClosedLoopRunContext:
    def __init__(self, codec: ClosedLoopNestedVQ, mode: str) -> None:
        self.codec = codec
        self.mode = mode
        self.rate_records: list[dict[str, Any]] = []
        # Optional side objective used by the isolated Line-A trajectory-matched
        # experiment.  It is off by default, so all historical callers retain
        # their exact forward and memory behavior.
        self.collect_trajectory_match = False
        self.trajectory_match_epsilon = 1e-8
        self.trajectory_match_losses: list[tuple[int, torch.Tensor]] = []

    def reset(self, mode: str) -> None:
        self.mode = mode
        self.rate_records.clear()
        self.trajectory_match_losses.clear()

    def trajectory_match_mean(self) -> torch.Tensor:
        if not self.trajectory_match_losses:
            raise RuntimeError("trajectory-matched attention losses were not collected")
        return torch.stack([loss for _layer, loss in self.trajectory_match_losses]).mean()


def relative_trajectory_attention_mse(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Scale-invariant local attention loss with a stopped raw reference."""

    if candidate.shape != reference.shape or candidate.ndim != 3:
        raise ValueError("trajectory attention outputs must share [B,T,D] geometry")
    if epsilon <= 0:
        raise ValueError("trajectory attention epsilon must be positive")
    target = reference.detach().float()
    return (candidate.float() - target).square().mean() / target.square().mean().clamp_min(
        epsilon
    )


def install_closed_loop_attention(
    model: torch.nn.Module,
    context: ClosedLoopRunContext,
    *,
    active_layers: int,
) -> None:
    """Replace Llama SDPA attention with quantized pre-RoPE K/V attention."""

    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        if layer_index >= active_layers:
            continue

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_value: Any = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: torch.LongTensor | None = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            _layer_index: int = layer_index,
            **kwargs: Any,
        ):
            del cache_position, kwargs
            if output_attentions or past_key_value is not None or use_cache:
                raise ValueError("closed-loop training requires SDPA without cache/attention outputs")
            batch, query_length, _ = hidden_states.shape
            query_states = self.q_proj(hidden_states).view(
                batch, query_length, self.num_heads, self.head_dim
            ).transpose(1, 2)
            raw_key = self.k_proj(hidden_states).view(
                batch, query_length, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            raw_value = self.v_proj(hidden_states).view(
                batch, query_length, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            quantized = context.codec(
                _layer_index,
                raw_key,
                raw_value,
                mode=context.mode,
            )
            if quantized.projected_depths is not None:
                for request, ledger in enumerate(quantized.projected_depths.ledgers):
                    context.rate_records.append(
                        {
                            "layer": _layer_index,
                            "request": request,
                            "total_bits": ledger.total_bits,
                            "physical_effective_bits": ledger.physical_effective_bits,
                            "low_blocks": quantized.projected_depths.low_blocks_per_request[request],
                            "anchor_blocks": quantized.projected_depths.anchor_blocks_per_request[request],
                            "high_blocks": quantized.projected_depths.high_blocks_per_request[request],
                        }
                    )
            if position_embeddings is None:
                cos, sin = self.rotary_emb(quantized.value, position_ids)
            else:
                cos, sin = position_embeddings
            unrotated_query_states = query_states
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                quantized.key,
                cos,
                sin,
            )
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(quantized.value, self.num_key_value_groups)
            causal_mask = attention_mask
            if causal_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            if query_states.device.type == "cuda" and causal_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()
            output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=0.0,
                is_causal=causal_mask is None and query_length > 1,
            )
            output = output.transpose(1, 2).contiguous().view(batch, query_length, -1)
            projected_output = self.o_proj(output)
            if context.collect_trajectory_match:
                # The reference uses exactly the same quantized incoming hidden
                # trajectory, positions and causal mask.  Only this layer's K/V
                # quantization is counterfactually removed, and the full raw path
                # is stopped so the auxiliary loss updates only the deployed VQ.
                with torch.no_grad():
                    raw_query_states, raw_key_states = apply_rotary_pos_emb(
                        unrotated_query_states.detach(),
                        raw_key.detach(),
                        cos.detach(),
                        sin.detach(),
                    )
                    raw_key_states = repeat_kv(
                        raw_key_states, self.num_key_value_groups
                    )
                    raw_value_states = repeat_kv(
                        raw_value.detach(), self.num_key_value_groups
                    )
                    if raw_query_states.device.type == "cuda" and causal_mask is not None:
                        raw_query_states = raw_query_states.contiguous()
                        raw_key_states = raw_key_states.contiguous()
                        raw_value_states = raw_value_states.contiguous()
                    raw_output = F.scaled_dot_product_attention(
                        raw_query_states,
                        raw_key_states,
                        raw_value_states,
                        attn_mask=causal_mask,
                        dropout_p=0.0,
                        is_causal=causal_mask is None and query_length > 1,
                    )
                    raw_output = raw_output.transpose(1, 2).contiguous().view(
                        batch, query_length, -1
                    )
                    raw_projected_output = self.o_proj(raw_output)
                context.trajectory_match_losses.append(
                    (
                        _layer_index,
                        relative_trajectory_attention_mse(
                            projected_output,
                            raw_projected_output,
                            epsilon=context.trajectory_match_epsilon,
                        ),
                    )
                )
            return projected_output, None, None

        attention.forward = types.MethodType(forward, attention)


def causal_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        tokens[:, 1:].reshape(-1),
        reduction="mean",
    )


def run_loss(
    model: torch.nn.Module,
    context: ClosedLoopRunContext,
    tokens: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    context.reset(mode)
    output = model(input_ids=tokens, use_cache=False, return_dict=True)
    loss = causal_loss(output.logits, tokens)
    return loss, list(context.rate_records)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    context: ClosedLoopRunContext,
    tokens: torch.Tensor,
    *,
    modes: tuple[str, ...] = ("mixed", "uniform"),
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in modes:
        total_nll = 0.0
        total_tokens = 0
        rates = []
        for row in range(tokens.shape[0]):
            loss, records = run_loss(model, context, tokens[row : row + 1], mode=mode)
            count = int(tokens.shape[1] - 1)
            total_nll += float(loss) * count
            total_tokens += count
            rates.extend(records)
        mean_nll = total_nll / total_tokens
        result[mode] = {
            "mean_nll": mean_nll,
            "perplexity": math.exp(mean_nll),
            "tokens": total_tokens,
            "maximum_physical_effective_bits": max(
                (float(row["physical_effective_bits"]) for row in rates),
                default=None,
            ),
            "rate_records": rates,
        }
    return result


def selector_static_bytes(codec: ClosedLoopNestedVQ) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in codec.selector.parameters()
    )


def save_artifact(
    output_dir: Path,
    source_artifact_dir: Path,
    source_manifest: dict[str, Any],
    source_payloads: list[dict[str, Any]],
    codec: ClosedLoopNestedVQ,
    *,
    training: dict[str, Any],
) -> Path:
    codebook_files = {}
    for layer, source in enumerate(source_payloads):
        payload = copy.deepcopy(source)
        payload["factorized_codebooks"] = {
            "key": direct_part(codec.key_tables[layer]),
            "value": direct_part(codec.value_tables[layer]),
        }
        payload.setdefault("training", {})["full_model_closed_loop"] = training
        payload.setdefault("provenance", {})["full_model_closed_loop"] = (
            "all-layer quantized trajectory; final next-token CE; hard residual-VQ "
            "and hard exact-rate depth forward with straight-through backward"
        )
        target = output_dir / "codebooks" / f"layer_{layer:02d}.pt"
        atomic_torch_save(payload, target)
        codebook_files[str(layer)] = {
            **source_manifest["codebook_files"][str(layer)],
            "path": str(target),
            "sha256": sha256_file(target),
            "source": source_manifest["codebook_files"][str(layer)]["path"],
        }
    selector_path = output_dir / "selector.pt"
    atomic_torch_save(
        {
            "format": "jointkv_closed_loop_precision_selector_v1",
            "state_dict": {key: value.detach().float().cpu() for key, value in codec.selector.state_dict().items()},
            "feature_dim": codec.selector.feature_dim,
            "stage_bits": list(codec.stage_bits),
            "action_depths": list(codec.exits),
            "block_size": codec.block_size,
            "effective_bits": codec.effective_bits,
            "trust_region_fraction": codec.trust_region_fraction,
            "temperature": codec.temperature,
            "static_bytes_fp32": sum(value.numel() * 4 for value in codec.selector.state_dict().values()),
        },
        selector_path,
    )
    manifest = copy.deepcopy(source_manifest)
    manifest.update(
        {
            "status": "complete",
            "scope": "full-model closed-loop jointly-trained nested VQ and precision-depth selector",
            "codebook_files": codebook_files,
            "source_artifact_dir": str(source_artifact_dir),
            "selector_file": str(selector_path),
            "selector_sha256": sha256_file(selector_path),
            "selector_static_bytes_fp32": sum(value.numel() * 4 for value in codec.selector.state_dict().values()),
            "inference_action_zero_enabled": False,
            "allocation_scope": "learned per-block selector with strict per-request/per-layer physical cap",
            "training_hyperparameters": {
                **source_manifest.get("training_hyperparameters", {}),
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
    if (
        args.sequence_length <= 1
        or args.train_samples <= 0
        or args.validation_samples <= 0
        or args.epochs <= 0
        or args.learning_rate <= 0
        or args.selector_learning_rate <= 0
        or args.gradient_clip <= 0
        or args.block_size <= 0
        or args.sequence_length % args.block_size
        or not 0 <= args.trust_region_fraction <= 1
        or args.effective_bits <= 0
        or args.temperature <= 0
        or args.token_chunk_size <= 0
        or args.uniform_anchor_weight < 0
        or args.validation_uniform_tolerance < 0
    ):
        raise ValueError("invalid closed-loop training hyperparameters")
    random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    started = time.time()
    model_path = Path(args.model).resolve()
    arrow_path = Path(args.c4_train_arrow).resolve()
    input_dir = Path(args.input_artifact_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_target = output_dir / "training_manifest.json"
    if manifest_target.exists() and not args.overwrite:
        raise FileExistsError(manifest_target)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
    rows = Dataset.from_file(str(arrow_path))
    train_tokens, train_rows = select_document_calibration_windows(
        tokenizer,
        rows,
        samples=args.train_samples,
        sequence_length=args.sequence_length,
        seed=args.train_seed,
    )
    validation_tokens, validation_rows = select_document_calibration_windows(
        tokenizer,
        rows,
        samples=args.validation_samples,
        sequence_length=args.sequence_length,
        seed=args.validation_seed,
    )
    if {row["token_sha256"] for row in train_rows} & {row["token_sha256"] for row in validation_rows}:
        raise ValueError("training and validation windows overlap")

    source_manifest, source_payloads, codec = load_initializer(
        input_dir,
        model_path=model_path,
        device=device,
    )
    initializer_hashes = {
        str(row.get("token_sha256"))
        for row in source_payloads[0].get("provenance", {}).get("sample_manifest", ())
    }
    # Older wrapped artifacts keep the calibration rows directly in the layer
    # provenance; newer ones may expose them only in the root provenance.
    if not initializer_hashes:
        initializer_hashes = {
            str(row.get("token_sha256"))
            for row in source_manifest.get("provenance", {}).get("sample_manifest", ())
        }
    validation_hashes = {str(row["token_sha256"]) for row in validation_rows}
    if initializer_hashes & validation_hashes:
        raise ValueError("held-out validation overlaps the initializer calibration set")
    codec.block_size = args.block_size
    codec.effective_bits = args.effective_bits
    codec.trust_region_fraction = args.trust_region_fraction
    codec.temperature = args.temperature
    codec.token_chunk_size = args.token_chunk_size
    active_layers = len(codec.key_tables)
    if args.smoke_layers:
        if not 0 < args.smoke_layers <= active_layers:
            raise ValueError("smoke layer count is outside the codec")
        active_layers = args.smoke_layers

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    # Llama-3 has zero attention/MLP dropout. Training mode is therefore
    # numerically deterministic here and is required for Transformers to apply
    # non-reentrant gradient checkpointing to the frozen backbone.
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    context = ClosedLoopRunContext(codec, "mixed")
    install_closed_loop_attention(model, context, active_layers=active_layers)
    codebook_parameters = [
        *codec.key_tables.parameters(),
        *codec.value_tables.parameters(),
    ]
    if args.codebook_optimizer == "sgd":
        codebook_optimizer = torch.optim.SGD(
            codebook_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        codebook_optimizer = torch.optim.AdamW(
            codebook_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    selector_optimizer = torch.optim.AdamW(
        codec.selector.parameters(),
        lr=args.selector_learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    step = 0
    best_state = copy.deepcopy(codec.state_dict())
    validation_before = evaluate(
        model,
        context,
        validation_tokens.to(device),
    )
    best_validation = validation_before["mixed"]["mean_nll"]
    uniform_validation_limit = validation_before["uniform"]["mean_nll"] * (
        1.0 + args.validation_uniform_tolerance
    )
    print(json.dumps({"event": "closed_loop_validation_before", "metrics": validation_before}), flush=True)
    for epoch in range(args.epochs):
        order = list(range(args.train_samples))
        random.Random(args.train_seed + epoch).shuffle(order)
        for row in order:
            if args.max_steps and step >= args.max_steps:
                break
            codebook_optimizer.zero_grad(set_to_none=True)
            selector_optimizer.zero_grad(set_to_none=True)
            local_tokens = train_tokens[row : row + 1].to(device)
            mixed_loss, rate_records = run_loss(model, context, local_tokens, mode="mixed")
            # Backpropagate before switching the mutable attention context.
            # Gradient checkpointing replays each attention forward during
            # backward; delaying this until after the uniform pass would make
            # the mixed graph replay with the wrong precision policy.
            mixed_loss.backward()
            total_loss_value = float(mixed_loss.detach())
            uniform_loss = None
            if args.uniform_anchor_weight:
                uniform_loss, _ = run_loss(model, context, local_tokens, mode="uniform")
                (args.uniform_anchor_weight * uniform_loss).backward()
                total_loss_value += args.uniform_anchor_weight * float(uniform_loss.detach())
            per_layer_codec_gradient_norms = []
            for layer_index in range(active_layers):
                squared = 0.0
                for parameter in (
                    codec.key_tables[layer_index],
                    codec.value_tables[layer_index],
                ):
                    if parameter.grad is not None:
                        squared += float(
                            parameter.grad.detach().float().square().sum()
                        )
                per_layer_codec_gradient_norms.append(math.sqrt(squared))
            codec_gradient_norm = torch.nn.utils.clip_grad_norm_(
                [*codec.key_tables.parameters(), *codec.value_tables.parameters()],
                args.gradient_clip,
            )
            selector_gradient_norm = torch.nn.utils.clip_grad_norm_(
                codec.selector.parameters(),
                args.gradient_clip,
            )
            codebook_optimizer.step()
            selector_optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in codec.parameters()):
                raise RuntimeError("closed-loop update produced non-finite parameters")
            step += 1
            local = {
                "step": step,
                "epoch": epoch,
                "sample": row,
                "mixed_loss": float(mixed_loss.detach()),
                "uniform_loss": None if uniform_loss is None else float(uniform_loss.detach()),
                "total_loss": total_loss_value,
                "codec_gradient_norm": float(codec_gradient_norm),
                "selector_gradient_norm": float(selector_gradient_norm),
                "per_layer_codec_gradient_norms": per_layer_codec_gradient_norms,
                "rate_records": rate_records,
            }
            history.append(local)
            print(json.dumps({"event": "closed_loop_train_step", **local}), flush=True)
        validation = evaluate(model, context, validation_tokens.to(device))
        mixed_improved = validation["mixed"]["mean_nll"] < best_validation
        uniform_guard = (
            validation["uniform"]["mean_nll"] <= uniform_validation_limit
        )
        accepted = bool(mixed_improved and uniform_guard)
        if accepted:
            best_validation = validation["mixed"]["mean_nll"]
            best_state = copy.deepcopy(codec.state_dict())
        print(
            json.dumps(
                {
                    "event": "closed_loop_epoch_validation",
                    "epoch": epoch,
                    "accepted": accepted,
                    "mixed_improved": mixed_improved,
                    "uniform_guard": uniform_guard,
                    "metrics": validation,
                }
            ),
            flush=True,
        )
        if args.max_steps and step >= args.max_steps:
            break
    codec.load_state_dict(best_state)
    validation_after = evaluate(model, context, validation_tokens.to(device))
    training = {
        "objective": "final next-token cross entropy on the full quantized model trajectory",
        "discrete_forward": "hard nearest-neighbour VQ and hard exact-rate {15,16,17} block actions",
        "gradient_estimator": "identity STE for VQ state path and softmax STE for projected actions",
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
        "epochs": args.epochs,
        "completed_steps": step,
        "learning_rate": args.learning_rate,
        "codebook_optimizer": args.codebook_optimizer,
        "selector_learning_rate": args.selector_learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "block_size": args.block_size,
        "trust_region_fraction": args.trust_region_fraction,
        "effective_bits": args.effective_bits,
        "temperature": args.temperature,
        "token_chunk_size": args.token_chunk_size,
        "uniform_anchor_weight": args.uniform_anchor_weight,
        "validation_uniform_tolerance": args.validation_uniform_tolerance,
        "uniform_validation_limit": uniform_validation_limit,
        "gradient_checkpointing": args.gradient_checkpointing,
        "active_layers": active_layers,
        "smoke_layers": args.smoke_layers,
        "validation_before": validation_before,
        "validation_after": validation_after,
        "optimizer_history": history,
        "selector_static_bytes_training_fp32": selector_static_bytes(codec),
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = save_artifact(
        output_dir,
        input_dir,
        source_manifest,
        source_payloads,
        codec,
        training=training,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(manifest_path),
                "validation_before": validation_before,
                "validation_after": validation_after,
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
