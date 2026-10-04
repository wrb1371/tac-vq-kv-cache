#!/usr/bin/env python3
"""Compare Mistral native SDPA with an identity version of the QAT adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from experiments.train_full_model_closed_loop_vq import ClosedLoopRunContext, install_closed_loop_attention
from release_config import INPUTS, MODEL_BASE, MODEL_INSTRUCT


ROOT = Path(__file__).resolve().parents[1]
CACHE = INPUTS / "smoke_tokens.pt"
TARGET = ROOT / "protocols_followup_mistral/model_and_attention_checks.json"
LENGTHS = (64, 256, 1024, 2048, 4097, 5120, 6144, 8192)


class IdentityCodec:
    def __call__(self, _layer, key, value, *, mode):
        assert mode == "uniform"
        return SimpleNamespace(key=key, value=value, projected_depths=None)


def nll(logits: torch.Tensor, ids: torch.Tensor) -> float:
    total = 0.0
    count = ids.shape[1] - 1
    for start in range(0, count, 256):
        end = min(count, start + 256)
        total += float(F.cross_entropy(logits[:, start:end].float().reshape(-1, logits.shape[-1]),
                                       ids[:, start + 1:end + 1].reshape(-1), reduction="sum"))
    return total / count


def one_model(name: str, device: str) -> dict:
    path = MODEL_BASE if name == "base" else MODEL_INSTRUCT
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa", local_files_only=True,
                                                 low_cpu_mem_usage=True).to(device).eval()
    model.config.use_cache = False
    assert model.config.sliding_window == 4096
    original = [layer.self_attn.forward for layer in model.model.layers]
    base_tokens = torch.load(CACHE, map_location="cpu", weights_only=False)["token_ids"][0].long()
    rows = []
    for length in LENGTHS:
        ids = base_tokens.repeat(math.ceil(length / base_tokens.numel()))[:length].unsqueeze(0).to(device)
        digest = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
        positions = torch.arange(length, device=ids.device)
        # This is the mask supplied by MistralModel to every layer in the native path.
        with torch.inference_mode():
            embedded = model.model.embed_tokens(ids)
            mask = model.model._update_causal_mask(None, embedded, positions, None, False, False)
            if mask is None:
                min_visible = 0
            else:
                visible = torch.nonzero(mask[0, 0, -1, :length] == 0, as_tuple=False).flatten()
                min_visible = int(visible[0])
                assert int(visible[-1]) == length - 1
                assert int(visible.numel()) == min(length, model.config.sliding_window)
            assert min_visible == max(0, length - model.config.sliding_window)
            del embedded, mask
            native = model(input_ids=ids, use_cache=False).logits
            native_loss = nll(native, ids)
            context = ClosedLoopRunContext(IdentityCodec(), "uniform")
            install_closed_loop_attention(model, context, active_layers=len(original))
            adapted = model(input_ids=ids, use_cache=False).logits
            adapted_loss = nll(adapted, ids)
            max_diff = float((native.float() - adapted.float()).abs().max())
            mean_diff = float((native.float() - adapted.float()).abs().mean())
            rows.append({"length": length, "input_tokens_sha256": digest,
                         "native_nll": native_loss, "identity_adapter_nll": adapted_loss,
                         "max_abs_logit_difference": max_diff,
                         "mean_abs_logit_difference": mean_diff,
                         "native_visible_key_start_at_final_query": min_visible,
                         "native_visible_key_count_at_final_query": length - min_visible})
            for layer, fn in zip(model.model.layers, original):
                layer.self_attn.forward = fn
            del native, adapted
            torch.cuda.empty_cache()
        print(json.dumps({"model": name, **rows[-1]}), flush=True)
    del model
    torch.cuda.empty_cache()
    return {"status": "complete", "model": str(path), "dtype": "float16",
            "attention_implementation": "sdpa", "sliding_window": 4096,
            "rows": rows,
            "tolerance": {"max_abs_logit_difference": 0.1, "max_abs_nll_difference": 1e-3},
            "pass": all(r["max_abs_logit_difference"] <= 0.1 and
                        abs(r["native_nll"] - r["identity_adapter_nll"]) <= 1e-3 for r in rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("base", "instruct"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(TARGET.read_text()) if TARGET.exists() else {"status": "partial", "models": {}}
    if args.model in result["models"]:
        raise FileExistsError(f"{args.model} already checked")
    result["models"][args.model] = one_model(args.model, args.device)
    result["status"] = "complete" if set(result["models"]) == {"base", "instruct"} else "partial"
    TARGET.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
