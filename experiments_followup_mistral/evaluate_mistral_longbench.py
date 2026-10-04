#!/usr/bin/env python3
"""Mistral-v0.1 native-window LongBench-7 with packed GSRQ indices."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import types

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv

from experiments_followup_mistral.train_mistral_v_only import make_codec, sha
from release_config import MODEL_INSTRUCT
from longbench_utils import load_official_assets, score_task


ROOT = Path(__file__).resolve().parents[1]
MODEL = MODEL_INSTRUCT


class LayerCache:
    def __init__(self, layer, codec):
        self.layer = layer
        self.codec = codec
        self.key = None
        self.value = None
        self.decoded_key = None
        self.decoded_value = None
        self.length = 0

    def reset(self):
        self.key = None; self.value = None
        self.decoded_key = None; self.decoded_value = None
        self.length = 0

    def append(self, key, value, decoded_key=None, decoded_value=None):
        if self.codec is not None:
            assert key.dtype == value.dtype == torch.uint8
        elif key.dtype != torch.float16 or value.dtype != torch.float16:
            raise RuntimeError("raw cache dtype changed")
        self.key = key if self.key is None else torch.cat((self.key, key), dim=2)
        self.value = value if self.value is None else torch.cat((self.value, value), dim=2)
        if self.codec is not None:
            assert decoded_key is not None and decoded_value is not None
            self.decoded_key = decoded_key if self.decoded_key is None else torch.cat(
                (self.decoded_key, decoded_key), dim=2)
            self.decoded_value = decoded_value if self.decoded_value is None else torch.cat(
                (self.decoded_value, decoded_value), dim=2)
            # The decoded window is a declared evaluation acceleration workspace.
            self.decoded_key = self.decoded_key[:, :, -4096:].contiguous()
            self.decoded_value = self.decoded_value[:, :, -4096:].contiguous()
        self.length += key.shape[2]

    def visible(self):
        start = max(0, self.length - 4096)
        if self.codec is not None:
            key = self.decoded_key
            value = self.decoded_value
        else:
            key = self.key[:, :, start:self.length]
            value = self.value[:, :, start:self.length]
        return key, value, start

    def payload_bytes(self):
        return sum(x.numel() * x.element_size() for x in (self.key, self.value))

    def decode_workspace_bytes(self):
        if self.codec is None:
            return 0
        return sum(x.numel() * x.element_size() for x in (self.decoded_key, self.decoded_value))


def attention_forward(self, hidden_states, attention_mask=None, position_ids=None,
                      past_key_value=None, output_attentions=False, use_cache=False,
                      cache_position=None, **kwargs):
    del cache_position, kwargs
    if past_key_value is not None or use_cache or output_attentions:
        raise ValueError("custom cache requires no native past_key_value/use_cache")
    cache = self._mistral_followup_cache
    batch, query_length, _ = hidden_states.shape
    if batch != 1 or position_ids is None:
        raise ValueError("Mistral followup generation uses batch one and absolute positions")
    q = self.q_proj(hidden_states).view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(batch, query_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(batch, query_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    if cache.codec is not None:
        quant = cache.codec(cache.layer, k, v, mode="uniform")
        k, v = quant.key, quant.value
        cache.append(quant.key_indices, quant.value_indices,
                     decoded_key=quant.key, decoded_value=quant.value)
    else:
        cache.append(k, v)
    if query_length > 1 or cache.length == 1:
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        mask = attention_mask[:, :, :, :query_length] if attention_mask is not None else None
        if mask is not None:
            q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                 dropout_p=0.0,
                                                 is_causal=mask is None and query_length > 1)
    else:
        visible_k, visible_v, start = cache.visible()
        query_cos, query_sin = self.rotary_emb(v, position_ids)
        q, _ = apply_rotary_pos_emb(q, k, query_cos, query_sin)
        all_positions = torch.arange(start, cache.length, device=q.device).unsqueeze(0)
        key_cos, key_sin = self.rotary_emb(visible_k, all_positions)
        _, visible_k = apply_rotary_pos_emb(visible_k, visible_k, key_cos, key_sin)
        visible_k = repeat_kv(visible_k, self.num_key_value_groups)
        visible_v = repeat_kv(visible_v, self.num_key_value_groups)
        output = F.scaled_dot_product_attention(q, visible_k, visible_v,
                                                 attn_mask=None, dropout_p=0.0, is_causal=False)
    output = output.transpose(1, 2).contiguous().view(batch, query_length, -1)
    return self.o_proj(output), None, None


def bind_endpoint(name, device):
    if name == "fp16":
        return None, {"endpoint": name, "model_config_sha256": sha(MODEL / "config.json")}
    fit_path = ROOT / "protocols_followup_mistral/instruct_parent_fit.json"
    fit = json.loads(fit_path.read_text())
    parent = fit["parents"]["gsrq1"]
    assert fit["status"] == "complete" and fit["model"] == str(MODEL)
    codec = make_codec("gsrq1", Path(parent["path"]), device)
    identity = {"endpoint": name, "parent_manifest_sha256": sha(fit_path),
                "parent_layer_sha256": [x["sha256"] for x in parent["layer_files"]]}
    if name != "gsrq1_parent":
        if not name.startswith("gsrq1_mix_seed") or not name.endswith("_qat"):
            raise ValueError(name)
        seed = int(name.removeprefix("gsrq1_mix_seed").removesuffix("_qat"))
        assert seed in (8601, 8607, 8613)
        receipt_path = ROOT / f"results_followup_mistral/training_runs/instruct_gsrq1_mix_seed{seed}.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "complete" and receipt["completed_steps"] == 512
        assert receipt["identity"]["parent_layer_sha256"] == identity["parent_layer_sha256"]
        assert len(receipt["value_layer_files"]) == 32
        for layer, item in enumerate(receipt["value_layer_files"]):
            path = Path(item["path"])
            assert sha(path) == item["sha256"]
            codec.value_tables[layer].data.copy_(torch.load(path, map_location="cpu", weights_only=False)["value_table"].to(device))
        identity["receipt_sha256"] = sha(receipt_path)
        identity["value_layer_sha256"] = [x["sha256"] for x in receipt["value_layer_files"]]
    return codec, identity


@torch.no_grad()
def generate(model, caches, row, device):
    for cache in caches:
        cache.reset()
    ids = torch.tensor(row["input_ids"], device=device, dtype=torch.long)[None]
    prompt_length = ids.shape[1]
    positions = torch.arange(prompt_length, device=device)
    hidden = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                         position_ids=positions[None], cache_position=positions,
                         use_cache=False).last_hidden_state
    logits = model.lm_head(hidden[:, -1:]).float()[:, -1]
    generated = []
    for step in range(row["max_new_tokens"]):
        if row["prohibit_first_eos"] and step == 0:
            logits = logits.clone(); logits[:, row["stop_token_ids"]] = -float("inf")
        token = int(logits.argmax(-1))
        generated.append(token)
        if token in row["stop_token_ids"] or step + 1 == row["max_new_tokens"]:
            break
        absolute = prompt_length + step
        current = torch.tensor([[token]], device=device, dtype=torch.long)
        position = torch.tensor([absolute], device=device, dtype=torch.long)
        hidden = model.model(input_ids=current, attention_mask=torch.ones((1, absolute + 1),
                             device=device, dtype=torch.long), position_ids=position[None],
                             cache_position=position, use_cache=False).last_hidden_state
        logits = model.lm_head(hidden[:, -1:]).float()[:, -1]
    lengths = {cache.length for cache in caches}
    if len(lengths) != 1:
        raise RuntimeError("layer caches diverged")
    cache_length = next(iter(lengths))
    assert cache_length == prompt_length + len(generated) - 1
    per_layer_payload = {cache.payload_bytes() for cache in caches}
    if len(per_layer_payload) != 1:
        raise RuntimeError("unequal layer payload")
    per_layer_workspace = {cache.decode_workspace_bytes() for cache in caches}
    if len(per_layer_workspace) != 1:
        raise RuntimeError("unequal layer decode workspace")
    return generated, {"logical_cache_tokens": cache_length,
                       "bytes_per_layer": next(iter(per_layer_payload)),
                       "decode_workspace_bytes_per_layer": next(iter(per_layer_workspace)),
                       "visible_window": 4096,
                       "cache_dtype": "uint8 GSRQ index plus declared FP16 decode workspace" if caches[0].codec else "float16 raw KV"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=1350)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    assert 0 <= args.start <= args.end <= 1350
    manifest_path = ROOT / "protocols_followup_mistral/table3_eval_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "frozen_before_gpu" and manifest["model"] == str(MODEL)
    data_path = Path(manifest["prompt_records"])
    assert sha(data_path) == manifest["prompt_records_sha256"]
    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 1350 and all(x["global_index"] == i for i, x in enumerate(rows))
    codec, endpoint_identity = bind_endpoint(args.endpoint, args.device)
    identity = {"endpoint": endpoint_identity, "manifest_sha256": sha(manifest_path),
                "runner_sha256": sha(Path(__file__)), "dtype": "float16", "SWA": 4096}
    output = ROOT / f"results_followup_mistral/table3_prediction_records/{args.endpoint}"
    output.mkdir(parents=True, exist_ok=True)
    todo = []
    for i in range(args.start, args.end):
        path = output / f"record_{i:04d}.json"
        if path.exists():
            old = json.loads(path.read_text())
            assert old["identity"] == identity and old["token_sha256"] == rows[i]["token_sha256"]
        else:
            todo.append(i)
    print(json.dumps({"endpoint": args.endpoint, "todo": len(todo), "range": [args.start, args.end]}), flush=True)
    if not todo:
        return
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa", low_cpu_mem_usage=True,
                                                 local_files_only=True).to(args.device).eval()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    caches = []
    for layer, block in enumerate(model.model.layers):
        cache = LayerCache(layer, codec)
        caches.append(cache)
        block.self_attn._mistral_followup_cache = cache
        block.self_attn.forward = types.MethodType(attention_forward, block.self_attn)
    for i in todo:
        row = rows[i]
        start = time.time()
        generated, cache_ledger = generate(model, caches, row, args.device)
        assert row["input_tokens"] + len(generated) <= 8192
        item = {"status": "complete", "identity": identity,
                "global_index": i, "task": row["task"], "row": row["row"],
                "token_sha256": row["token_sha256"], "pred": tokenizer.decode(generated, skip_special_tokens=True),
                "answers": row["answers"], "all_classes": row["all_classes"],
                "input_tokens": row["input_tokens"], "generated_tokens": len(generated),
                "generated_token_ids": generated, "cache": cache_ledger,
                "seconds": time.time() - start}
        path = output / f"record_{i:04d}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(path)
        print(json.dumps({"endpoint": args.endpoint, "index": i, "task": row["task"],
                          "seconds": item["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
