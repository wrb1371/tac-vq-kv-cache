"""Native before-question media capture and label-free empirical-Fisher readout."""

import contextlib, hashlib, pathlib, sys
import torch
from transformers.cache_utils import DynamicCache

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/OmniKVQuant_snapshot")]
from ar32kv.runtime import Supplier, answer


def sha_tensor(t):
    return hashlib.sha256(
        t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def load(kind):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if kind == "VL":
        from ar28kv.runtime import load as base

        return base()
    from r14kv.common import runtime
    from ar20kv.attention import install_reference_attention

    torch.cuda.set_per_process_memory_fraction(
        26 * 2**30 / torch.cuda.get_device_properties(0).total_memory
    )
    model, p, _, _ = runtime()
    install_reference_attention(model, "gqa")
    return model, p


def geometry(model, kind):
    language = model.thinker.model if kind == "Omni" else model.model.language_model
    c = language.config
    return language, (
        c.num_hidden_layers,
        c.num_attention_heads,
        c.num_key_value_heads,
        c.hidden_size // c.num_attention_heads,
    )


class Teacher:
    def __init__(self, model, kind):
        self.model = model
        self.kind = kind
        self.language, self.geo = geometry(model, kind)
        if kind == "VL":
            from ar26kv.runtime import Teacher as T
        else:
            from ar22kv.teacher import NativeMediaTeacher as T
        self.base = T(model)

    def capture(self, x, pos, end, identity):
        if self.kind == "VL":
            self.base.capture(x, pos, end)
        else:
            self.base.capture(x, pos, end, identity)
        self.states = {
            l: {k: (v[0] if v.ndim == 4 else v) for k, v in st.items()}
            for l, st in self.base.states.items()
        }
        self.end = end

    def assert_same(self, x, pos, end, identity):
        if self.kind == "VL":
            self.base.assert_same(x, pos, end)
        else:
            self.base.assert_same_media(x, pos, end, identity)

    def bf16_cache(self):
        c = DynamicCache()
        c.key_cache = [s["K_post"][None].cuda() for s in self.states.values()]
        c.value_cache = [s["V"][None].cuda() for s in self.states.values()]
        c._seen_tokens = self.end
        return c

    def clear(self):
        self.states = {}
        self.base.states = {}
        self.base.raw = {}

    def close(self):
        self.base.close()


def native_rope(model, kind, pre, pos, end):
    language, geo = geometry(model, kind)
    pp = pos[:, :end][:, None] if kind == "Omni" else pos[:, :, :end]
    if kind == "Omni":
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            apply_multimodal_rotary_pos_emb,
        )
    else:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            apply_multimodal_rotary_pos_emb,
        )
    co, si = language.rotary_emb(pre[None], pp)
    return apply_multimodal_rotary_pos_emb(
        pre[None],
        pre[None],
        co,
        si,
        language.layers[0].self_attn.rope_scaling["mrope_section"],
    )[1]


def fisher_readout(model, p, kind, teacher, x, pos, end, vi):
    assert not model.training and all(not v.requires_grad for v in model.parameters())
    n = x["input_ids"].shape[1]
    leaves = []
    cache = DynamicCache()
    cache._seen_tokens = end
    cache.key_cache = []
    cache.value_cache = []
    for l, st in teacher.states.items():
        kp = st["K_pre"].cuda().clone().requires_grad_(True)
        vv = st["V"].cuda().clone().requires_grad_(True)
        rotated = native_rope(model, kind, kp, pos, end)
        assert torch.equal(rotated.detach()[0], st["K_post"].cuda()), (
            "native Kpre/RoPE identity",
            l,
        )
        cache.key_cache.append(rotated)
        cache.value_cache.append(vv[None])
        leaves.append((kp, vv))
    if kind == "Omni":
        from ar18kv.prefix import question_forward

        logits = question_forward(model, x, cache, pos, end, False)[0].float()
    else:
        from ar26kv.runtime import last_projection

        with torch.autograd.graph.save_on_cpu(pin_memory=True), last_projection(model):
            out = model(
                input_ids=x["input_ids"][:, end:],
                attention_mask=x["attention_mask"],
                position_ids=pos[:, :, end:],
                cache_position=torch.arange(end, n, device="cuda"),
                past_key_values=cache,
                use_cache=True,
            )
        logits = out.logits[0, -1].float()
    target = logits.detach().argmax()
    loss = torch.nn.functional.cross_entropy(logits[None], target[None])
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    take = vi[torch.linspace(0, len(vi) - 1, min(512, len(vi)), device="cuda").long()]
    result = {}
    for l, (k, v) in enumerate(leaves):
        assert k.grad is not None and v.grad is not None
        result[l] = {
            "K": teacher.states[l]["K_pre"][:, take.cpu()].clone(),
            "V": teacher.states[l]["V"][:, take.cpu()].clone(),
            "K_grad": k.grad[:, take].cpu().clone(),
            "V_grad": v.grad[:, take].cpu().clone(),
        }
        assert all(torch.isfinite(z).all() for z in result[l].values())
        assert all(
            float(result[l][s + "_grad"].float().square().sum()) > 0 for s in ["K", "V"]
        )
    record = {
        "loss": float(loss.detach()),
        "target_token_id": int(target),
        "target_text": p.tokenizer.decode([int(target)]),
        "all_layers_native_K_RoPE_exact": True,
        "samples_per_head": len(take),
        "all_gradient_tensors_finite_nonzero": True,
        "gradient_dtype": str(leaves[0][0].grad.dtype),
        "calibration_prompt": "Describe the video.",
        "evaluation_question_answer_used": False,
        "GPU_peak_MiB": torch.cuda.max_memory_allocated() / 2**20,
    }
    del cache, leaves, logits, loss
    assert all(v.grad is None for v in model.parameters())
    return result, record
