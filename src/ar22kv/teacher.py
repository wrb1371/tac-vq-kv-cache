"""RAM-only native media-prefix acceleration for quality measurements.

Original BF16 attention processes the media once. Queries and labels never enter
the snapshot. An explicit replay gate is required before the full benchmark.
These timings are never used as native deployment speed measurements.
"""
import hashlib

import torch
from transformers.cache_utils import DynamicCache

from ar18kv.prefix import media_prefill


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class NativeMediaTeacher:
    def __init__(self, model):
        self.model = model
        self.raw = {}
        self.q_moments = {}
        self.active = False
        self.handles = []
        for layer, block in enumerate(model.thinker.model.layers):
            def key_hook(module, args, output, layer=layer):
                if self.active:
                    self.raw[layer] = output.detach().view(1, -1, 2, 128).transpose(1, 2).contiguous().cpu()

            def query_hook(module, args, output, layer=layer):
                if self.active:
                    q = output.detach().view(1, -1, 2, 8, 128)
                    take = torch.linspace(0, q.shape[1]-1, min(128, q.shape[1]), device=q.device).long()
                    self.q_moments[layer] = q[:, take].float().square().mean((0, 1, 3)).cpu()

            self.handles.append(block.self_attn.k_proj.register_forward_hook(key_hook))
            self.handles.append(block.self_attn.q_proj.register_forward_hook(query_hook))

    def capture(self, inputs, positions, split, input_identity):
        self.raw, self.q_moments = {}, {}
        cache = DynamicCache()
        self.active = True
        try:
            with torch.inference_mode():
                media_prefill(self.model, inputs, cache, positions, split, False)
        finally:
            self.active = False
        torch.cuda.synchronize()
        assert len(cache.key_cache) == len(cache.value_cache) == len(self.raw) == 36
        self.states = {}
        for layer in range(36):
            k = cache.key_cache[layer].detach().contiguous().cpu()
            v = cache.value_cache[layer].detach().contiguous().cpu()
            assert k.shape == v.shape == self.raw[layer].shape == (1, 2, split, 128)
            self.states[layer] = {'K_post': k, 'V': v, 'K_pre': self.raw[layer]}
        del cache
        self.media_end = split
        self.prefix_ids_SHA256 = tensor_sha(inputs['input_ids'][:, :split].cpu())
        self.prefix_positions_SHA256 = tensor_sha(positions[:, :split].cpu())
        self.media_features_SHA256 = input_identity['native_media_feature_SHA256']
        return {'CPU_native_prefix_bytes': sum(t.numel()*t.element_size()
                for state in self.states.values() for t in state.values()),
                'media_end': split, 'prefix_ids_SHA256': self.prefix_ids_SHA256,
                'prefix_positions_SHA256': self.prefix_positions_SHA256,
                'question_or_answer_used': False, 'full_dense_KV_written_to_disk': False}

    def assert_same_media(self, inputs, positions, split, identity):
        assert split == self.media_end
        assert tensor_sha(inputs['input_ids'][:, :split].cpu()) == self.prefix_ids_SHA256
        assert tensor_sha(positions[:, :split].cpu()) == self.prefix_positions_SHA256
        assert identity['native_media_feature_SHA256'] == self.media_features_SHA256

    def populate(self, cache):
        if not hasattr(cache, 'codec'):
            cache.key_cache = [self.states[layer]['K_post'].cuda() for layer in range(36)]
            cache.value_cache = [self.states[layer]['V'].cuda() for layer in range(36)]
            cache._seen_tokens = self.media_end
            return
        # Preserve the original Cache append/arena/accounting path; only the
        # immutable before-question media K/V supplier differs.
        cache.key_cache = [torch.empty(0, device='cuda') for _ in range(36)]
        cache.value_cache = [torch.empty(0, device='cuda') for _ in range(36)]
        for layer in range(36):
            state = self.states[layer]
            cache.pre[layer] = state['K_pre'].cuda()
            cache.incoming[layer] = (state['K_post'].cuda(), state['V'].cuda())
            cache.encode_after_attention(layer)
        cache._seen_tokens = self.media_end

    def close(self):
        for hook in self.handles:
            hook.remove()
        self.handles = []
        self.raw, self.q_moments = {}, {}
        self.states = {}


def compressed_checkpoint(cache):
    """One profile's arena bytes and scalar counters, CPU RAM only."""
    return {layer: {'tensor': state['arena'].tensor.detach().cpu().clone(),
                    'scalars': {key: value for key, value in state.items()
                                if isinstance(value, (int, float, bool, str))}}
            for layer, state in cache.states.items()}


def restore_compressed_checkpoint(cache, saved, media_end):
    assert not cache.pre and not cache.incoming
    for layer, record in saved.items():
        state = cache.states[layer]
        state['arena'].tensor.copy_(record['tensor'])
        for key, value in record['scalars'].items():
            state[key] = value
    cache.seen_positions = media_end
    cache._seen_tokens = media_end
    cache.registry.observe('quality_only_compressed_prefix_RAM_restore')
