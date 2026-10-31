"""Common native/quality execution helpers with an explicit reserved query tail."""
import importlib
import re

import torch
from transformers.cache_utils import DynamicCache

from ar18kv.prefix import question_forward
from ar20kv.attention import KernelCache
from model.codec.modality import modality_tags
from r14kv.common import ROOT
from r15kv.budget import RequestMeta


def make_cache(model, spec, item, inputs, positions, identity, media_end):
    if spec['family'] == 'BF16':
        return None, DynamicCache()
    cls = getattr(importlib.import_module(spec['module']), spec['class'])
    budget = spec.get('force_budget_mib', item.get('budget_mib', 16))
    codec = cls(ROOT/spec['package'], budget, **spec.get('policy', {}))
    codec.set_context(tags=modality_tags(inputs['input_ids'], model), positions=positions,
                      input_ids=inputs['input_ids'], item_id=item['id'],
                      video_grid_thw=inputs['video_grid_thw'], generation_cap=1,
                      duration_group=item.get('planner_profile_group', item.get('duration_group', 'short')))
    reserve = item.get('query_reservation_tokens', {'short': 192, 'medium': 256, 'long': 288}[item.get('duration_group', 'short')])
    assert inputs['input_ids'].shape[1]-media_end+1 <= reserve
    # Capacity is a prepaid reservation, not the observed question length.
    maximum = tuple(int(x)+reserve for x in positions[:, :media_end].amax(1).tolist())
    meta = RequestMeta(media_end+reserve, 1, maximum, identity['input_sha256'])
    rotary = model.thinker.model.rotary_emb
    cache_cls = KernelCache
    if spec.get('policy', {}).get('content_adaptive', False):
        from .cache import ContentPlannedCache
        cache_cls = ContentPlannedCache
    cache = cache_cls(codec, rotary.inv_freq, meta, rotary.attention_scaling)
    return codec, cache


def answer(model, processor, cache, inputs, positions, media_end, item):
    letters = [chr(65+i) for i in range(len(item['options']))]
    token_ids = [processor.tokenizer.encode(x, add_special_tokens=False) for x in letters]
    assert all(len(x) == 1 for x in token_ids)
    with torch.inference_mode():
        logits = question_forward(model, inputs, cache, positions, media_end, False)
    assert torch.isfinite(logits).all(), 'NONFINITE_MODEL_LOGITS'
    ids = [x[0] for x in token_ids]
    options = logits[0, ids].float().cpu().tolist()
    token = int(logits.argmax(-1).item())
    text = processor.tokenizer.decode([token], skip_special_tokens=True).strip()
    parsed = re.search(r'\b([A-H])\b', text)
    prediction = parsed.group(1) if parsed else None
    conditional = letters[max(range(len(options)), key=lambda i: options[i])]
    return {'token_ids': [token], 'text': text, 'prediction': prediction,
            'correct': prediction == item['answer'], 'option_letters': letters,
            'option_logits': options, 'conditional_option_prediction': conditional,
            'conditional_option_correct': conditional == item['answer'],
            'scorer_primary': 'greedy first token and fixed option-letter parser',
            'conditional_option_score_secondary_predeclared': True}


def receipt(cache, inputs, media_end):
    if hasattr(cache, 'codec'):
        cache.assert_compact()
        row = cache.receipt()
        row.update(actual_prompt_tokens=inputs['input_ids'].shape[1],
                   prefilling_media_tokens=media_end, prefill_cached_tokens=media_end,
                   prepaid_query_tail_tokens=cache.meta.capacity-media_end, generation_cap=1,
                   quality_oracle_capacity_field_scope='media prefix plus predeclared duration-group query reservation; actual prompt tokens separate')
        assert row['budget_pass']
        return row
    size = sum(x.untyped_storage().nbytes() for x in cache.key_cache+cache.value_cache)
    return {'resident_peak_bytes': size, 'codec_static_bytes': 0,
            'cache_payload_bytes': size, 'budget_reference_only': True,
            'budget_pass': None, 'actual_prompt_tokens': inputs['input_ids'].shape[1]}
