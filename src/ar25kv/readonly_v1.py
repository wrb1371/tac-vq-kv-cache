"""Read-only single-forward suffix scoring; prepaid tail capacity is retained.

Current suffix K/V still participates natively in causal GQA. Only persistence
AFTER each layer's attention is elided; no following token may consume it.
"""
from contextlib import contextmanager

@contextmanager
def read_only_suffix(cache):
 assert hasattr(cache,'codec') and cache.meta.generation_cap==1
 start=cache.seen_positions;old_seen=cache._seen_tokens
 assert len(cache.states)==36 and not cache.incoming and not cache.pre
 assert all(s['n']==start for s in cache.states.values())
 original=cache.encode_after_attention;seen=set();skipped=0
 def skip(layer):
  nonlocal skipped
  assert layer not in seen,'READONLY_REQUIRES_EXACTLY_ONE_FORWARD'
  assert cache.states[layer]['n']==start and layer in cache.incoming
  key,value=cache.incoming.pop(layer);cache.pre.pop(layer,None)
  assert key.shape[-2]==value.shape[-2]
  skipped+=key.shape[-2];seen.add(layer)
  assert cache.key_cache[layer].numel()==cache.value_cache[layer].numel()==0
 cache.encode_after_attention=skip
 completed=False
 try:
  yield
  assert seen==set(range(36)) and not cache.pre and not cache.incoming
  assert cache.seen_positions>start and all(s['n']==start for s in cache.states.values())
  completed=True
 finally:
  cache.encode_after_attention=original;cache.seen_positions=start;cache._seen_tokens=old_seen
  if completed:
   cache._ar25_readonly_calls=getattr(cache,'_ar25_readonly_calls',0)+1
   cache._ar25_skipped_layer_tokens=getattr(cache,'_ar25_skipped_layer_tokens',0)+skipped
   cache.registry.observe('read_only_single_forward_suffix_complete')
   cache.assert_compact()

def install_scoring_operation():
 """Explicitly opt in by policy; other execution and multi-step calls unchanged."""
 import ar22kv.evaluation as evaluation
 if getattr(evaluation,'_ar25_readonly_installed',False):return
 original_answer,original_receipt=evaluation.answer,evaluation.receipt
 def answer(model,processor,cache,inputs,positions,media_end,item):
  use=hasattr(cache,'codec') and cache.codec.policy.get('read_only_single_step',False)
  if not use:return original_answer(model,processor,cache,inputs,positions,media_end,item)
  assert cache.seen_positions==media_end
  with read_only_suffix(cache):return original_answer(model,processor,cache,inputs,positions,media_end,item)
 def receipt(cache,inputs,media_end):
  r=original_receipt(cache,inputs,media_end)
  if hasattr(cache,'_ar25_readonly_calls'):
   r.update(read_only_single_forward_scoring=True,query_writeback_elided=True,prepaid_query_capacity_unchanged=True,read_only_queries_completed=cache._ar25_readonly_calls,skipped_query_layer_tokens=cache._ar25_skipped_layer_tokens,returned_state_scope='immutable media prefix; ephemeral question suffix is not a resumable generation prefix')
  return r
 evaluation.answer=answer;evaluation.receipt=receipt;evaluation._ar25_readonly_installed=True
