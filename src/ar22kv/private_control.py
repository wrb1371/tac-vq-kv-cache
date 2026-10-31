"""Budget-tier GSRQ control with an optional exact fused private-book reader."""
import torch
import triton
import triton.language as tl

from ar18kv.strong import HybridPrivateProtected
from r14kv.common import ROOT


@triton.jit(do_not_specialize=['N', 'IS0', 'IS1', 'BS0'])
def _lookup_private(I, B, O, N, IS0, IS1, BS0, R: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    row, dim = offset//128, offset % 128
    head, token = row//N, row % N
    result = tl.full((BLOCK,), 0., tl.float32)
    for stage in tl.static_range(R):
        code = tl.load(I+head*IS0+token*IS1+stage, row < 2*N, 0).to(tl.int32)
        value = tl.load(B+head*BS0+stage*256*128+code*128+dim, row < 2*N, 0)
        result = result+value
    tl.store(O+offset, result, row < 2*N)


def private_read(index, books):
    n, stages = index.shape[1:]
    # Resident codebooks remain FP8. Only the currently decoded layer's book
    # is expanded in transient workspace, never cached as a second dictionary.
    expanded = books.float()
    output = torch.empty((2, n, 128), device=index.device, dtype=torch.float32)
    _lookup_private[(triton.cdiv(2*n*128, 256),)](
        index, expanded, output, n, index.stride(0), index.stride(1),
        expanded.stride(0), stages, 256, enable_fp_fusion=False)
    return output


class TierPrivate(HybridPrivateProtected):
    def __init__(self, package, budget_mib, **policy):
        mapping = policy.pop('package_by_budget')
        chosen = mapping[str(int(budget_mib))]
        self.tier_package = chosen
        super().__init__(ROOT/chosen, budget_mib, **policy)
        self.metadata.update(tier_package=chosen,
             fixed_dictionary_per_budget_tier=True,
             read_backend='fused private vector lookup + original MRoPE' if policy.get('read_mode', 'fused') == 'fused' else 'original Torch reader',
             high_precision_dictionary_mirror_persistent=False)

    def make_plan(self, meta):
        plan = super().make_plan(meta)
        plan.update(tier_package=self.tier_package,
                    tier_package_completion_SHA256=self.package_sha256,
                    fixed_dictionary_stages=self.package['Rmax_K'],
                    read_mode=self.policy.get('read_mode', 'fused'))
        return plan

    def raw_decode_kind(self, layer, kind, index):
        if self.policy.get('read_mode', 'fused') == 'torch':
            return super().raw_decode_kind(layer, kind, index)
        return private_read(index, self.tables[layer][kind])
