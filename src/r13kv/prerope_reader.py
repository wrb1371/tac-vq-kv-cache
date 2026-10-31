# SPDX-License-Identifier: Apache-2.0
# Prototype K-preR derivative of OmniKVQuant e8d880faae1a35275c8abe41003c8d8d2a8c8b51; native V/mask/merge retained.
import math
from typing import Optional
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None
from model.codec.packed_cache import CODEC_LOCAL_LM
from model.codec.turboquant import lloyd_max

if triton is not None:

    @triton.jit
    def _decode_mxfp4_metadata(packed, exponent_byte):
        magnitude_code = packed & 7
        magnitude = tl.where(
            magnitude_code == 0,
            0.0,
            tl.where(
                magnitude_code == 1,
                0.5,
                tl.where(
                    magnitude_code == 2,
                    1.0,
                    tl.where(
                        magnitude_code == 3,
                        1.5,
                        tl.where(
                            magnitude_code == 4,
                            2.0,
                            tl.where(
                                magnitude_code == 5,
                                3.0,
                                tl.where(magnitude_code == 6, 4.0, 6.0),
                            ),
                        ),
                    ),
                ),
            ),
        )
        sign = tl.where(packed & 8 != 0, -1.0, 1.0)
        scale = tl.exp2(exponent_byte.to(tl.float32) - 127.0)
        return magnitude * sign * scale

    @triton.jit(
        do_not_specialize=(
            "q_s0",
            "q_s1",
            "ki_s0",
            "ki_s1",
            "ki_s2",
            "ks_s0",
            "ks_s1",
            "ks_s2",
            "kms_s0",
            "kms_s1",
            "kms_s2",
            "kss_s0",
            "kss_s1",
            "kss_s2",
            "kn_s0",
            "kn_s1",
            "vi_s0",
            "vi_s1",
            "vi_s2",
            "vn_s0",
            "vn_s1",
            "mask_s0",
            "kcb_s0",
            "N_TOKENS",
            "mrope_s0",
        )
    )
    def _packed_partial_attention_kernel(
        q_ptr,
        k_index_ptr,
        k_min_ptr,
        k_norm_ptr,
        k_span_ptr,
        k_min_mxscale_ptr,
        k_span_mxscale_ptr,
        v_index_ptr,
        v_norm_ptr,
        k_codebook_ptr,
        k_positions_ptr,
        mrope_positions_ptr,
        cos_lut_ptr,
        sin_lut_ptr,
        v_codebook_ptr,
        additive_mask_ptr,
        numerator_ptr,
        max_ptr,
        sum_ptr,
        q_s0,
        q_s1,
        ki_s0,
        ki_s1,
        ki_s2,
        ks_s0,
        ks_s1,
        ks_s2,
        kms_s0,
        kms_s1,
        kms_s2,
        kss_s0,
        kss_s1,
        kss_s2,
        kn_s0,
        kn_s1,
        vi_s0,
        vi_s1,
        vi_s2,
        vn_s0,
        vn_s1,
        mask_s0,
        kcb_s0,
        mrope_s0,
        B: tl.constexpr,
        HQ: tl.constexpr,
        GROUPS: tl.constexpr,
        N_TOKENS,
        D: tl.constexpr,
        K_INDEX_WIDTH: tl.constexpr,
        V_INDEX_WIDTH: tl.constexpr,
        K_LOCAL_LM: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        INV_SQRT_D: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_block = tl.program_id(1)
        batch = pid_bh // HQ
        q_head = pid_bh - batch * HQ
        kv_head = q_head // GROUPS
        offs_t = pid_block * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = tl.arange(0, BLOCK_D)
        valid_t = offs_t < N_TOKENS
        valid_d = offs_d < D
        q = tl.load(
            q_ptr + batch * q_s0 + q_head * q_s1 + offs_d, mask=valid_d, other=0.0
        ).to(tl.float32)
        pair = offs_d % 64
        vector_group = pair // 2
        component = (offs_d // 64) * 2 + (pair % 2)
        bit = vector_group * 6
        byte = bit // 8
        shift = bit % 8
        base = batch * ki_s0 + kv_head * ki_s1 + offs_t[:, None] * ki_s2 + byte[None, :]
        lo = tl.load(k_index_ptr + base, mask=valid_t[:, None] & valid_d[None, :], other=0).to(tl.int32)
        hi = tl.load(k_index_ptr + base + 1,
                     mask=valid_t[:, None] & valid_d[None, :] & (byte[None, :] + 1 < K_INDEX_WIDTH), other=0).to(tl.int32)
        code = ((lo | (hi << 8)) >> shift[None, :]) & 63
        address = kv_head * kcb_s0 + vector_group[None, :] * 64 * 4 + code * 4
        centroid = tl.load(k_codebook_ptr + address + component[None, :],
                          mask=valid_t[:, None] & valid_d[None, :], other=0).to(tl.float32)
        mate = tl.load(k_codebook_ptr + address + ((component + 2) % 4)[None, :],
                      mask=valid_t[:, None] & valid_d[None, :], other=0).to(tl.float32)
        size = tl.sqrt(tl.sum(centroid * centroid, axis=1) + 1e-16)
        norm = tl.load(k_norm_ptr + batch * kn_s0 + kv_head * kn_s1 + offs_t,
                       mask=valid_t, other=0).to(tl.float32)
        pre = (centroid / size[:, None] * norm[:, None]).to(tl.bfloat16).to(tl.float32)
        pre_mate = (mate / size[:, None] * norm[:, None]).to(tl.bfloat16).to(tl.float32)
        global_position = tl.load(k_positions_ptr + offs_t, mask=valid_t, other=0).to(tl.int32)
        axis = tl.where(pair < 16, 0, tl.where(pair < 40, 1, 2))
        coordinate = tl.load(mrope_positions_ptr + axis[None, :] * mrope_s0 + global_position[:, None],
                            mask=valid_t[:, None] & valid_d[None, :], other=0).to(tl.int32)
        cos = tl.load(cos_lut_ptr + coordinate * 64 + pair[None, :],
                      mask=valid_t[:, None] & valid_d[None, :], other=0).to(tl.float32)
        sin = tl.load(sin_lut_ptr + coordinate * 64 + pair[None, :],
                      mask=valid_t[:, None] & valid_d[None, :], other=0).to(tl.float32)
        first = (pre * cos).to(tl.bfloat16).to(tl.float32)
        second = (tl.where(offs_d[None, :] < 64, -pre_mate, pre_mate) * sin).to(tl.bfloat16).to(tl.float32)
        k = (first + second).to(tl.bfloat16).to(tl.float32)
        logits = tl.sum(k * q[None, :], axis=1) * INV_SQRT_D
        additive = tl.load(
            additive_mask_ptr + batch * mask_s0 + offs_t,
            mask=valid_t,
            other=-float("inf"),
        ).to(tl.float32)
        usable = valid_t & (additive > -10000.0)
        logits = tl.where(usable, logits + additive, -float("inf"))
        usable_count = tl.sum(usable.to(tl.int32), axis=0)
        block_max = tl.max(logits, axis=0)
        safe_max = tl.where(usable_count > 0, block_max, 0.0)
        probs = tl.where(usable, tl.exp(logits - safe_max), 0.0)
        block_sum = tl.sum(probs, axis=0)
        v_bit_pos = offs_d * 2
        v_index_byte = v_bit_pos // 8
        v_index_shift = v_bit_pos - v_index_byte * 8
        v_index_base = (
            batch * vi_s0
            + kv_head * vi_s1
            + offs_t[:, None] * vi_s2
            + v_index_byte[None, :]
        )
        vlo = tl.load(
            v_index_ptr + v_index_base,
            mask=valid_t[:, None] & valid_d[None, :],
            other=0,
        ).to(tl.int32)
        vhi = tl.load(
            v_index_ptr + v_index_base + 1,
            mask=valid_t[:, None]
            & valid_d[None, :]
            & (v_index_byte + 1 < V_INDEX_WIDTH)[None, :],
            other=0,
        ).to(tl.int32)
        v_code = (vlo | vhi << 8) >> v_index_shift[None, :] & (1 << 2) - 1
        v_norm = tl.load(
            v_norm_ptr + batch * vn_s0 + kv_head * vn_s1 + offs_t,
            mask=valid_t,
            other=0.0,
        ).to(tl.float32)
        v_centroid = tl.load(
            v_codebook_ptr + v_code, mask=valid_d[None, :], other=0.0
        ).to(tl.float32)
        centroid_norm = tl.sqrt(
            tl.sum(tl.where(valid_d[None, :], v_centroid * v_centroid, 0.0), axis=1)
            + 1e-16
        )
        v_centroid = v_centroid / centroid_norm[:, None]
        v = v_centroid * v_norm[:, None]
        partial = tl.sum(probs[:, None] * v, axis=0)
        out_base = (pid_block * B * HQ + pid_bh) * D
        tl.store(numerator_ptr + out_base + offs_d, partial, mask=valid_d)
        stat_offset = pid_block * B * HQ + pid_bh
        tl.store(
            max_ptr + stat_offset, tl.where(usable_count > 0, block_max, -float("inf"))
        )
        tl.store(sum_ptr + stat_offset, block_sum)


def _prepare_additive_mask(
    attention_mask: Optional[torch.Tensor],
    positions: torch.Tensor,
    *,
    batch: int,
    total: int,
    sliding_window: Optional[int],
) -> torch.Tensor:
    n = int(positions.numel())
    device = positions.device
    additive = torch.zeros((batch, n), dtype=torch.float32, device=device)
    if attention_mask is not None:
        gather_pos = positions.long()
        if attention_mask.ndim == 2:
            keep = attention_mask.index_select(-1, gather_pos).bool()
            additive.masked_fill_(~keep, float("-inf"))
        elif attention_mask.ndim == 4:
            source = attention_mask[..., -1, :]
            if source.shape[1] != 1:
                raise ValueError(
                    "head-specific 4-D masks are unsupported by packed decode"
                )
            source = source[:, 0].float().index_select(-1, gather_pos)
            additive.copy_(source)
            additive.masked_fill_(source < -10000.0, float("-inf"))
        else:
            raise ValueError("packed decode expects a 2-D or 4-D attention mask")
    if sliding_window is not None:
        first = max(0, total - int(sliding_window))
        additive[:, positions < first] = float("-inf")
    return additive.contiguous()


def _merge_partials(
    numerators: torch.Tensor, maxima: torch.Tensor, sums: torch.Tensor
) -> torch.Tensor:
    global_max = maxima.amax(dim=0)
    finite = torch.isfinite(global_max)
    factors = torch.where(
        torch.isfinite(maxima) & finite.unsqueeze(0),
        torch.exp(maxima - global_max.unsqueeze(0)),
        torch.zeros_like(maxima),
    )
    denominator = (factors * sums).sum(dim=0)
    numerator = (factors.unsqueeze(-1) * numerators).sum(dim=0)
    return numerator / denominator.clamp_min(1e-20).unsqueeze(-1)


def _raw_residual_partial(
    query: torch.Tensor,
    cache,
    layer_idx: int,
    *,
    num_key_value_groups: int,
    attention_mask: Optional[torch.Tensor],
    sliding_window: Optional[int],
):
    raw_k, raw_v, positions = cache.current_token(layer_idx)
    if raw_k is None:
        return None
    batch, hq, dim = query.shape
    positions = positions.to(device=query.device, dtype=torch.int32)
    key = raw_k.repeat_interleave(num_key_value_groups, dim=1).float()
    value = raw_v.repeat_interleave(num_key_value_groups, dim=1).float()
    logits = torch.einsum("bhd,bhtd->bht", query.float(), key) / math.sqrt(dim)
    additive = _prepare_additive_mask(
        attention_mask,
        positions,
        batch=batch,
        total=cache.get_seq_length(layer_idx),
        sliding_window=sliding_window,
    )
    logits = logits + additive[:, None, :]
    usable = torch.isfinite(logits)
    maximum = logits.amax(dim=-1)
    safe_max = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    probs = torch.where(
        usable, torch.exp(logits - safe_max.unsqueeze(-1)), torch.zeros_like(logits)
    )
    numerator = torch.einsum("bht,bhtd->bhd", probs, value)
    return (numerator, maximum, probs.sum(dim=-1))


def packed_decode_attention(
    query_states,
    cache,
    layer_idx,
    *,
    num_key_value_groups,
    attention_mask=None,
    sliding_window=None,
):
    if triton is None or not query_states.is_cuda:
        raise RuntimeError("Packed attention requires CUDA Triton")
    if query_states.ndim != 4 or query_states.shape[-2] != 1:
        raise ValueError("Packed attention requires one query token")
    k_layer = cache._packed["k"][layer_idx]
    v_layer = cache._packed["v"][layer_idx]
    if k_layer.groups.keys() != v_layer.groups.keys():
        raise RuntimeError("K/V modality groups differ")
    batch, hq, _, dim = query_states.shape
    hkv = k_layer.heads
    if hq != hkv * num_key_value_groups:
        raise ValueError("GQA head count mismatch")
    q = query_states[:, :, 0].float()
    q_rot = q.contiguous() # native post-MRoPE query; K rotates after vector lookup
    block_t, block_d = (64, triton.next_power_of_2(dim))
    total = cache.get_seq_length(layer_idx)
    numerators, maxima, sums = ([], [], [])
    scratch_bytes = 0
    for mod, k_group in k_layer.groups.items():
        v_group = v_layer.groups[mod]
        n = k_group.length
        if n != v_group.length:
            raise RuntimeError("K/V group lengths differ")
        if not n:
            continue
        local = k_group.codec == CODEC_LOCAL_LM
        key = (str(q.device), dim, local)
        if key not in cache._packed_codebooks:
            centroids = lloyd_max(dim)[0].to(q.device)
            levels = (
                (centroids - centroids[0]) / (centroids[-1] - centroids[0])
                if local
                else centroids
            )
            cache._packed_codebooks[key] = (
                levels.view(1, -1).expand(hkv, -1).contiguous(),
                centroids,
            )
        _, v_codebook = cache._packed_codebooks[key]
        k_codebook = cache._packed_codebooks[("preR", layer_idx)]
        k_minimum = k_group.local_min if local else k_group.index
        k_scale = k_group.local_span if local else k_group.index
        k_min_mxscale = k_group.local_min_scale if local else k_group.index
        k_span_mxscale = k_group.local_span_scale if local else k_group.index
        k_norm, v_norm = (k_group.norm, v_group.norm)
        mask = _prepare_additive_mask(
            attention_mask,
            k_group.positions[:n],
            batch=batch,
            total=total,
            sliding_window=sliding_window,
        )
        n_blocks = triton.cdiv(n, block_t)
        partial = torch.empty(
            n_blocks, batch, hq, dim, dtype=torch.float32, device=q.device
        )
        part_max = torch.empty(
            n_blocks, batch, hq, dtype=torch.float32, device=q.device
        )
        part_sum = torch.empty_like(part_max)
        grid = (batch * hq, n_blocks)
        _packed_partial_attention_kernel[grid](
            q_rot,
            k_group.index,
            k_minimum,
            k_norm,
            k_scale,
            k_min_mxscale,
            k_span_mxscale,
            v_group.index,
            v_norm,
            k_codebook,
            k_group.positions,
            cache.preR_coordinates,
            cache.preR_cos,
            cache.preR_sin,
            v_codebook,
            mask,
            partial,
            part_max,
            part_sum,
            q_rot.stride(0),
            q_rot.stride(1),
            k_group.index.stride(0),
            k_group.index.stride(1),
            k_group.index.stride(2),
            k_minimum.stride(0),
            k_minimum.stride(1),
            k_minimum.stride(2),
            k_min_mxscale.stride(0),
            k_min_mxscale.stride(1),
            k_min_mxscale.stride(2),
            k_span_mxscale.stride(0),
            k_span_mxscale.stride(1),
            k_span_mxscale.stride(2),
            k_norm.stride(0),
            k_norm.stride(1),
            v_group.index.stride(0),
            v_group.index.stride(1),
            v_group.index.stride(2),
            v_norm.stride(0),
            v_norm.stride(1),
            mask.stride(0),
            k_codebook.stride(0),
            cache.preR_coordinates.stride(0),
            B=batch,
            HQ=hq,
            GROUPS=num_key_value_groups,
            N_TOKENS=n,
            D=dim,
            K_INDEX_WIDTH=k_group.index_width,
            V_INDEX_WIDTH=v_group.index_width,
            K_LOCAL_LM=k_group.codec == CODEC_LOCAL_LM,
            BLOCK_T=block_t,
            BLOCK_D=block_d,
            INV_SQRT_D=1.0 / math.sqrt(dim),
            num_warps=4,
        )
        rotation_v = cache._rot(layer_idx, mod, hkv, dim, q.device, which="v")
        partial = torch.einsum(
            "nbhgc,hcd->nbhgd",
            partial.reshape(n_blocks, batch, hkv, num_key_value_groups, dim),
            rotation_v,
        ).reshape(n_blocks, batch, hq, dim)
        numerators.append(partial)
        maxima.append(part_max)
        sums.append(part_sum)
        scratch_bytes += sum(
            (
                t.numel() * t.element_size()
                for t in (q_rot, mask, partial, part_max, part_sum)
            )
        )
    raw = _raw_residual_partial(
        q,
        cache,
        layer_idx,
        num_key_value_groups=num_key_value_groups,
        attention_mask=attention_mask,
        sliding_window=sliding_window,
    )
    if raw is not None:
        for dest, tensor in zip((numerators, maxima, sums), raw):
            dest.append(tensor.unsqueeze(0))
            scratch_bytes += tensor.numel() * tensor.element_size()
    if not numerators:
        raise RuntimeError("Cannot attend to an empty cache")
    output = _merge_partials(torch.cat(numerators), torch.cat(maxima), torch.cat(sums))
    cache._packed_fused_decode = True
    cache._packed_fused_scratch_bytes = max(
        cache._packed_fused_scratch_bytes, scratch_bytes
    )
    return output.unsqueeze(2).to(query_states.dtype)
