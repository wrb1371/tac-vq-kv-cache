"""
CoRDS KV-cache compression: coreset-based representative and diverse selection.

The public entry point is `process_kv_cache(...)`, dispatched by `--compression_method`:
    cords   : this work (coreset selection with orthogonal-bonus D^2 seeding)
    uniform : uniform-stride frame selection (baseline)
    swa     : sliding window attention (baseline)
"""

import math
import warnings
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Set, Tuple

_DIRECT_LEGACY_WARNINGS_EMITTED: Set[str] = set()


def _emit_direct_legacy_warning(legacy_key: str, message: str) -> None:
    if legacy_key in _DIRECT_LEGACY_WARNINGS_EMITTED:
        return
    _DIRECT_LEGACY_WARNINGS_EMITTED.add(legacy_key)
    warnings.warn(message, UserWarning, stacklevel=3)


def _is_direct_method(method: str) -> bool:
    return str(method).startswith("legacy-coreset-direct")


def _resolve_direct_params(
    method: str,
    coreset_feature_mode: str,
    coreset_recent_aware: bool,
    coreset_recent_summary_k: int,
    coreset_direct_trim_mode: str,
) -> Dict[str, Any]:
    """
    Map compression_method to effective direct-coreset parameters.
    Algorithm unchanged; this is naming/transparency only.
    Returns keys: selector_mode, feature_mode, recent_aware, recent_summary_k,
    direct_trim_mode, is_legacy, recommended_canonical (str or None).
    """
    m = str(method)
    summary_k = int(coreset_recent_summary_k)
    trim_mode = str(coreset_direct_trim_mode)
    feat = str(coreset_feature_mode)
    ra = bool(coreset_recent_aware)

    def _ret(
        selector_mode: str,
        feature_mode: str,
        recent_aware: bool,
        *,
        is_legacy: bool = False,
        recommended_canonical: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "selector_mode": selector_mode,
            "feature_mode": feature_mode,
            "recent_aware": recent_aware,
            "recent_summary_k": summary_k,
            "direct_trim_mode": trim_mode,
            "is_legacy": is_legacy,
            "recommended_canonical": recommended_canonical,
        }

    # --- Canonical (no warning) ---
    if m == "legacy-coreset-direct-kmeans-kv":
        return _ret("kmeans", "kv_joint", ra)
    if m == "legacy-coreset-direct-d2-kv":
        return _ret("d2", "kv_joint", ra)
    if m == "legacy-coreset-direct-greedy-kv":
        return _ret("greedy", "kv_joint", ra)
    if m == "legacy-coreset-direct-d2-recent":
        return _ret("d2", feat, True)
    if m == "legacy-coreset-direct-d2-k":
        return _ret("d2", "k", ra)
    if m == "legacy-coreset-direct-d2-v":
        return _ret("d2", "v", ra)

    # --- Legacy aliases (warn once) ---
    if m == "legacy-coreset-direct-kv":
        _emit_direct_legacy_warning(
            "direct-kv",
            "compression_method 'legacy-coreset-direct-kv' uses k-means (Lloyd) coreset on "
            "kv_joint, not D² sampling. For explicit naming use "
            "'legacy-coreset-direct-kmeans-kv'. "
            "'--coreset_feature_mode' is ignored for this legacy alias.",
        )
        return _ret("kmeans", "kv_joint", ra, is_legacy=True, recommended_canonical="legacy-coreset-direct-kmeans-kv")

    if m == "legacy-coreset-direct":
        _emit_direct_legacy_warning(
            "direct",
            "compression_method 'legacy-coreset-direct' uses selector_mode='d2' with "
            "--coreset_feature_mode (not k-means). Prefer 'legacy-coreset-direct-d2-kv', "
            "'-d2-k', or '-d2-v' so selector+feature are unambiguous.",
        )
        return _ret("d2", feat, ra, is_legacy=True, recommended_canonical="legacy-coreset-direct-d2-<feature>")

    if m == "legacy-coreset-direct-d2":
        _emit_direct_legacy_warning(
            "direct-d2",
            "compression_method 'legacy-coreset-direct-d2' couples D² to --coreset_feature_mode. "
            "Prefer 'legacy-coreset-direct-d2-kv', '-d2-k', or '-d2-v'.",
        )
        return _ret("d2", feat, ra, is_legacy=True, recommended_canonical="legacy-coreset-direct-d2-<feature>")

    if m == "legacy-coreset-direct-greedy":
        _emit_direct_legacy_warning(
            "direct-greedy",
            "compression_method 'legacy-coreset-direct-greedy' couples greedy selector to "
            "--coreset_feature_mode. For kv_joint use 'legacy-coreset-direct-greedy-kv'.",
        )
        return _ret("greedy", feat, ra, is_legacy=True, recommended_canonical="legacy-coreset-direct-greedy-kv")

    if m == "legacy-coreset-direct-recent":
        _emit_direct_legacy_warning(
            "direct-recent",
            "compression_method 'legacy-coreset-direct-recent' uses D² with recent_aware=True "
            "and --coreset_feature_mode. Prefer 'legacy-coreset-direct-d2-recent'.",
        )
        return _ret("d2", feat, True, is_legacy=True, recommended_canonical="legacy-coreset-direct-d2-recent")

    raise ValueError(f"Unknown legacy-coreset-direct* method: {m}")


def resolve_compression_layer_indices(
    num_layers: int,
    mode: str = "all",
    custom_indices: Optional[Any] = None,
) -> List[int]:
    """
    Resolve active transformer layers for KV compression.
    Supports mode in: all, last, top25, top50, bottom25, even, odd, custom.
    """
    n = int(num_layers)
    if n <= 0:
        return []
    m = str(mode).strip().lower()

    def _top_k(frac: float) -> List[int]:
        k = max(1, int(math.ceil(frac * n)))
        start = max(0, n - k)
        return list(range(start, n))

    def _bottom_k(frac: float) -> List[int]:
        k = max(1, int(math.ceil(frac * n)))
        return list(range(0, min(k, n)))

    if m == "all":
        out = list(range(n))
    elif m == "last":
        out = [n - 1]
    elif m == "top25":
        out = _top_k(0.25)
    elif m == "top50":
        out = _top_k(0.50)
    elif m == "bottom25":
        out = _bottom_k(0.25)
    elif m == "even":
        out = [i for i in range(n) if i % 2 == 0]
    elif m == "odd":
        out = [i for i in range(n) if i % 2 == 1]
    elif m == "custom":
        raw = custom_indices
        if raw is None:
            out = []
        elif isinstance(raw, str):
            toks = [t.strip() for t in raw.split(",") if t.strip()]
            out = [int(t) for t in toks]
        else:
            out = [int(x) for x in raw]
        out = sorted(set(i for i in out if 0 <= i < n))
    else:
        raise ValueError(f"Unknown compression_layer_mode: {mode!r}")
    return out


def build_resolved_kv_run_config(
    compression_method: str,
    block_size: int,
    compress_frame_num: int,
    max_frames_num: int,
    coreset_feature_mode: str,
    coreset_recent_aware: bool,
    coreset_recent_summary_k: int,
    coreset_direct_trim_mode: str,
    coreset_bicriteria_alpha: float = 0.5,
    coreset_bicriteria_mode: str = "mixed",
    probe_mode: str = "centroid",
    probe_feature_mode: str = "kv_joint",
    num_probe_vectors: int = 8,
    coreset_orthbicrit_alpha: float = 0.25,
    coreset_orthbicrit_beta: float = 0.0,
    coreset_orthbonus_lambda: float = 1.0,
    coreset_orthbonus_eta: float = 0.25,
    coreset_orthbonus_mode: str = "maxcos",
    coreset_orthbonus_beta: float = 0.0,
    coreset_orthbonus_prefilter_ratio: float = 1.0,
    coreset_orthbonus_score_norm: str = "none",
    compression_layer_mode: str = "all",
    compression_layer_indices: Optional[Any] = None,
    num_layers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Flat dict for resolved_config.json (effective settings actually used for KV compression).
    """
    out: Dict[str, Any] = {
        "compression_method": str(compression_method),
        "block_size": int(block_size),
        "compress_frame_num": int(compress_frame_num),
        "max_frames_num": int(max_frames_num),
        "selector_mode": None,
        "feature_mode": None,
        "recent_aware": bool(coreset_recent_aware),
        "recent_summary_k": int(coreset_recent_summary_k),
        "direct_trim_mode": str(coreset_direct_trim_mode),
        "bicriteria_alpha": float(coreset_bicriteria_alpha),
        "bicriteria_mode": str(coreset_bicriteria_mode),
        "probe_mode": str(probe_mode),
        "probe_feature_mode": str(probe_feature_mode),
        "num_probe_vectors": int(num_probe_vectors),
        "orthbicrit_alpha": float(coreset_orthbicrit_alpha),
        "orthbicrit_beta": float(coreset_orthbicrit_beta),
        "coreset_orthbonus_lambda": float(coreset_orthbonus_lambda),
        "coreset_orthbonus_eta": float(coreset_orthbonus_eta),
        "coreset_orthbonus_mode": str(coreset_orthbonus_mode),
        "coreset_orthbonus_beta": float(coreset_orthbonus_beta),
        "coreset_orthbonus_prefilter_ratio": float(coreset_orthbonus_prefilter_ratio),
        "coreset_orthbonus_score_norm": str(coreset_orthbonus_score_norm),
        "compression_layer_mode": str(compression_layer_mode),
        "compression_layer_indices_resolved": None,
        "num_active_layers": None,
        "feature_level": None,
    }
    if num_layers is not None:
        resolved_layers = resolve_compression_layer_indices(
            int(num_layers),
            str(compression_layer_mode),
            compression_layer_indices,
        )
        out["compression_layer_indices_resolved"] = [int(i) for i in resolved_layers]
        out["num_active_layers"] = int(len(resolved_layers))

    if _is_direct_method(compression_method):
        r = _resolve_direct_params(
            compression_method,
            coreset_feature_mode,
            coreset_recent_aware,
            coreset_recent_summary_k,
            coreset_direct_trim_mode,
        )
        out["selector_mode"] = r["selector_mode"]
        out["feature_mode"] = r["feature_mode"]
        out["recent_aware"] = r["recent_aware"]
        out["recent_summary_k"] = r["recent_summary_k"]
        out["direct_trim_mode"] = r["direct_trim_mode"]
        out["legacy_compression_method_alias"] = bool(r["is_legacy"])
        if r["is_legacy"]:
            out["recommended_canonical_method"] = r["recommended_canonical"]
    elif _is_bicriteria_method(compression_method):
        cm = str(compression_method)
        if cm == "legacy-coreset-bicriteria-d2":
            sel, ra = "d2", bool(coreset_recent_aware)
        elif cm == "legacy-coreset-bicriteria-kmeans":
            sel, ra = "kmeans", bool(coreset_recent_aware)
        elif cm == "legacy-coreset-bicriteria-greedy":
            sel, ra = "greedy", bool(coreset_recent_aware)
        elif cm == "legacy-coreset-bicriteria-recent":
            sel, ra = "d2", True
        else:
            raise ValueError(f"Unhandled bicriteria method in resolved config: {cm}")
        out["selector_mode"] = sel
        out["recent_aware"] = ra
        out["recent_summary_k"] = int(coreset_recent_summary_k)
        out["bicriteria_alpha"] = float(coreset_bicriteria_alpha)
        out["bicriteria_mode"] = str(coreset_bicriteria_mode)
    elif _is_probe_method(compression_method):
        cm = str(compression_method)
        ra = True if cm == "legacy-coreset-probe-recent" else bool(coreset_recent_aware)
        out["recent_aware"] = ra
        out["recent_summary_k"] = int(coreset_recent_summary_k)
        out["probe_mode"] = str(probe_mode)
        out["probe_feature_mode"] = str(probe_feature_mode)
        out["num_probe_vectors"] = int(num_probe_vectors)
    elif _is_orthbicrit_method(compression_method):
        cm = str(compression_method)
        ra = True if cm == "legacy-coreset-orthbicrit-recent" else bool(coreset_recent_aware)
        out["recent_aware"] = ra
        out["recent_summary_k"] = int(coreset_recent_summary_k)
        out["orthbicrit_alpha"] = float(coreset_orthbicrit_alpha)
        out["orthbicrit_beta"] = float(coreset_orthbicrit_beta)
        if cm == "legacy-coreset-orthbicrit-frame":
            out["feature_level"] = "frame"
        else:
            out["feature_level"] = None
    elif is_cords_method(compression_method):
        cm = str(compression_method)
        out["selector_mode"] = "d2_orthbonus"
        out["recent_aware"] = bool(coreset_recent_aware)
        out["recent_summary_k"] = int(coreset_recent_summary_k)
        out["coreset_bicriteria_alpha"] = float(coreset_bicriteria_alpha)
        out["coreset_orthbonus_lambda"] = float(coreset_orthbonus_lambda)
        out["coreset_orthbonus_eta"] = float(coreset_orthbonus_eta)
        out["coreset_orthbonus_mode"] = str(coreset_orthbonus_mode)
        out["coreset_orthbonus_beta"] = float(coreset_orthbonus_beta)
        out["coreset_orthbonus_prefilter_ratio"] = float(coreset_orthbonus_prefilter_ratio)
        out["coreset_orthbonus_score_norm"] = str(coreset_orthbonus_score_norm)
        out["orthbonus_prefilter_active"] = bool(
            float(coreset_orthbonus_prefilter_ratio) < 1.0 - 1e-12
        )
        out["feature_level"] = None
    return out


def select_coreset_indices_kmeans(
    x: torch.Tensor, budget: int, num_iters: int = 4
) -> torch.Tensor:
    """
    Lightweight k-means-style coreset: pick `budget` real row indices from x.
    x: [num_tokens, head_dim]. Returns sorted unique LongTensor indices in [0, num_tokens).
    """
    device = x.device
    N = int(x.shape[0])
    budget = int(budget)
    if budget <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    if N == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    if budget >= N:
        return torch.arange(N, device=device, dtype=torch.long)

    K = min(budget, N)
    xf = x.float()

    # Deterministic centroid init: evenly spaced token indices
    if K == 1:
        init_list = [0]
    else:
        pos = torch.linspace(0, N - 1, K, device=device).round().long().clamp(0, N - 1)
        init_list = []
        seen_i = set()
        for t in pos.tolist():
            if t not in seen_i:
                seen_i.add(int(t))
                init_list.append(int(t))
        cfill = 0
        while len(init_list) < K and cfill < N:
            if cfill not in seen_i:
                seen_i.add(cfill)
                init_list.append(cfill)
            cfill += 1
    init_idx = torch.tensor(init_list[:K], device=device, dtype=torch.long)
    centroids = xf[init_idx].clone()

    for _ in range(max(1, int(num_iters))):
        dist2 = (xf.unsqueeze(1) - centroids.unsqueeze(0)).pow(2).sum(dim=-1)
        assign = dist2.argmin(dim=1)
        new_rows = []
        for k in range(K):
            mask = assign == k
            if mask.any():
                new_rows.append(xf[mask].mean(dim=0))
            else:
                new_rows.append(centroids[k])
        centroids = torch.stack(new_rows, dim=0)

    dist2c = (xf.unsqueeze(1) - centroids.unsqueeze(0)).pow(2).sum(dim=-1)
    nearest_per_centroid = dist2c.argmin(dim=0)

    picked: List[int] = []
    seen: set = set()
    for k in range(K):
        idx = int(nearest_per_centroid[k].item())
        if idx not in seen:
            seen.add(idx)
            picked.append(idx)

    def _farthest_fill() -> Optional[int]:
        if len(picked) >= budget:
            return None
        mask_unpicked = torch.tensor([i not in seen for i in range(N)], device=device)
        if not bool(mask_unpicked.any()):
            return None
        rem = mask_unpicked.nonzero(as_tuple=False).squeeze(-1)
        if rem.numel() == 0:
            return None
        S = torch.tensor(picked, device=device, dtype=torch.long)
        dmin = (xf[rem].unsqueeze(1) - xf[S].unsqueeze(0)).pow(2).sum(dim=-1).min(dim=1).values
        j = int(rem[int(torch.argmax(dmin).item())].item())
        return j

    while len(picked) < budget:
        j = _farthest_fill()
        if j is None:
            break
        picked.append(j)
        seen.add(j)

    c = 0
    while len(picked) < budget and c < N:
        if c not in seen:
            picked.append(c)
            seen.add(c)
        c += 1

    picked = sorted(picked[:budget])
    return torch.tensor(picked, device=device, dtype=torch.long)


def select_coreset_indices_d2(x: torch.Tensor, budget: int) -> torch.Tensor:
    """
    Deterministic D^2-style farthest sampling: pick `budget` real row indices from x.
    x: [num_tokens, head_dim]. Returns sorted unique LongTensor indices.
    """
    device = x.device
    N = int(x.shape[0])
    budget = int(budget)
    if budget <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    if N == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    if budget >= N:
        return torch.arange(N, device=device, dtype=torch.long)

    xf = x.float()
    centroid = xf.mean(dim=0, keepdim=True)
    first = int(torch.argmax((xf - centroid).pow(2).sum(dim=-1)).item())
    selected = [first]
    selected_set = {first}

    while len(selected) < budget:
        S = torch.tensor(selected, device=device, dtype=torch.long)
        d2 = (xf.unsqueeze(1) - xf[S].unsqueeze(0)).pow(2).sum(dim=-1).min(dim=1).values
        d2[S] = -1.0
        j = int(torch.argmax(d2).item())
        if d2[j] < 0:
            break
        if j not in selected_set:
            selected.append(j)
            selected_set.add(j)

    c = 0
    while len(selected) < budget and c < N:
        if c not in selected_set:
            selected.append(c)
            selected_set.add(c)
        c += 1

    return torch.tensor(sorted(selected[:budget]), device=device, dtype=torch.long)


def _baseline_select_indices(
    key_states_to_compress: torch.Tensor,
    value_states_to_compress: torch.Tensor,
    num_heads: int,
    token_per_frame: int,
    current_frame_num: int,
    compress_frame_num: int,
    tar_ratio: float,
    query_ratio: float,
    avg_pooling_nd: int,
    attn_pool_size: int,
    model,
) -> torch.Tensor:
    """Original this work token selection (unchanged semantics)."""
    attn_budget = round((1 - tar_ratio) * int(round(compress_frame_num * token_per_frame)))
    tar_indices = None

    if tar_ratio > 0:
        tar_budget = int(round(compress_frame_num * token_per_frame)) - attn_budget
        query_frame = int(current_frame_num * query_ratio)
        query_length = int(query_frame * token_per_frame)
        total_length = key_states_to_compress.shape[2]

        query_frame_emb = key_states_to_compress[:, :, -query_length:, :]
        query_emb_norm = query_frame_emb / (query_frame_emb.norm(dim=-1, keepdim=True) + 1e-9)

        key_frame_emb = key_states_to_compress[:, :, :-query_length, :]
        key_emb_norm = key_frame_emb / (key_frame_emb.norm(dim=-1, keepdim=True) + 1e-9)

        query_emb_norm_reshaped = query_emb_norm.reshape(
            query_emb_norm.shape[0],
            query_emb_norm.shape[1],
            query_frame,
            token_per_frame,
            query_emb_norm.shape[-1],
        )
        key_emb_norm_reshaped = key_emb_norm.reshape(
            key_emb_norm.shape[0],
            key_emb_norm.shape[1],
            -1,
            token_per_frame,
            key_emb_norm.shape[-1],
        )

        key_score = -(
            query_emb_norm_reshaped.unsqueeze(3) * key_emb_norm_reshaped.unsqueeze(2)
        ).sum(dim=-1).mean(dim=2).reshape(
            query_emb_norm.shape[0], query_emb_norm.shape[1], -1
        )

        recent_indices = torch.arange(
            total_length - query_length, total_length, device=key_states_to_compress.device
        ).unsqueeze(0).expand(num_heads, -1)

        selected_indices = key_score.topk(tar_budget - query_length, dim=-1).indices.squeeze(0)
        tar_indices_combined = torch.cat([selected_indices, recent_indices], dim=-1)
        tar_indices, _ = tar_indices_combined.sort(dim=-1)

    val_norm_score = value_states_to_compress.norm(dim=-1)[0, :, :]

    if avg_pooling_nd > 0:
        pool_size = attn_pool_size
        head_num, _ = val_norm_score.shape
        frame_num = val_norm_score.shape[-1] // token_per_frame

        if avg_pooling_nd == 1:
            avg_pool = torch.nn.AvgPool1d(
                kernel_size=pool_size, stride=1, padding=pool_size // 2
            )
        elif avg_pooling_nd == 2:
            avg_pool = torch.nn.AvgPool2d(
                kernel_size=(pool_size, pool_size),
                stride=(1, 1),
                padding=(pool_size // 2, pool_size // 2),
            )
        elif avg_pooling_nd == 3:
            avg_pool = torch.nn.AvgPool3d(
                kernel_size=(pool_size, pool_size, pool_size),
                stride=(1, 1, 1),
                padding=(pool_size // 2, pool_size // 2, pool_size // 2),
            )
        else:
            raise NotImplementedError("We support 1D - 3D Pooling")

        patch_size = (
            model.config.height_width if hasattr(model.config, "height_width") else 14
        )

        if avg_pooling_nd > 1:
            original_numel = val_norm_score.numel()
            val_norm_reshaped = val_norm_score.reshape(head_num, frame_num, patch_size, -1)

            T = val_norm_reshaped.shape[-3]
            H = val_norm_reshaped.shape[-2]
            W = val_norm_reshaped.shape[-1]

            pool_size_eff = min(pool_size, T, H, W)

            if pool_size_eff % 2 == 0:
                pool_size_eff -= 1

            if pool_size_eff < 2:
                pass
            else:
                if pool_size_eff != pool_size:
                    pool_size = pool_size_eff
                    if avg_pooling_nd == 2:
                        avg_pool = torch.nn.AvgPool2d(
                            kernel_size=(pool_size, pool_size),
                            stride=(1, 1),
                            padding=(pool_size // 2, pool_size // 2),
                        )
                    elif avg_pooling_nd == 3:
                        avg_pool = torch.nn.AvgPool3d(
                            kernel_size=(pool_size, pool_size, pool_size),
                            stride=(1, 1, 1),
                            padding=(pool_size // 2, pool_size // 2, pool_size // 2),
                        )

                val_norm_pooled = avg_pool(val_norm_reshaped)
                val_norm_score = val_norm_pooled.reshape(head_num, -1)
                assert original_numel == val_norm_score.numel()
        else:
            val_norm_score = avg_pool(val_norm_score)

    if tar_indices is not None:
        head_rng = torch.arange(num_heads, device=key_states_to_compress.device).unsqueeze(1)
        head_rng = head_rng.expand(-1, tar_indices.size(1))
        val_norm_score = val_norm_score.clone()
        val_norm_score[head_rng, tar_indices] = val_norm_score.max() + 1

    total_tokens_to_keep = int(round(compress_frame_num * token_per_frame))
    all_indices = val_norm_score.topk(total_tokens_to_keep, dim=-1).indices
    all_indices, _ = all_indices.sort(dim=-1)
    return all_indices


def _minmax01_scores(t: torch.Tensor) -> torch.Tensor:
    """Map 1D scores to [0, 1] for fusion; near-constant -> 0.5."""
    lo, hi = t.min(), t.max()
    span = hi - lo
    if float(span) <= 1e-8:
        return torch.full_like(t, 0.5)
    return (t - lo) / span


def _coreset_select_indices(
    key_states_to_compress: torch.Tensor,
    value_states_to_compress: torch.Tensor,
    num_heads: int,
    token_per_frame: int,
    current_frame_num: int,
    compress_frame_num: int,
    tar_ratio: float,
    query_ratio: float,
    avg_pooling_nd: int,
    attn_pool_size: int,
    model,
    coreset_num_iters: int,
    coreset_on_value: bool,
    coreset_candidate_multiplier: float,
    selector_mode: str = "kmeans",
    use_value_rerank: bool = True,
    frame_feature: str = "k",
    rerank_mode: str = "value",
    bicrit_alpha: float = 0.5,
) -> torch.Tensor:
    """
    this work structure: same pooled V-norm scores; protected recent tokens unchanged;
    older tokens from K-means coreset pool, then V-norm top-k (optional V coreset on candidates).
    """
    device = key_states_to_compress.device
    total_length = key_states_to_compress.shape[2]
    total_tokens_to_keep = int(round(compress_frame_num * token_per_frame))

    query_length = 0
    if tar_ratio > 0:
        query_frame = int(current_frame_num * query_ratio)
        query_length = int(query_frame * token_per_frame)
    query_length = min(query_length, total_length)
    split = total_length - query_length
    if split < 0:
        split = 0
        query_length = total_length

    if query_length > 0:
        protected = torch.arange(split, total_length, device=device, dtype=torch.long)
    else:
        protected = torch.zeros(0, dtype=torch.long, device=device)

    old_budget = max(0, total_tokens_to_keep - int(protected.numel()))
    if old_budget > split:
        raise RuntimeError("legacy-coreset: old_budget exceeds non-protected length")

    # Same val_norm_score as legacy-baseline (pooling only; no tar boost — we force-include protected)
    val_norm_score = value_states_to_compress.norm(dim=-1)[0, :, :].clone()

    if avg_pooling_nd > 0:
        pool_size = attn_pool_size
        head_num, _ = val_norm_score.shape
        frame_num = val_norm_score.shape[-1] // token_per_frame

        if avg_pooling_nd == 1:
            avg_pool = torch.nn.AvgPool1d(
                kernel_size=pool_size, stride=1, padding=pool_size // 2
            )
        elif avg_pooling_nd == 2:
            avg_pool = torch.nn.AvgPool2d(
                kernel_size=(pool_size, pool_size),
                stride=(1, 1),
                padding=(pool_size // 2, pool_size // 2),
            )
        elif avg_pooling_nd == 3:
            avg_pool = torch.nn.AvgPool3d(
                kernel_size=(pool_size, pool_size, pool_size),
                stride=(1, 1, 1),
                padding=(pool_size // 2, pool_size // 2, pool_size // 2),
            )
        else:
            raise NotImplementedError("We support 1D - 3D Pooling")

        patch_size = (
            model.config.height_width if hasattr(model.config, "height_width") else 14
        )

        if avg_pooling_nd > 1:
            original_numel = val_norm_score.numel()
            val_norm_reshaped = val_norm_score.reshape(head_num, frame_num, patch_size, -1)

            T = val_norm_reshaped.shape[-3]
            H = val_norm_reshaped.shape[-2]
            W = val_norm_reshaped.shape[-1]

            pool_size_eff = min(pool_size, T, H, W)

            if pool_size_eff % 2 == 0:
                pool_size_eff -= 1

            if pool_size_eff < 2:
                pass
            else:
                if pool_size_eff != pool_size:
                    pool_size = pool_size_eff
                    if avg_pooling_nd == 2:
                        avg_pool = torch.nn.AvgPool2d(
                            kernel_size=(pool_size, pool_size),
                            stride=(1, 1),
                            padding=(pool_size // 2, pool_size // 2),
                        )
                    elif avg_pooling_nd == 3:
                        avg_pool = torch.nn.AvgPool3d(
                            kernel_size=(pool_size, pool_size, pool_size),
                            stride=(1, 1, 1),
                            padding=(pool_size // 2, pool_size // 2, pool_size // 2),
                        )

                val_norm_pooled = avg_pool(val_norm_reshaped)
                val_norm_score = val_norm_pooled.reshape(head_num, -1)
                assert original_numel == val_norm_score.numel()
        else:
            val_norm_score = avg_pool(val_norm_score)

    mult = max(float(coreset_candidate_multiplier), 1.0)
    out_rows: List[torch.Tensor] = []

    for h in range(num_heads):
        if old_budget == 0:
            pn = int(protected.numel())
            if pn == total_tokens_to_keep:
                row = protected
            elif pn > total_tokens_to_keep:
                sc = val_norm_score[h, protected]
                row = protected[
                    torch.topk(sc, k=total_tokens_to_keep, largest=True).indices
                ].sort().values
            else:
                raise RuntimeError("legacy-coreset: protected shorter than budget")
            out_rows.append(row)
            continue
        if split <= 0:
            raise RuntimeError("no old tokens for coreset")

        x_old = key_states_to_compress[0, h, :split, :].float()
        v_old = value_states_to_compress[0, h, :split, :].float()
        N_old = int(x_old.shape[0])
        cand_budget = int(min(N_old, max(1, math.ceil(old_budget * mult))))

        # Runtime guard: select coreset candidates on old frames (not tokens), then
        # expand back to token indices. This keeps real tokens while avoiding NxK blowups.
        if token_per_frame > 1 and split % token_per_frame == 0:
            F_old = split // token_per_frame
            k_old_f = x_old.reshape(F_old, token_per_frame, -1).mean(dim=1)
            if frame_feature == "kv_joint":
                v_old_f = v_old.reshape(F_old, token_per_frame, -1).mean(dim=1)
                x_old_f = torch.cat([k_old_f, v_old_f], dim=-1)
            elif frame_feature == "k":
                x_old_f = k_old_f
            else:
                raise ValueError(f"Unknown frame_feature: {frame_feature}")
            cand_frame_budget = int(
                min(F_old, max(1, math.ceil(cand_budget / token_per_frame)))
            )
            if selector_mode == "kmeans":
                idx_f = select_coreset_indices_kmeans(
                    x_old_f, cand_frame_budget, coreset_num_iters
                )
            elif selector_mode == "d2":
                idx_f = select_coreset_indices_d2(x_old_f, cand_frame_budget)
            elif selector_mode == "greedy":
                x_old_f = x_old_f / x_old_f.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                idx_f = _greedy_linear_coreset_indices(x_old_f, cand_frame_budget)
            else:
                raise ValueError(f"Unknown selector_mode: {selector_mode}")
            flat: List[int] = []
            for f in idx_f.detach().cpu().tolist():
                base = int(f) * token_per_frame
                flat.extend(range(base, base + token_per_frame))
            abs_cand = torch.tensor(flat, device=device, dtype=torch.long)
            abs_cand = torch.unique(abs_cand, sorted=True)
        else:
            if frame_feature == "kv_joint":
                x_sel = torch.cat([x_old, v_old], dim=-1)
            else:
                x_sel = x_old
            if selector_mode == "kmeans":
                idx_rel = select_coreset_indices_kmeans(x_sel, cand_budget, coreset_num_iters)
            elif selector_mode == "d2":
                idx_rel = select_coreset_indices_d2(x_sel, cand_budget)
            elif selector_mode == "greedy":
                x_sel = x_sel / x_sel.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                idx_rel = _greedy_linear_coreset_indices(x_sel, cand_budget)
            else:
                raise ValueError(f"Unknown selector_mode: {selector_mode}")
            abs_cand = idx_rel.long()

        if coreset_on_value and abs_cand.numel() > 0:
            v_sub = value_states_to_compress[0, h, abs_cand, :].float()
            v_budget = int(
                min(
                    int(abs_cand.numel()),
                    max(old_budget, int(math.ceil(old_budget * mult))),
                )
            )
            v_budget = max(1, v_budget)
            sub_rel = select_coreset_indices_kmeans(v_sub, v_budget, coreset_num_iters)
            abs_cand = abs_cand[sub_rel.long()]

        if rerank_mode == "bicrit":
            v_sc = val_norm_score[h, abs_cand]
            k_sc = key_states_to_compress[0, h, abs_cand].float().norm(dim=-1)
            a = float(bicrit_alpha)
            a = min(1.0, max(0.0, a))
            scores = a * _minmax01_scores(v_sc) + (1.0 - a) * _minmax01_scores(k_sc)
            k_take = min(old_budget, int(abs_cand.numel()))
            top_rel = torch.topk(scores, k=k_take, largest=True).indices
            sel_old = abs_cand[top_rel].sort().values
        elif use_value_rerank:
            scores = val_norm_score[h, abs_cand]
            k_take = min(old_budget, int(abs_cand.numel()))
            top_rel = torch.topk(scores, k=k_take, largest=True).indices
            sel_old = abs_cand[top_rel].sort().values
        else:
            sel_old = abs_cand.sort().values[:old_budget]

        if int(sel_old.numel()) < old_budget:
            mask_old = torch.ones(split, dtype=torch.bool, device=device)
            mask_old[sel_old] = False
            rest = mask_old.nonzero(as_tuple=False).squeeze(-1)
            if rest.numel() > 0:
                srest = val_norm_score[h, rest]
                need = old_budget - int(sel_old.numel())
                extra = rest[torch.topk(srest, k=min(need, int(rest.numel())), largest=True).indices]
                sel_old = torch.unique(torch.cat([sel_old, extra]), sorted=True)

        if int(sel_old.numel()) < old_budget:
            raise RuntimeError("coreset could not fill old_budget")

        row = torch.unique(torch.cat([sel_old, protected]), sorted=True)
        if int(row.numel()) != total_tokens_to_keep:
            raise RuntimeError("coreset row length mismatch")
        out_rows.append(row)

    return torch.stack(out_rows, dim=0)


def _direct_coreset_pick_indices(
    X: torch.Tensor,
    budget: int,
    selector_mode: str,
    coreset_num_iters: int,
) -> torch.Tensor:
    """Pick exactly `budget` row indices from X [N,D] using selector_mode (no shortlist)."""
    if budget <= 0:
        return torch.zeros(0, dtype=torch.long, device=X.device)
    N = int(X.shape[0])
    if N == 0:
        return torch.zeros(0, dtype=torch.long, device=X.device)
    b = min(budget, N)
    if selector_mode == "kmeans":
        return select_coreset_indices_kmeans(X, b, coreset_num_iters)
    if selector_mode == "d2":
        return select_coreset_indices_d2(X, b)
    if selector_mode == "greedy":
        Xn = X.float()
        Xn = Xn / Xn.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return _greedy_linear_coreset_indices(Xn, b)
    raise ValueError(f"Unknown selector_mode for direct coreset: {selector_mode}")


def _trim_expanded_token_indices_farthest(
    token_feats: torch.Tensor,
    abs_token_idx: torch.Tensor,
    budget: int,
) -> torch.Tensor:
    """token_feats: [M, D], abs_token_idx: [M] — keep `budget` rows via D^2 on feats."""
    device = token_feats.device
    M = int(abs_token_idx.numel())
    if M <= budget:
        return abs_token_idx.sort().values
    sub = select_coreset_indices_d2(token_feats.float(), budget)
    return abs_token_idx[sub.long()].sort().values


def _trim_expanded_token_indices_first(
    abs_token_idx: torch.Tensor,
    budget: int,
) -> torch.Tensor:
    return abs_token_idx.sort().values[:budget]


def _build_direct_token_features(
    x_old: torch.Tensor,
    v_old: torch.Tensor,
    feature_mode: str,
    recent_aware: bool,
    k_mean_prot: Optional[torch.Tensor],
    v_mean_prot: Optional[torch.Tensor],
) -> torch.Tensor:
    """Per old-token feature rows [N_old, *]."""
    if feature_mode == "k":
        f = x_old
        if recent_aware and k_mean_prot is not None:
            f = f - k_mean_prot.unsqueeze(0)
        return f
    if feature_mode == "v":
        f = v_old
        if recent_aware and v_mean_prot is not None:
            f = f - v_mean_prot.unsqueeze(0)
        return f
    if feature_mode == "kv_joint":
        if recent_aware and k_mean_prot is not None and v_mean_prot is not None:
            f = torch.cat(
                [x_old - k_mean_prot.unsqueeze(0), v_old - v_mean_prot.unsqueeze(0)], dim=-1
            )
        else:
            f = torch.cat([x_old, v_old], dim=-1)
        return f
    if feature_mode == "k_recent_residual":
        if k_mean_prot is None:
            raise RuntimeError("k_recent_residual requires non-empty protected summary")
        return x_old - k_mean_prot.unsqueeze(0)
    if feature_mode == "kv_recent_residual":
        if k_mean_prot is None or v_mean_prot is None:
            raise RuntimeError("kv_recent_residual requires non-empty protected summary")
        return torch.cat(
            [x_old - k_mean_prot.unsqueeze(0), v_old - v_mean_prot.unsqueeze(0)], dim=-1
        )
    raise ValueError(f"Unknown feature_mode: {feature_mode}")


_BICRITERIA_METHODS = frozenset(
    {
        "legacy-coreset-bicriteria-d2",
        "legacy-coreset-bicriteria-kmeans",
        "legacy-coreset-bicriteria-greedy",
        "legacy-coreset-bicriteria-recent",
    }
)


def _is_bicriteria_method(method: str) -> bool:
    return str(method) in _BICRITERIA_METHODS


def is_cords_method(method: str) -> bool:
    return str(method) == "cords"


def _bicriteria_apply_residual(
    K: torch.Tensor,
    V: torch.Tensor,
    k_mean_prot: Optional[torch.Tensor],
    v_mean_prot: Optional[torch.Tensor],
    recent_aware: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not recent_aware or k_mean_prot is None or v_mean_prot is None:
        return K, V
    return K - k_mean_prot.unsqueeze(0), V - v_mean_prot.unsqueeze(0)


def select_bicriteria_farthest_mixed(
    K: torch.Tensor, V: torch.Tensor, budget: int, alpha: float
) -> torch.Tensor:
    """
    Farthest-point / D²-style subset: iteratively add i maximizing
    d_mix(i,S) = alpha * min_{j in S} ||K_i-K_j||^2 + (1-alpha) * min_{j in S} ||V_i-V_j||^2.
    Empty S uses distance to mean K/V.
    """
    device = K.device
    N = int(K.shape[0])
    b = min(int(budget), N)
    if b <= 0 or N == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    if b >= N:
        return torch.arange(N, device=device, dtype=torch.long)
    a = min(1.0, max(0.0, float(alpha)))
    Kf = K.float()
    Vf = V.float()
    k_cent = Kf.mean(0, keepdim=True)
    v_cent = Vf.mean(0, keepdim=True)
    selected: List[int] = []
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    for _ in range(b):
        if not selected:
            dK = (Kf - k_cent).pow(2).sum(dim=-1)
            dV = (Vf - v_cent).pow(2).sum(dim=-1)
        else:
            S = torch.tensor(selected, device=device, dtype=torch.long)
            dK = (Kf.unsqueeze(1) - Kf[S].unsqueeze(0)).pow(2).sum(dim=-1).min(dim=1).values
            dV = (Vf.unsqueeze(1) - Vf[S].unsqueeze(0)).pow(2).sum(dim=-1).min(dim=1).values
        dm = a * dK + (1.0 - a) * dV
        dm = dm.masked_fill(mask, -1.0)
        j = int(torch.argmax(dm).item())
        if mask[j] or float(dm[j].item()) < 0:
            break
        selected.append(j)
        mask[j] = True
    sel_set = set(selected)
    c = 0
    while len(selected) < b and c < N:
        if c not in sel_set:
            selected.append(c)
            sel_set.add(c)
        c += 1
    return torch.tensor(sorted(selected[:b]), device=device, dtype=torch.long)


def _merge_bicrit_orthbonus_sel_stats(dst: Dict[str, float], src: Dict[str, float]) -> None:
    if src.get("n_steps", 0) <= 0:
        return
    w = float(src["n_steps"])
    dst["_w_base"] = dst.get("_w_base", 0.0) + w * float(src["base_bicrit_score_mean"])
    dst["_w_imp"] = dst.get("_w_imp", 0.0) + w * float(src.get("importance_score_mean", 0.0))
    dst["_w_orth"] = dst.get("_w_orth", 0.0) + w * float(src["orthbonus_mean"])
    dst["_w_red"] = dst.get("_w_red", 0.0) + w * float(src["redundancy_penalty_mean"])
    dst["_w_final"] = dst.get("_w_final", 0.0) + w * float(src["final_score_mean"])
    dst["_w_pool"] = dst.get("_w_pool", 0.0) + w * float(src.get("candidate_pool_mean_size", 0.0))
    dst["_w_pref"] = dst.get("_w_pref", 0.0) + w * float(src.get("prefilter_candidate_mean_size", 0.0))
    dst["_w_tot"] = dst.get("_w_tot", 0.0) + w


def _finalize_bicrit_orthbonus_sel_stats(acc: Dict[str, float]) -> None:
    tot = float(acc.get("_w_tot", 0.0))
    if tot <= 0.0:
        acc["base_bicrit_score_mean"] = float("nan")
        acc["importance_score_mean"] = float("nan")
        acc["orthbonus_mean"] = float("nan")
        acc["redundancy_penalty_mean"] = float("nan")
        acc["final_score_mean"] = float("nan")
        acc["candidate_pool_mean_size"] = float("nan")
        acc["prefilter_candidate_mean_size"] = float("nan")
        return
    acc["base_bicrit_score_mean"] = acc["_w_base"] / tot
    acc["importance_score_mean"] = acc["_w_imp"] / tot
    acc["orthbonus_mean"] = acc["_w_orth"] / tot
    acc["redundancy_penalty_mean"] = acc["_w_red"] / tot
    acc["final_score_mean"] = acc["_w_final"] / tot
    acc["candidate_pool_mean_size"] = acc["_w_pool"] / tot
    acc["prefilter_candidate_mean_size"] = acc["_w_pref"] / tot
    for k in list(acc.keys()):
        if k.startswith("_w_"):
            del acc[k]


def _orthbonus_normalize_pool(
    vals: torch.Tensor,
    eps: float = 1e-8,
    mode: str = "none",
) -> torch.Tensor:
    """Min-max or z-score over all entries in ``vals`` (already restricted to the pool)."""
    m = str(mode)
    if m == "none":
        return vals
    if vals.numel() == 0:
        return vals
    if m == "minmax":
        lo = vals.min()
        hi = vals.max()
        return (vals - lo) / (hi - lo).clamp(min=eps)
    if m == "zscore":
        mu = vals.mean()
        sig = vals.std(unbiased=False).clamp(min=eps)
        return (vals - mu) / sig
    raise ValueError(f"Unknown coreset_orthbonus_score_norm: {mode!r}")


def select_bicriteria_farthest_mixed_orthbonus(
    K: torch.Tensor,
    V: torch.Tensor,
    u_importance: torch.Tensor,
    budget: int,
    bicriteria_alpha: float,
    orthbonus_lambda: float,
    orthbonus_eta: float,
    orthbonus_mode: str,
    orthbonus_beta: float,
    orthbonus_prefilter_ratio: float,
    orthbonus_score_norm: str,
    *,
    force_prefilter: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Greedy D² mixed bicriteria (``d_mix``) plus orthogonal novelty, with optional importance
    mixing, prefilter-by-``d_mix``, and per-step score normalization over the active pool.

    Final score (per unpicked candidate in the pool)::

        inner = base_bicrit + orthbonus_lambda * orthbonus
        score = beta * importance + (1 - beta) * inner

    ``orthbonus_mode``:
    - ``maxcos`` / ``lowrank_frame``: cheap cosine redundancy bonus (same as legacy orthbonus).
    - ``residual_l2``: Gram–Schmidt orthogonal residual energy, K/V weighted by ``orthbonus_eta``.
    """
    device = K.device
    N = int(K.shape[0])
    b = min(int(budget), N)
    nan_stats: Dict[str, float] = {
        "base_bicrit_score_mean": float("nan"),
        "importance_score_mean": float("nan"),
        "orthbonus_mean": float("nan"),
        "redundancy_penalty_mean": float("nan"),
        "final_score_mean": float("nan"),
        "candidate_pool_mean_size": float("nan"),
        "prefilter_candidate_mean_size": float("nan"),
        "n_steps": 0.0,
    }
    if b <= 0 or N == 0:
        return torch.zeros(0, dtype=torch.long, device=device), nan_stats
    if b >= N:
        return torch.arange(N, device=device, dtype=torch.long), nan_stats

    if force_prefilter and float(orthbonus_prefilter_ratio) >= 1.0 - 1e-12:
        raise ValueError(
            "force_prefilter=True requires --coreset_orthbonus_prefilter_ratio < 1.0"
        )

    a = min(1.0, max(0.0, float(bicriteria_alpha)))
    lam = float(orthbonus_lambda)
    eta = min(1.0, max(0.0, float(orthbonus_eta)))
    beta = min(1.0, max(0.0, float(orthbonus_beta)))
    mode = str(orthbonus_mode)
    ratio_pf = float(orthbonus_prefilter_ratio)
    use_prefilter = bool(force_prefilter) or (ratio_pf < 1.0 - 1e-12)
    norm_mode = str(orthbonus_score_norm)

    if mode not in ("maxcos", "lowrank_frame", "residual_l2"):
        raise ValueError(f"Unknown coreset_orthbonus_mode: {orthbonus_mode!r}")
    if norm_mode not in ("none", "minmax", "zscore"):
        raise ValueError(f"Unknown coreset_orthbonus_score_norm: {orthbonus_score_norm!r}")

    Kf = K.float()
    Vf = V.float()
    u_imp = u_importance.float().reshape(-1)
    if int(u_imp.shape[0]) != N:
        raise RuntimeError("u_importance length must match K/V rows")

    k_cent = Kf.mean(0, keepdim=True)
    v_cent = Vf.mean(0, keepdim=True)
    kn = Kf / Kf.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    vn = Vf / Vf.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    basis_k = _IncOrthonormalBasis(int(K.shape[1]), device)
    basis_v = _IncOrthonormalBasis(int(V.shape[1]), device)

    mask = torch.zeros(N, dtype=torch.bool, device=device)
    selected: List[int] = []

    base_sum = 0.0
    imp_sum = 0.0
    orth_sum = 0.0
    red_sum = 0.0
    final_sum = 0.0
    pool_sum = 0.0
    pref_sum = 0.0
    n_pick = 0

    for _ in range(b):
        idx_u = (~mask).nonzero(as_tuple=False).squeeze(-1)
        num_rem = int(idx_u.numel())
        if num_rem == 0:
            break

        if not selected:
            dK = (Kf - k_cent).pow(2).sum(dim=-1)
            dV = (Vf - v_cent).pow(2).sum(dim=-1)
            if mode == "residual_l2":
                rk = basis_k.residual_norm_sq_batch(Kf)
                rv = basis_v.residual_norm_sq_batch(Vf)
                orth_bonus = eta * rk + (1.0 - eta) * rv
                redundancy = torch.zeros(N, device=device, dtype=torch.float32)
            else:
                redundancy = torch.zeros(N, device=device, dtype=torch.float32)
                orth_bonus = torch.ones(N, device=device, dtype=torch.float32)
        else:
            S = torch.tensor(selected, device=device, dtype=torch.long)
            KS = Kf.index_select(0, S)
            VS = Vf.index_select(0, S)
            dK = (Kf.unsqueeze(1) - KS.unsqueeze(0)).pow(2).sum(dim=-1).min(dim=1).values
            dV = (Vf.unsqueeze(1) - VS.unsqueeze(0)).pow(2).sum(dim=-1).min(dim=1).values
            if mode in ("maxcos", "lowrank_frame"):
                KS_n = kn.index_select(0, S)
                VS_n = vn.index_select(0, S)
                cos_k = torch.mm(kn, KS_n.T).clamp(min=0.0)
                cos_v = torch.mm(vn, VS_n.T).clamp(min=0.0)
                max_cos_k, _ = cos_k.max(dim=1)
                max_cos_v, _ = cos_v.max(dim=1)
                if mode == "maxcos":
                    redundancy = eta * max_cos_k + (1.0 - eta) * max_cos_v
                    orth_bonus = (1.0 - redundancy).clamp(min=0.0, max=1.0)
                else:
                    u_k = kn.index_select(0, S).mean(dim=0, keepdim=True)
                    u_k = u_k / u_k.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                    u_v = vn.index_select(0, S).mean(dim=0, keepdim=True)
                    u_v = u_v / u_v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                    align_k = (kn * u_k).sum(dim=-1).pow(2)
                    align_v = (vn * u_v).sum(dim=-1).pow(2)
                    redundancy = eta * align_k + (1.0 - eta) * align_v
                    orth_bonus = (1.0 - redundancy).clamp(min=0.0, max=1.0)
            else:
                rk = basis_k.residual_norm_sq_batch(Kf)
                rv = basis_v.residual_norm_sq_batch(Vf)
                orth_bonus = eta * rk + (1.0 - eta) * rv
                redundancy = torch.zeros(N, device=device, dtype=torch.float32)

        base_dm = a * dK + (1.0 - a) * dV

        base_u = base_dm.index_select(0, idx_u)
        if use_prefilter:
            M = max(8, int(math.ceil(ratio_pf * num_rem)))
            M = min(M, num_rem)
            _, sub = torch.topk(base_u, k=M, largest=True)
            pool_idx = idx_u.index_select(0, sub)
        else:
            M = num_rem
            pool_idx = idx_u

        pool_sum += float(num_rem)
        pref_sum += float(M)

        base_p = base_dm.index_select(0, pool_idx)
        imp_p = u_imp.index_select(0, pool_idx)
        orth_p = orth_bonus.index_select(0, pool_idx)
        red_p = redundancy.index_select(0, pool_idx)

        if norm_mode != "none":
            base_p = _orthbonus_normalize_pool(base_p, mode=norm_mode)
            imp_p = _orthbonus_normalize_pool(imp_p, mode=norm_mode)
            orth_p = _orthbonus_normalize_pool(orth_p, mode=norm_mode)

        inner_p = base_p + lam * orth_p
        score_p = beta * imp_p + (1.0 - beta) * inner_p

        j_rel = int(torch.argmax(score_p).item())
        j = int(pool_idx[j_rel].item())
        if mask[j]:
            break

        selected.append(j)
        mask[j] = True
        if mode == "residual_l2":
            basis_k.append_vector(Kf[j])
            basis_v.append_vector(Vf[j])

        base_sum += float(base_dm[j].item())
        imp_sum += float(u_imp[j].item())
        orth_sum += float(orth_bonus[j].item())
        red_sum += float(redundancy[j].item())
        final_sum += float((beta * u_imp[j] + (1.0 - beta) * (base_dm[j] + lam * orth_bonus[j])).item())
        n_pick += 1

    sel_set = set(selected)
    c = 0
    while len(selected) < b and c < N:
        if c not in sel_set:
            selected.append(c)
            sel_set.add(c)
        c += 1
    idx = torch.tensor(sorted(selected[:b]), device=device, dtype=torch.long)
    n_f = float(max(n_pick, 1))
    stats: Dict[str, float] = {
        "base_bicrit_score_mean": base_sum / n_f,
        "importance_score_mean": imp_sum / n_f,
        "orthbonus_mean": orth_sum / n_f,
        "redundancy_penalty_mean": red_sum / n_f,
        "final_score_mean": final_sum / n_f,
        "candidate_pool_mean_size": pool_sum / n_f,
        "prefilter_candidate_mean_size": pref_sum / n_f,
        "n_steps": float(n_pick),
    }
    return idx, stats


def _cords_select_indices(
    key_states_to_compress: torch.Tensor,
    value_states_to_compress: torch.Tensor,
    num_heads: int,
    token_per_frame: int,
    current_frame_num: int,
    compress_frame_num: int,
    tar_ratio: float,
    query_ratio: float,
    avg_pooling_nd: int,
    attn_pool_size: int,
    model,
    coreset_num_iters: int,
    compression_method: str,
    bicriteria_alpha: float,
    orthbonus_lambda: float,
    orthbonus_mode: str,
    orthbonus_eta: float,
    orthbonus_beta: float,
    orthbonus_prefilter_ratio: float,
    orthbonus_score_norm: str,
    recent_aware: bool,
    recent_summary_k: int,
    diagnostics_out: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Protected recent + old region; old tokens via D² bicriteria + orthogonal bonus (configurable).
    """
    device = key_states_to_compress.device
    total_length = key_states_to_compress.shape[2]
    total_tokens_to_keep = int(round(compress_frame_num * token_per_frame))
    cm = str(compression_method)
    force_pf = False

    query_length = 0
    if tar_ratio > 0:
        query_frame = int(current_frame_num * query_ratio)
        query_length = int(query_frame * token_per_frame)
    query_length = min(query_length, total_length)
    split = total_length - query_length
    if split < 0:
        split = 0
        query_length = total_length

    if query_length > 0:
        protected = torch.arange(split, total_length, device=device, dtype=torch.long)
    else:
        protected = torch.zeros(0, dtype=torch.long, device=device)

    old_budget = max(0, total_tokens_to_keep - int(protected.numel()))
    if old_budget > split:
        raise RuntimeError(
            "cords: old_budget exceeds non-protected length"
        )

    val_norm_score = _val_norm_pooled(
        value_states_to_compress,
        token_per_frame,
        avg_pooling_nd,
        attn_pool_size,
        model,
    )

    alpha = min(1.0, max(0.0, float(bicriteria_alpha)))

    out_rows: List[torch.Tensor] = []
    last_cov_k: Optional[float] = None
    last_cov_v: Optional[float] = None
    last_cov_j: Optional[float] = None
    any_direct_safety_fill = False
    diag_before_fill = 0
    diag_final = 0
    feat_level_last = "token"
    orth_sel_accum: Dict[str, float] = {}

    for h in range(num_heads):
        key_h = key_states_to_compress[0, h, :, :].float()
        value_h = value_states_to_compress[0, h, :, :].float()

        need_means = recent_aware and protected.numel() > 0
        if need_means:
            k_mean_prot, v_mean_prot = _protected_kv_means(
                key_h, value_h, protected, recent_summary_k
            )
        else:
            k_mean_prot, v_mean_prot = None, None

        if old_budget == 0:
            pn = int(protected.numel())
            if pn == total_tokens_to_keep:
                row = protected
            elif pn > total_tokens_to_keep:
                sc = val_norm_score[h, protected]
                row = protected[
                    torch.topk(sc, k=total_tokens_to_keep, largest=True).indices
                ].sort().values
            else:
                raise RuntimeError(
                    "cords: protected shorter than budget"
                )
            diag_before_fill = 0
            diag_final = 0
            out_rows.append(row)
            continue
        if split <= 0:
            raise RuntimeError("no old tokens for bicrit orthbonus coreset")

        x_old = key_h[:split]
        v_old = value_h[:split]
        Kp, Vp = _bicriteria_apply_residual(x_old, v_old, k_mean_prot, v_mean_prot, recent_aware)

        use_frame_path = token_per_frame > 1 and split % token_per_frame == 0
        feat_level = "frame" if use_frame_path else "token"
        feat_level_last = feat_level

        u_old = val_norm_score[h, :split].float()

        if use_frame_path:
            F_old = split // token_per_frame
            n_frames = min(F_old, int(math.ceil(old_budget / float(token_per_frame))))
            k_old_f = Kp.reshape(F_old, token_per_frame, -1).mean(dim=1)
            v_old_f = Vp.reshape(F_old, token_per_frame, -1).mean(dim=1)
            u_f = u_old.reshape(F_old, token_per_frame).mean(dim=1)
            idx_f, sel_stats = select_bicriteria_farthest_mixed_orthbonus(
                k_old_f,
                v_old_f,
                u_f,
                n_frames,
                alpha,
                float(orthbonus_lambda),
                float(orthbonus_eta),
                str(orthbonus_mode),
                float(orthbonus_beta),
                float(orthbonus_prefilter_ratio),
                str(orthbonus_score_norm),
                force_prefilter=force_pf,
            )
            _merge_bicrit_orthbonus_sel_stats(orth_sel_accum, sel_stats)
            flat: List[int] = []
            for f in idx_f.detach().cpu().tolist():
                base = int(f) * token_per_frame
                flat.extend(range(base, base + token_per_frame))
            abs_cand = torch.tensor(flat, device=device, dtype=torch.long)
            abs_cand = torch.unique(abs_cand, sorted=True)
            if int(abs_cand.numel()) > old_budget:
                sel_old = _trim_token_indices_bicriteria_mixed(
                    Kp, Vp, abs_cand, old_budget, alpha
                )
            else:
                sel_old = abs_cand
        else:
            sel_old, sel_stats = select_bicriteria_farthest_mixed_orthbonus(
                Kp,
                Vp,
                u_old,
                old_budget,
                alpha,
                float(orthbonus_lambda),
                float(orthbonus_eta),
                str(orthbonus_mode),
                float(orthbonus_beta),
                float(orthbonus_prefilter_ratio),
                str(orthbonus_score_norm),
                force_prefilter=force_pf,
            )
            _merge_bicrit_orthbonus_sel_stats(orth_sel_accum, sel_stats)

        selected_old_token_count_before_fill = int(sel_old.numel())

        if selected_old_token_count_before_fill < old_budget:
            n_pre_vfill = selected_old_token_count_before_fill
            mask_old = torch.ones(split, dtype=torch.bool, device=device)
            mask_old[sel_old] = False
            rest = mask_old.nonzero(as_tuple=False).squeeze(-1)
            if rest.numel() > 0:
                srest = val_norm_score[h, rest]
                need = old_budget - int(sel_old.numel())
                extra = rest[
                    torch.topk(srest, k=min(need, int(rest.numel())), largest=True).indices
                ]
                sel_old = torch.unique(torch.cat([sel_old, extra]), sorted=True)
            if int(sel_old.numel()) > n_pre_vfill:
                any_direct_safety_fill = True

        if int(sel_old.numel()) < old_budget:
            raise RuntimeError("bicrit orthbonus coreset could not fill old_budget")

        if int(sel_old.numel()) > old_budget:
            sel_old = _trim_token_indices_bicriteria_mixed(Kp, Vp, sel_old, old_budget, alpha)

        selected_old_token_count_final = int(sel_old.numel())

        k_sum_all = x_old.sum(dim=0)
        v_sum_all = v_old.sum(dim=0)
        k_sum_sel = x_old[sel_old.long()].sum(dim=0)
        v_sum_sel = v_old[sel_old.long()].sum(dim=0)
        last_cov_k = float((k_sum_all - k_sum_sel).norm().item())
        last_cov_v = float((v_sum_all - v_sum_sel).norm().item())
        last_cov_j = float(
            torch.cat([k_sum_all - k_sum_sel, v_sum_all - v_sum_sel], dim=0).norm().item()
        )

        row = torch.unique(torch.cat([sel_old, protected]), sorted=True)
        if int(row.numel()) != total_tokens_to_keep:
            raise RuntimeError("bicrit orthbonus coreset row length mismatch")
        out_rows.append(row)

        diag_before_fill = selected_old_token_count_before_fill
        diag_final = selected_old_token_count_final

    if diagnostics_out is not None:
        _finalize_bicrit_orthbonus_sel_stats(orth_sel_accum)
        eff_ratio = float(orthbonus_prefilter_ratio)
        prefilter_active = bool(force_pf) or (eff_ratio < 1.0 - 1e-12)
        diagnostics_out.clear()
        diagnostics_out.update(
            {
                "method": str(compression_method),
                "selector_mode": "d2_orthbonus",
                "alpha": float(alpha),
                "beta": float(orthbonus_beta),
                "coreset_bicriteria_alpha": float(alpha),
                "orthbonus_lambda": float(orthbonus_lambda),
                "coreset_orthbonus_lambda": float(orthbonus_lambda),
                "coreset_orthbonus_mode": str(orthbonus_mode),
                "orthbonus_mode": str(orthbonus_mode),
                "coreset_orthbonus_eta": float(orthbonus_eta),
                "coreset_orthbonus_beta": float(orthbonus_beta),
                "coreset_orthbonus_prefilter_ratio": float(orthbonus_prefilter_ratio),
                "orthbonus_prefilter_ratio": float(orthbonus_prefilter_ratio),
                "orthbonus_prefilter_active": bool(prefilter_active),
                "coreset_orthbonus_score_norm": str(orthbonus_score_norm),
                "orthbonus_score_norm": str(orthbonus_score_norm),
                "base_bicrit_score_mean": orth_sel_accum.get(
                    "base_bicrit_score_mean", float("nan")
                ),
                "importance_score_mean": orth_sel_accum.get(
                    "importance_score_mean", float("nan")
                ),
                "orthbonus_mean": orth_sel_accum.get("orthbonus_mean", float("nan")),
                "redundancy_penalty_mean": orth_sel_accum.get(
                    "redundancy_penalty_mean", float("nan")
                ),
                "final_score_mean": orth_sel_accum.get("final_score_mean", float("nan")),
                "candidate_pool_mean_size": orth_sel_accum.get(
                    "candidate_pool_mean_size", float("nan")
                ),
                "prefilter_candidate_mean_size": orth_sel_accum.get(
                    "prefilter_candidate_mean_size", float("nan")
                ),
                "selected_old_token_count_before_fill": int(diag_before_fill),
                "selected_old_token_count_final": int(diag_final),
                "used_direct_safety_fill": bool(any_direct_safety_fill),
                "coverage_residual_k_l2": last_cov_k,
                "coverage_residual_v_l2": last_cov_v,
                "coverage_residual_joint_l2": last_cov_j,
                "feature_level": feat_level_last,
            }
        )

    return torch.stack(out_rows, dim=0)


def select_bicriteria_greedy_mixed(
    K: torch.Tensor, V: torch.Tensor, budget: int, alpha: float
) -> torch.Tensor:
    """
    Bicriteria analog of ``_greedy_linear_coreset_indices``: greedily add rows so nonnegative
    conic combinations approximate ``sum(K)`` and ``sum(V)`` with separate LS weights per
    space. Each step picks ``j`` maximizing
    ``alpha * <K_j, r_K> + (1-alpha) * <V_j, r_V>`` with residuals ``r_K``, ``r_V`` after
    refitting on the current selected set (not farthest-point / D²).
    """
    device = K.device
    N = int(K.shape[0])
    b = min(int(budget), N)
    if b <= 0 or N == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    if b >= N:
        return torch.arange(N, device=device, dtype=torch.long)
    a = min(1.0, max(0.0, float(alpha)))
    Kf = K.float()
    Vf = V.float()
    k_full = Kf.sum(dim=0)
    v_full = Vf.sum(dim=0)
    r_k = k_full.clone()
    r_v = v_full.clone()
    selected: List[int] = []
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=torch.float32)

    for _ in range(b):
        scores = a * torch.mv(Kf, r_k) + (1.0 - a) * torch.mv(Vf, r_v)
        if selected:
            scores = scores.clone()
            scores[torch.tensor(selected, device=device, dtype=torch.long)] = neg_inf
        j = int(torch.argmax(scores).item())
        if selected and j in selected:
            break
        selected.append(j)
        KS = Kf[selected].T
        VS = Vf[selected].T
        wk = _fit_weights_ls_nonneg(KS, k_full)
        wv = _fit_weights_ls_nonneg(VS, v_full)
        r_k = k_full - KS @ wk
        r_v = v_full - VS @ wv

    sel_set = set(selected)
    c = 0
    while len(selected) < b and c < N:
        if c not in sel_set:
            selected.append(c)
            sel_set.add(c)
        c += 1
    return torch.tensor(sorted(selected[:b]), device=device, dtype=torch.long)


def select_bicriteria_kmeans_mixed(
    K: torch.Tensor, V: torch.Tensor, budget: int, num_iters: int, alpha: float
) -> torch.Tensor:
    """Lloyd on mixed distance; pick nearest real row per centroid (mixed dist)."""
    device = K.device
    N = int(K.shape[0])
    b = min(int(budget), N)
    if b <= 0 or N == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    if b >= N:
        return torch.arange(N, device=device, dtype=torch.long)
    a = min(1.0, max(0.0, float(alpha)))
    Kf = K.float()
    Vf = V.float()
    if b == 1:
        dm0 = a * (Kf - Kf.mean(0, keepdim=True)).pow(2).sum(-1) + (1 - a) * (
            Vf - Vf.mean(0, keepdim=True)
        ).pow(2).sum(-1)
        return torch.tensor([int(torch.argmax(dm0).item())], device=device, dtype=torch.long)

    pos = torch.linspace(0, N - 1, b, device=device).round().long().clamp(0, N - 1)
    init_list: List[int] = []
    seen_i: set = set()
    for t in pos.tolist():
        ti = int(t)
        if ti not in seen_i:
            seen_i.add(ti)
            init_list.append(ti)
    cfill = 0
    while len(init_list) < b and cfill < N:
        if cfill not in seen_i:
            seen_i.add(cfill)
            init_list.append(cfill)
        cfill += 1
    init_idx = torch.tensor(init_list[:b], device=device, dtype=torch.long)
    CK = Kf[init_idx].clone()
    CV = Vf[init_idx].clone()

    for _ in range(max(1, int(num_iters))):
        dist_mix = a * (Kf.unsqueeze(1) - CK.unsqueeze(0)).pow(2).sum(-1) + (1 - a) * (
            Vf.unsqueeze(1) - CV.unsqueeze(0)
        ).pow(2).sum(-1)
        assign = dist_mix.argmin(dim=1)
        new_ck: List[torch.Tensor] = []
        new_cv: List[torch.Tensor] = []
        for kk in range(b):
            m = assign == kk
            if m.any():
                new_ck.append(Kf[m].mean(dim=0))
                new_cv.append(Vf[m].mean(dim=0))
            else:
                new_ck.append(CK[kk])
                new_cv.append(CV[kk])
        CK = torch.stack(new_ck, dim=0)
        CV = torch.stack(new_cv, dim=0)

    dist_c = a * (Kf.unsqueeze(1) - CK.unsqueeze(0)).pow(2).sum(-1) + (1 - a) * (
        Vf.unsqueeze(1) - CV.unsqueeze(0)
    ).pow(2).sum(-1)
    nearest_per_centroid = dist_c.argmin(dim=0)

    picked: List[int] = []
    seen: set = set()
    for kk in range(b):
        idx = int(nearest_per_centroid[kk].item())
        if idx not in seen:
            seen.add(idx)
            picked.append(idx)

    def _farthest_fill_mixed() -> Optional[int]:
        if len(picked) >= b:
            return None
        mask_unpicked = torch.tensor([i not in seen for i in range(N)], device=device)
        if not bool(mask_unpicked.any()):
            return None
        rem = mask_unpicked.nonzero(as_tuple=False).squeeze(-1)
        if rem.numel() == 0:
            return None
        S = torch.tensor(picked, device=device, dtype=torch.long)
        dK = (Kf[rem].unsqueeze(1) - Kf[S].unsqueeze(0)).pow(2).sum(dim=-1).min(dim=1).values
        dV = (Vf[rem].unsqueeze(1) - Vf[S].unsqueeze(0)).pow(2).sum(dim=-1).min(dim=1).values
        dm = a * dK + (1 - a) * dV
        j = int(rem[int(torch.argmax(dm).item())].item())
        return j

    while len(picked) < b:
        j = _farthest_fill_mixed()
        if j is None:
            break
        picked.append(j)
        seen.add(j)
    c2 = 0
    while len(picked) < b and c2 < N:
        if c2 not in seen:
            picked.append(c2)
            seen.add(c2)
        c2 += 1
    return torch.tensor(sorted(picked[:b]), device=device, dtype=torch.long)


def select_bicriteria_separate_merge(
    K: torch.Tensor,
    V: torch.Tensor,
    budget: int,
    alpha: float,
    per_space_selector: str,
    coreset_num_iters: int,
) -> torch.Tensor:
    """
    K-only and V-only coresets of size ``budget`` each (selector per ``per_space_selector``:
    ``kmeans``, ``greedy`` [_greedy_linear_coreset_indices], or ``d2``); union; if union
    exceeds ``budget``, trim with ``select_bicriteria_farthest_mixed`` (mixed d_mix).
    """
    device = K.device
    N = int(K.shape[0])
    b = min(int(budget), N)
    if b <= 0 or N == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    if b >= N:
        return torch.arange(N, device=device, dtype=torch.long)
    if per_space_selector == "kmeans":
        sk = select_coreset_indices_kmeans(K.float(), b, coreset_num_iters)
        sv = select_coreset_indices_kmeans(V.float(), b, coreset_num_iters)
    elif per_space_selector == "greedy":
        sk = _greedy_linear_coreset_indices(K.float(), b)
        sv = _greedy_linear_coreset_indices(V.float(), b)
    else:
        sk = select_coreset_indices_d2(K.float(), b)
        sv = select_coreset_indices_d2(V.float(), b)
    U = torch.unique(torch.cat([sk.long(), sv.long()]).sort().values)
    if int(U.numel()) <= b:
        return U
    Ks = K.float()[U]
    Vs = V.float()[U]
    loc = select_bicriteria_farthest_mixed(Ks, Vs, b, alpha)
    return U[loc.long()].sort().values


def _trim_token_indices_bicriteria_mixed(
    K: torch.Tensor,
    V: torch.Tensor,
    abs_token_idx: torch.Tensor,
    budget: int,
    alpha: float,
) -> torch.Tensor:
    """Subset of abs_token_idx (global indices into K/V rows) of size budget via mixed farthest."""
    if int(abs_token_idx.numel()) <= budget:
        return abs_token_idx.sort().values
    Ks = K[abs_token_idx.long()].float()
    Vs = V[abs_token_idx.long()].float()
    loc = select_bicriteria_farthest_mixed(Ks, Vs, budget, alpha)
    return abs_token_idx[loc.long()].sort().values


def _protected_kv_means(
    key_h: torch.Tensor,
    value_h: torch.Tensor,
    protected: torch.Tensor,
    recent_summary_k: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Mean K/V over protected tokens (last min(k, len) positions if recent_summary_k > 0)."""
    if protected.numel() == 0:
        return None, None
    if recent_summary_k is None or recent_summary_k <= 0:
        idx = protected.long()
    else:
        take = min(int(recent_summary_k), int(protected.numel()))
        idx = protected[-take:].long()
    k_mean = key_h[idx].float().mean(dim=0)
    v_mean = value_h[idx].float().mean(dim=0)
    return k_mean, v_mean


def _val_norm_pooled(
    value_states_to_compress: torch.Tensor,
    token_per_frame: int,
    avg_pooling_nd: int,
    attn_pool_size: int,
    model,
) -> torch.Tensor:
    """Match direct-coreset pooled V-norm scores for in-method safety fill only."""
    val_norm_score = value_states_to_compress.norm(dim=-1)[0, :, :].clone()
    if avg_pooling_nd <= 0:
        return val_norm_score
    pool_size = attn_pool_size
    head_num, _ = val_norm_score.shape
    frame_num = val_norm_score.shape[-1] // token_per_frame
    if avg_pooling_nd == 1:
        avg_pool = torch.nn.AvgPool1d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
    elif avg_pooling_nd == 2:
        avg_pool = torch.nn.AvgPool2d(
            kernel_size=(pool_size, pool_size),
            stride=(1, 1),
            padding=(pool_size // 2, pool_size // 2),
        )
    elif avg_pooling_nd == 3:
        avg_pool = torch.nn.AvgPool3d(
            kernel_size=(pool_size, pool_size, pool_size),
            stride=(1, 1, 1),
            padding=(pool_size // 2, pool_size // 2, pool_size // 2),
        )
    else:
        raise NotImplementedError("We support 1D - 3D Pooling")
    patch_size = model.config.height_width if hasattr(model.config, "height_width") else 14
    if avg_pooling_nd > 1:
        original_numel = val_norm_score.numel()
        val_norm_reshaped = val_norm_score.reshape(head_num, frame_num, patch_size, -1)
        T, H, W = val_norm_reshaped.shape[-3], val_norm_reshaped.shape[-2], val_norm_reshaped.shape[-1]
        pool_size_eff = min(pool_size, T, H, W)
        if pool_size_eff % 2 == 0:
            pool_size_eff -= 1
        if pool_size_eff >= 2:
            if pool_size_eff != pool_size:
                pool_size = pool_size_eff
                if avg_pooling_nd == 2:
                    avg_pool = torch.nn.AvgPool2d(
                        kernel_size=(pool_size, pool_size),
                        stride=(1, 1),
                        padding=(pool_size // 2, pool_size // 2),
                    )
                elif avg_pooling_nd == 3:
                    avg_pool = torch.nn.AvgPool3d(
                        kernel_size=(pool_size, pool_size, pool_size),
                        stride=(1, 1, 1),
                        padding=(pool_size // 2, pool_size // 2, pool_size // 2),
                    )
            val_norm_pooled = avg_pool(val_norm_reshaped)
            val_norm_score = val_norm_pooled.reshape(head_num, -1)
            assert original_numel == val_norm_score.numel()
    else:
        val_norm_score = avg_pool(val_norm_score)
    return val_norm_score


_PROBE_RANDOM_GLOBAL_SEED = 9_204_117

_PROBE_METHODS = frozenset(
    {
        "legacy-coreset-probe",
        "legacy-coreset-probe-d2",
        "legacy-coreset-probe-recent",
    }
)


def _is_probe_method(method: str) -> bool:
    return str(method) in _PROBE_METHODS


_ORTHBICRIT_METHODS = frozenset(
    {
        "legacy-coreset-orthbicrit-d2",
        "legacy-coreset-orthbicrit-recent",
        "legacy-coreset-orthbicrit-frame",
    }
)


def _is_orthbicrit_method(method: str) -> bool:
    return str(method) in _ORTHBICRIT_METHODS


class _IncOrthonormalBasis:
    """Incremental orthonormal columns (modified Gram–Schmidt) for stable Proj / residual."""

    def __init__(self, d: int, device: torch.device, eps: float = 1e-5) -> None:
        self.d = int(d)
        self.device = device
        self.eps = float(eps)
        self.Q: Optional[torch.Tensor] = None

    @property
    def rank(self) -> int:
        return 0 if self.Q is None else int(self.Q.shape[1])

    def residual_norm_sq_batch(self, X: torch.Tensor) -> torch.Tensor:
        """X: [N, d] -> per-row ||x - Q Q^T x||^2."""
        Xf = X.float()
        if self.Q is None or self.Q.shape[1] == 0:
            return (Xf * Xf).sum(dim=1)
        coeff = Xf @ self.Q
        proj = coeff @ self.Q.T
        r = Xf - proj
        return (r * r).sum(dim=1)

    def append_vector(self, v: torch.Tensor) -> None:
        v = v.reshape(-1).float().clone()
        if self.Q is not None:
            for j in range(self.Q.shape[1]):
                qj = self.Q[:, j]
                v = v - (qj @ v) * qj
        n = float(v.norm().item())
        if n <= self.eps:
            return
        col = (v / n).unsqueeze(1)
        if self.Q is None:
            self.Q = col
        else:
            self.Q = torch.cat([self.Q, col], dim=1)


def select_orthbicrit_greedy_indices(
    K: torch.Tensor,
    V: torch.Tensor,
    u_scores: torch.Tensor,
    budget: int,
    alpha: float,
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Greedy orthogonal bicriteria: each step maximize
    beta*u + (1-beta)*(alpha*||r^K||^2 + (1-alpha)*||r^V||^2) with r orthogonal to selected K/V spans.
    """
    device = K.device
    N = int(K.shape[0])
    dk = int(K.shape[1])
    dv = int(V.shape[1])
    b = min(int(budget), N)
    if b <= 0 or N == 0:
        return torch.zeros(0, dtype=torch.long, device=device), {
            "basis_rank_k": 0,
            "basis_rank_v": 0,
            "orth_residual_k_mean": 0.0,
            "orth_residual_v_mean": 0.0,
            "orth_residual_joint_mean": 0.0,
        }
    if b >= N:
        rk0 = (K.float().pow(2).sum(dim=1)).mean().item()
        rv0 = (V.float().pow(2).sum(dim=1)).mean().item()
        a = min(1.0, max(0.0, float(alpha)))
        return torch.arange(N, device=device, dtype=torch.long), {
            "basis_rank_k": min(N, dk),
            "basis_rank_v": min(N, dv),
            "orth_residual_k_mean": float(rk0),
            "orth_residual_v_mean": float(rv0),
            "orth_residual_joint_mean": float(a * rk0 + (1.0 - a) * rv0),
        }

    a = min(1.0, max(0.0, float(alpha)))
    bt = min(1.0, max(0.0, float(beta)))
    basis_k = _IncOrthonormalBasis(dk, device)
    basis_v = _IncOrthonormalBasis(dv, device)
    picked_mask = torch.zeros(N, dtype=torch.bool, device=device)
    step_rk: List[float] = []
    step_rv: List[float] = []
    step_joint: List[float] = []

    for _ in range(b):
        rk = basis_k.residual_norm_sq_batch(K)
        rv = basis_v.residual_norm_sq_batch(V)
        score_resid = a * rk + (1.0 - a) * rv
        if bt > 0:
            score = bt * u_scores.float() + (1.0 - bt) * score_resid
        else:
            score = score_resid
        score = score.clone()
        score[picked_mask] = float("-inf")
        j = int(torch.argmax(score).item())
        if picked_mask[j]:
            break
        step_rk.append(float(rk[j].item()))
        step_rv.append(float(rv[j].item()))
        step_joint.append(float((a * rk[j] + (1.0 - a) * rv[j]).item()))
        picked_mask[j] = True
        basis_k.append_vector(K[j])
        basis_v.append_vector(V[j])

    idx = picked_mask.nonzero(as_tuple=False).squeeze(-1).sort().values
    meta = {
        "basis_rank_k": basis_k.rank,
        "basis_rank_v": basis_v.rank,
        "orth_residual_k_mean": float(sum(step_rk) / max(len(step_rk), 1)),
        "orth_residual_v_mean": float(sum(step_rv) / max(len(step_rv), 1)),
        "orth_residual_joint_mean": float(sum(step_joint) / max(len(step_joint), 1)),
    }
    return idx, meta


def _trim_with_orthbicrit(
    Kp: torch.Tensor,
    Vp: torch.Tensor,
    u_old: torch.Tensor,
    abs_idx: torch.Tensor,
    budget: int,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """abs_idx: global old-region indices; keep `budget` rows via orthogonal greedy on that subset."""
    if int(abs_idx.numel()) <= budget:
        return abs_idx.sort().values
    Ka = Kp[abs_idx.long()]
    Va = Vp[abs_idx.long()]
    ua = u_old[abs_idx.long()]
    sub_rel, _ = select_orthbicrit_greedy_indices(Ka, Va, ua, budget, alpha, beta)
    return abs_idx[sub_rel.long()].sort().values


def _orthbicrit_select_indices(
    key_states_to_compress: torch.Tensor,
    value_states_to_compress: torch.Tensor,
    num_heads: int,
    token_per_frame: int,
    current_frame_num: int,
    compress_frame_num: int,
    tar_ratio: float,
    query_ratio: float,
    avg_pooling_nd: int,
    attn_pool_size: int,
    model,
    compression_method: str,
    orthbicrit_alpha: float,
    orthbicrit_beta: float,
    recent_aware: bool,
    recent_summary_k: int,
    diagnostics_out: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Orthogonal bicriteria residual greedy on old-region K/V (optional protected-mean residualization).
    legacy-coreset-orthbicrit-frame: frame summaries + expand + orthbicrit trim.
    In-method pooled V-norm safety fill only; no silent fallback.
    """
    device = key_states_to_compress.device
    total_length = key_states_to_compress.shape[2]
    total_tokens_to_keep = int(round(compress_frame_num * token_per_frame))
    cm = str(compression_method)

    if cm == "legacy-coreset-orthbicrit-frame":
        force_frame = True
    elif cm in (
        "legacy-coreset-orthbicrit-d2",
        "legacy-coreset-orthbicrit-recent",
    ):
        force_frame = False
    else:
        raise ValueError(f"Unknown orthbicrit method: {compression_method}")

    if cm == "legacy-coreset-orthbicrit-recent":
        recent_eff = True
    else:
        recent_eff = bool(recent_aware)

    query_length = 0
    if tar_ratio > 0:
        query_frame = int(current_frame_num * query_ratio)
        query_length = int(query_frame * token_per_frame)
    query_length = min(query_length, total_length)
    split = total_length - query_length
    if split < 0:
        split = 0
        query_length = total_length

    if query_length > 0:
        protected = torch.arange(split, total_length, device=device, dtype=torch.long)
    else:
        protected = torch.zeros(0, dtype=torch.long, device=device)

    old_budget = max(0, total_tokens_to_keep - int(protected.numel()))
    if old_budget > split:
        raise RuntimeError("orthbicrit: old_budget exceeds non-protected length")

    val_norm_score = _val_norm_pooled(
        value_states_to_compress,
        token_per_frame,
        avg_pooling_nd,
        attn_pool_size,
        model,
    )

    alpha = min(1.0, max(0.0, float(orthbicrit_alpha)))
    beta = min(1.0, max(0.0, float(orthbicrit_beta)))

    out_rows: List[torch.Tensor] = []
    any_direct_safety_fill = False
    diag_before_fill = 0
    diag_final = 0
    feat_level_last = "token"
    last_meta: Dict[str, Any] = {
        "orth_residual_k_mean": 0.0,
        "orth_residual_v_mean": 0.0,
        "orth_residual_joint_mean": 0.0,
        "basis_rank_k": 0,
        "basis_rank_v": 0,
    }

    for h in range(num_heads):
        key_h = key_states_to_compress[0, h, :, :].float()
        value_h = value_states_to_compress[0, h, :, :].float()

        need_means = recent_eff and protected.numel() > 0
        if need_means:
            k_mean_prot, v_mean_prot = _protected_kv_means(
                key_h, value_h, protected, recent_summary_k
            )
        else:
            k_mean_prot, v_mean_prot = None, None

        if old_budget == 0:
            pn = int(protected.numel())
            if pn == total_tokens_to_keep:
                row = protected
            elif pn > total_tokens_to_keep:
                sc = val_norm_score[h, protected]
                row = protected[
                    torch.topk(sc, k=total_tokens_to_keep, largest=True).indices
                ].sort().values
            else:
                raise RuntimeError("orthbicrit: protected shorter than budget")
            diag_before_fill = 0
            diag_final = 0
            out_rows.append(row)
            continue
        if split <= 0:
            raise RuntimeError("no old tokens for orthbicrit coreset")

        x_old = key_h[:split]
        v_old = value_h[:split]
        Kp, Vp = _bicriteria_apply_residual(
            x_old, v_old, k_mean_prot, v_mean_prot, recent_eff
        )
        u_old = val_norm_score[h, :split].float()

        aligned = token_per_frame > 1 and split % token_per_frame == 0
        if force_frame:
            if not aligned:
                raise RuntimeError(
                    "legacy-coreset-orthbicrit-frame requires vision length divisible by "
                    "token_per_frame (frame-aligned old region)."
                )
            use_frame_path = True
            feat_level = "frame"
        else:
            use_frame_path = aligned
            feat_level = "frame" if use_frame_path else "token"
        feat_level_last = feat_level

        if use_frame_path:
            F_old = split // token_per_frame
            n_frames = min(F_old, int(math.ceil(old_budget / float(token_per_frame))))
            k_old_f = Kp.reshape(F_old, token_per_frame, -1).mean(dim=1)
            v_old_f = Vp.reshape(F_old, token_per_frame, -1).mean(dim=1)
            u_f = u_old.reshape(F_old, token_per_frame).mean(dim=1)
            idx_f, meta = select_orthbicrit_greedy_indices(
                k_old_f, v_old_f, u_f, n_frames, alpha, beta
            )
            flat: List[int] = []
            for f in idx_f.detach().cpu().tolist():
                base = int(f) * token_per_frame
                flat.extend(range(base, base + token_per_frame))
            abs_cand = torch.tensor(flat, device=device, dtype=torch.long)
            abs_cand = torch.unique(abs_cand, sorted=True)
            if int(abs_cand.numel()) > old_budget:
                sel_old = _trim_with_orthbicrit(
                    Kp, Vp, u_old, abs_cand, old_budget, alpha, beta
                )
            else:
                sel_old = abs_cand
        else:
            idx_t, meta = select_orthbicrit_greedy_indices(
                Kp, Vp, u_old, old_budget, alpha, beta
            )
            sel_old = idx_t

        last_meta = meta
        selected_old_token_count_before_fill = int(sel_old.numel())

        if selected_old_token_count_before_fill < old_budget:
            n_pre_vfill = selected_old_token_count_before_fill
            mask_old = torch.ones(split, dtype=torch.bool, device=device)
            mask_old[sel_old] = False
            rest = mask_old.nonzero(as_tuple=False).squeeze(-1)
            if rest.numel() > 0:
                srest = val_norm_score[h, rest]
                need = old_budget - int(sel_old.numel())
                extra = rest[
                    torch.topk(srest, k=min(need, int(rest.numel())), largest=True).indices
                ]
                sel_old = torch.unique(torch.cat([sel_old, extra]), sorted=True)
            if int(sel_old.numel()) > n_pre_vfill:
                any_direct_safety_fill = True

        if int(sel_old.numel()) < old_budget:
            raise RuntimeError("orthbicrit coreset could not fill old_budget")

        if int(sel_old.numel()) > old_budget:
            K_sub = Kp[sel_old]
            V_sub = Vp[sel_old]
            u_sub = u_old[sel_old]
            sub_rel, _ = select_orthbicrit_greedy_indices(
                K_sub, V_sub, u_sub, old_budget, alpha, beta
            )
            sel_old = sel_old[sub_rel.long()].sort().values

        selected_old_token_count_final = int(sel_old.numel())

        row = torch.unique(torch.cat([sel_old, protected]), sorted=True)
        if int(row.numel()) != total_tokens_to_keep:
            raise RuntimeError("orthbicrit coreset row length mismatch")
        out_rows.append(row)

        diag_before_fill = selected_old_token_count_before_fill
        diag_final = selected_old_token_count_final

    if diagnostics_out is not None:
        diagnostics_out.clear()
        diagnostics_out.update(
            {
                "method": str(compression_method),
                "orthbicrit_alpha": float(alpha),
                "orthbicrit_beta": float(beta),
                "selected_old_token_count_before_fill": int(diag_before_fill),
                "selected_old_token_count_final": int(diag_final),
                "used_direct_safety_fill": bool(any_direct_safety_fill),
                "orth_residual_k_mean": float(last_meta.get("orth_residual_k_mean", 0.0)),
                "orth_residual_v_mean": float(last_meta.get("orth_residual_v_mean", 0.0)),
                "orth_residual_joint_mean": float(
                    last_meta.get("orth_residual_joint_mean", 0.0)
                ),
                "feature_level": feat_level_last,
                "basis_rank_k": int(last_meta.get("basis_rank_k", 0)),
                "basis_rank_v": int(last_meta.get("basis_rank_v", 0)),
            }
        )

    return torch.stack(out_rows, dim=0)


def _probe_build_x_features(
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    probe_feature_mode: str,
    recent_aware: bool,
    k_mean_prot: Optional[torch.Tensor],
    v_mean_prot: Optional[torch.Tensor],
) -> torch.Tensor:
    if probe_feature_mode == "k":
        f = k_rows
        if recent_aware and k_mean_prot is not None:
            f = f - k_mean_prot.unsqueeze(0)
        return f
    if probe_feature_mode == "v":
        f = v_rows
        if recent_aware and v_mean_prot is not None:
            f = f - v_mean_prot.unsqueeze(0)
        return f
    if probe_feature_mode == "kv_joint":
        if recent_aware and k_mean_prot is not None and v_mean_prot is not None:
            return torch.cat(
                [k_rows - k_mean_prot.unsqueeze(0), v_rows - v_mean_prot.unsqueeze(0)], dim=-1
            )
        return torch.cat([k_rows, v_rows], dim=-1)
    raise ValueError(f"Unknown probe_feature_mode: {probe_feature_mode}")


def _l2_normalize_probe_rows(X: torch.Tensor) -> torch.Tensor:
    Xf = X.float()
    return Xf / Xf.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def _probe_equal_temporal_bin_edges(F: int, num_bins: int) -> List[Tuple[int, int]]:
    """Partition F consecutive frames into num_bins contiguous index ranges (nearly equal size)."""
    F = int(F)
    num_bins = int(min(max(num_bins, 1), F))
    if F <= 0:
        return []
    base = F // num_bins
    extra = F % num_bins
    out: List[Tuple[int, int]] = []
    start = 0
    for b in range(num_bins):
        length = base + (1 if b < extra else 0)
        end = start + length
        out.append((start, end))
        start = end
    return out


def _probe_frame_summary_matrix(
    x_old: torch.Tensor,
    v_old: torch.Tensor,
    token_per_frame: int,
    split: int,
    probe_feature_mode: str,
) -> torch.Tensor:
    """
    Frame-level summaries only: one row per full frame (tpf tokens) plus optional one row for
    a trailing partial chunk. Order is temporal (early → late).
    """
    tpf = max(1, int(token_per_frame))
    n_full = split // tpf
    rem = split % tpf
    rows: List[torch.Tensor] = []
    for f in range(n_full):
        sl = x_old[f * tpf : (f + 1) * tpf]
        sv = v_old[f * tpf : (f + 1) * tpf]
        xf = _probe_build_x_features(sl, sv, probe_feature_mode, False, None, None).mean(dim=0)
        rows.append(xf)
    if rem > 0:
        sl = x_old[n_full * tpf : split]
        sv = v_old[n_full * tpf : split]
        xf = _probe_build_x_features(sl, sv, probe_feature_mode, False, None, None).mean(dim=0)
        rows.append(xf)
    if not rows:
        return torch.zeros(0, 0, device=x_old.device, dtype=x_old.dtype)
    return torch.stack(rows, dim=0)


def _kmeans_centroid_matrix(
    X: torch.Tensor, m: int, num_iters: int, gen: torch.Generator
) -> torch.Tensor:
    """Lloyd centroids, shape [m, D]."""
    device = X.device
    N = int(X.shape[0])
    m = int(min(max(m, 1), N))
    Xf = X.float()
    perm = torch.randperm(N, generator=gen, device=device)[:m]
    C = Xf[perm].clone()
    for _ in range(max(1, int(num_iters))):
        dist = (Xf.unsqueeze(1) - C.unsqueeze(0)).pow(2).sum(dim=-1)
        assign = dist.argmin(dim=1)
        new_c: List[torch.Tensor] = []
        for j in range(m):
            mask = assign == j
            if mask.any():
                new_c.append(Xf[mask].mean(dim=0))
            else:
                new_c.append(C[j])
        C = torch.stack(new_c, dim=0)
    return C


def _build_probe_directions(
    probe_mode: str,
    X_centroid_pool: torch.Tensor,
    feat_dim: int,
    m: int,
    num_iters: int,
    device: torch.device,
    gen: torch.Generator,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Raw probe directions [m, D] (L2-normalized by caller for cosine).
    Second return: diagnostics fragment (probe_source_pool_size, actual_num_probe_vectors_used,
    probe_mode_detail).
    """
    m = int(max(m, 1))
    if probe_mode == "random":
        P = torch.randn(m, feat_dim, generator=gen, device=device, dtype=torch.float32)
        return P, {
            "probe_source_pool_size": 0,
            "actual_num_probe_vectors_used": m,
            "probe_mode_detail": "random:std_normal_iid",
        }
    if probe_mode == "centroid":
        Np = int(X_centroid_pool.shape[0])
        if Np == 0:
            P = torch.randn(m, feat_dim, generator=gen, device=device, dtype=torch.float32)
            return P, {
                "probe_source_pool_size": 0,
                "actual_num_probe_vectors_used": m,
                "probe_mode_detail": "centroid:empty_pool_fallback_random",
            }
        mc = min(m, Np)
        C = _kmeans_centroid_matrix(X_centroid_pool, mc, num_iters, gen)
        pad_n = m - mc
        if pad_n > 0:
            pad = torch.randn(pad_n, feat_dim, generator=gen, device=device, dtype=torch.float32)
            C = torch.cat([C, pad], dim=0)
        detail = f"centroid:kmeans_lloyd_pool={Np},lloyd_centroids={mc},random_pad={pad_n}"
        return C, {
            "probe_source_pool_size": Np,
            "actual_num_probe_vectors_used": m,
            "probe_mode_detail": detail,
        }
    if probe_mode == "temporal_centroid_bank":
        Xf = X_centroid_pool.float()
        F = int(Xf.shape[0])
        D = int(Xf.shape[1]) if F > 0 else int(feat_dim)
        if F == 0:
            P = torch.randn(m, D, generator=gen, device=device, dtype=torch.float32)
            return P, {
                "probe_source_pool_size": 0,
                "actual_num_probe_vectors_used": m,
                "probe_mode_detail": "temporal_centroid_bank:empty_frame_pool_fallback_random",
            }
        structured = min(m, F)
        bins = _probe_equal_temporal_bin_edges(F, structured)
        cents = torch.stack([Xf[s:e].mean(dim=0) for s, e in bins], dim=0)
        pad_n = m - structured
        if pad_n > 0:
            pad = torch.randn(pad_n, D, generator=gen, device=device, dtype=torch.float32)
            cents = torch.cat([cents, pad], dim=0)
        detail = (
            f"temporal_centroid_bank:pool_frames={F},temporal_bins={structured},"
            f"random_pad={pad_n},bins_equal_time"
        )
        return cents, {
            "probe_source_pool_size": F,
            "actual_num_probe_vectors_used": m,
            "probe_mode_detail": detail,
        }
    raise ValueError(f"Unknown probe_mode: {probe_mode}")


def _compute_probe_response_matrix(X_unit: torch.Tensor, P_unit: torch.Tensor) -> torch.Tensor:
    """R [N, m], cosine similarity per probe (unit rows)."""
    return X_unit @ P_unit.T


def _probe_select_indices(
    key_states_to_compress: torch.Tensor,
    value_states_to_compress: torch.Tensor,
    num_heads: int,
    token_per_frame: int,
    current_frame_num: int,
    compress_frame_num: int,
    tar_ratio: float,
    query_ratio: float,
    avg_pooling_nd: int,
    attn_pool_size: int,
    model,
    coreset_num_iters: int,
    compression_method: str,
    probe_mode: str,
    probe_feature_mode: str,
    num_probe_vectors: int,
    recent_aware: bool,
    recent_summary_k: int,
    direct_trim_mode: str = "farthest",
    diagnostics_out: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Query-agnostic probe coreset: fixed surrogate probes (random, k-means centroids on the
    pool, or temporal bin centroids on frame summaries only), cosine responses R_i, then
    k-means or D² coreset on rows R_i. No question text or question embeddings.
    In-method pooled V-norm safety fill only.
    """
    device = key_states_to_compress.device
    total_length = key_states_to_compress.shape[2]
    total_tokens_to_keep = int(round(compress_frame_num * token_per_frame))

    cm = str(compression_method)
    if cm == "legacy-coreset-probe":
        selector_mode = "kmeans"
    elif cm in ("legacy-coreset-probe-d2", "legacy-coreset-probe-recent"):
        selector_mode = "d2"
    else:
        raise ValueError(f"Unknown probe coreset method: {compression_method}")

    if cm == "legacy-coreset-probe-recent":
        recent_aware_eff = True
    else:
        recent_aware_eff = bool(recent_aware)

    query_length = 0
    if tar_ratio > 0:
        query_frame = int(current_frame_num * query_ratio)
        query_length = int(query_frame * token_per_frame)
    query_length = min(query_length, total_length)
    split = total_length - query_length
    if split < 0:
        split = 0
        query_length = total_length

    if query_length > 0:
        protected = torch.arange(split, total_length, device=device, dtype=torch.long)
    else:
        protected = torch.zeros(0, dtype=torch.long, device=device)

    old_budget = max(0, total_tokens_to_keep - int(protected.numel()))
    if old_budget > split:
        raise RuntimeError("legacy-coreset-probe: old_budget exceeds non-protected length")

    val_norm_score = _val_norm_pooled(
        value_states_to_compress,
        token_per_frame,
        avg_pooling_nd,
        attn_pool_size,
        model,
    )

    pm = str(probe_mode)
    pfm = str(probe_feature_mode)
    m_probes = int(max(1, num_probe_vectors))
    bmode = str(direct_trim_mode)

    out_rows: List[torch.Tensor] = []
    last_probe_res: Optional[float] = None
    last_cov: Optional[float] = None
    any_direct_safety_fill = False
    diag_before_fill = 0
    diag_final = 0
    feat_level_last = "token"
    last_probe_meta: Dict[str, Any] = {
        "probe_source_pool_size": 0,
        "actual_num_probe_vectors_used": m_probes,
        "probe_mode_detail": "unset",
    }

    for h in range(num_heads):
        key_h = key_states_to_compress[0, h, :, :].float()
        value_h = value_states_to_compress[0, h, :, :].float()

        need_means = recent_aware_eff and protected.numel() > 0
        if need_means:
            k_mean_prot, v_mean_prot = _protected_kv_means(
                key_h, value_h, protected, recent_summary_k
            )
        else:
            k_mean_prot, v_mean_prot = None, None

        if old_budget == 0:
            pn = int(protected.numel())
            if pn == total_tokens_to_keep:
                row = protected
            elif pn > total_tokens_to_keep:
                sc = val_norm_score[h, protected]
                row = protected[
                    torch.topk(sc, k=total_tokens_to_keep, largest=True).indices
                ].sort().values
            else:
                raise RuntimeError("legacy-coreset-probe: protected shorter than budget")
            diag_before_fill = 0
            diag_final = 0
            out_rows.append(row)
            continue
        if split <= 0:
            raise RuntimeError("no old tokens for probe coreset")

        x_old = key_h[:split]
        v_old = value_h[:split]

        gen = torch.Generator(device=device)
        gen.manual_seed(int(_PROBE_RANDOM_GLOBAL_SEED + h * 100_003 + split * 31))

        X_f_pool = _probe_frame_summary_matrix(
            x_old, v_old, token_per_frame, split, pfm
        )

        use_frame_path = token_per_frame > 1 and split % token_per_frame == 0
        feat_level = "frame" if use_frame_path else "token"
        feat_level_last = feat_level

        if use_frame_path:
            F_old = split // token_per_frame
            n_frames = min(F_old, int(math.ceil(old_budget / float(token_per_frame))))
            k_old_f = x_old.reshape(F_old, token_per_frame, -1).mean(dim=1)
            v_old_f = v_old.reshape(F_old, token_per_frame, -1).mean(dim=1)
            if pm == "temporal_centroid_bank":
                fd = (
                    int(X_f_pool.shape[1])
                    if int(X_f_pool.shape[0]) > 0
                    else int(k_old_f.shape[-1]) * (2 if pfm == "kv_joint" else 1)
                )
                P_raw, probe_meta = _build_probe_directions(
                    pm, X_f_pool.float(), fd, m_probes, coreset_num_iters, device, gen
                )
            else:
                X_pool = _probe_build_x_features(
                    k_old_f, v_old_f, pfm, False, None, None
                )
                feat_dim = int(X_pool.shape[1])
                P_raw, probe_meta = _build_probe_directions(
                    pm, X_pool.float(), feat_dim, m_probes, coreset_num_iters, device, gen
                )
            last_probe_meta = probe_meta
            P_unit = _l2_normalize_probe_rows(P_raw)
            Xf = _probe_build_x_features(
                k_old_f, v_old_f, pfm, recent_aware_eff, k_mean_prot, v_mean_prot
            )
            X_unit = _l2_normalize_probe_rows(Xf)
            R = _compute_probe_response_matrix(X_unit, P_unit)
            idx_f = _direct_coreset_pick_indices(
                R.float(), n_frames, selector_mode, coreset_num_iters
            )
            flat: List[int] = []
            for f in idx_f.detach().cpu().tolist():
                base = int(f) * token_per_frame
                flat.extend(range(base, base + token_per_frame))
            abs_cand = torch.tensor(flat, device=device, dtype=torch.long)
            abs_cand = torch.unique(abs_cand, sorted=True)
            tok_k = x_old[abs_cand]
            tok_v = v_old[abs_cand]
            tok_X = _probe_build_x_features(
                tok_k, tok_v, pfm, recent_aware_eff, k_mean_prot, v_mean_prot
            )
            R_tok = _compute_probe_response_matrix(
                _l2_normalize_probe_rows(tok_X), P_unit
            )
            if int(abs_cand.numel()) > old_budget:
                if bmode == "farthest":
                    sel_old = _trim_expanded_token_indices_farthest(
                        R_tok.float(), abs_cand, old_budget
                    )
                elif bmode == "first":
                    sel_old = _trim_expanded_token_indices_first(abs_cand, old_budget)
                else:
                    raise ValueError(f"Unknown direct_trim_mode: {bmode}")
            else:
                sel_old = abs_cand
        else:
            if pm == "temporal_centroid_bank":
                fd = (
                    int(X_f_pool.shape[1])
                    if int(X_f_pool.shape[0]) > 0
                    else int(x_old.shape[-1]) * (2 if pfm == "kv_joint" else 1)
                )
                P_raw, probe_meta = _build_probe_directions(
                    pm, X_f_pool.float(), fd, m_probes, coreset_num_iters, device, gen
                )
            else:
                X_pool = _probe_build_x_features(x_old, v_old, pfm, False, None, None)
                feat_dim = int(X_pool.shape[1])
                P_raw, probe_meta = _build_probe_directions(
                    pm, X_pool.float(), feat_dim, m_probes, coreset_num_iters, device, gen
                )
            last_probe_meta = probe_meta
            P_unit = _l2_normalize_probe_rows(P_raw)
            X_rows = _probe_build_x_features(
                x_old, v_old, pfm, recent_aware_eff, k_mean_prot, v_mean_prot
            )
            X_unit = _l2_normalize_probe_rows(X_rows)
            R = _compute_probe_response_matrix(X_unit, P_unit)
            sel_old = _direct_coreset_pick_indices(
                R.float(), old_budget, selector_mode, coreset_num_iters
            )

        selected_old_token_count_before_fill = int(sel_old.numel())

        if selected_old_token_count_before_fill < old_budget:
            n_pre_vfill = selected_old_token_count_before_fill
            mask_old = torch.ones(split, dtype=torch.bool, device=device)
            mask_old[sel_old] = False
            rest = mask_old.nonzero(as_tuple=False).squeeze(-1)
            if rest.numel() > 0:
                srest = val_norm_score[h, rest]
                need = old_budget - int(sel_old.numel())
                extra = rest[
                    torch.topk(srest, k=min(need, int(rest.numel())), largest=True).indices
                ]
                sel_old = torch.unique(torch.cat([sel_old, extra]), sorted=True)
            if int(sel_old.numel()) > n_pre_vfill:
                any_direct_safety_fill = True

        if int(sel_old.numel()) < old_budget:
            raise RuntimeError("probe coreset could not fill old_budget")

        if int(sel_old.numel()) > old_budget:
            tok_X2 = _probe_build_x_features(
                x_old[sel_old], v_old[sel_old], pfm, recent_aware_eff, k_mean_prot, v_mean_prot
            )
            R_sub = _compute_probe_response_matrix(
                _l2_normalize_probe_rows(tok_X2), P_unit
            )
            if bmode == "farthest":
                sub = select_coreset_indices_d2(R_sub.float(), old_budget)
                sel_old = sel_old[sub.long()].sort().values
            elif bmode == "first":
                sel_old = sel_old.sort().values[:old_budget]
            else:
                raise ValueError(f"Unknown direct_trim_mode: {bmode}")

        selected_old_token_count_final = int(sel_old.numel())

        R_all = _compute_probe_response_matrix(
            _l2_normalize_probe_rows(
                _probe_build_x_features(
                    x_old, v_old, pfm, recent_aware_eff, k_mean_prot, v_mean_prot
                )
            ),
            P_unit,
        )
        R_sel = R_all[sel_old.long()]
        r_sum_all = R_all.sum(dim=0)
        r_sum_sel = R_sel.sum(dim=0)
        last_probe_res = float((r_sum_all - r_sum_sel).norm().item())

        v_sum_all = v_old.sum(dim=0)
        v_sum_sel = v_old[sel_old.long()].sum(dim=0)
        last_cov = float((v_sum_all - v_sum_sel).norm().item())

        row = torch.unique(torch.cat([sel_old, protected]), sorted=True)
        if int(row.numel()) != total_tokens_to_keep:
            raise RuntimeError("probe coreset row length mismatch")
        out_rows.append(row)

        diag_before_fill = selected_old_token_count_before_fill
        diag_final = selected_old_token_count_final

    if diagnostics_out is not None:
        diagnostics_out.clear()
        diagnostics_out.update(
            {
                "method": str(compression_method),
                "probe_mode": pm,
                "probe_feature_mode": pfm,
                "num_probe_vectors": int(m_probes),
                "actual_num_probe_vectors_used": int(
                    last_probe_meta.get("actual_num_probe_vectors_used", m_probes)
                ),
                "probe_source_pool_size": int(last_probe_meta.get("probe_source_pool_size", 0)),
                "probe_mode_detail": str(last_probe_meta.get("probe_mode_detail", "")),
                "selected_old_token_count_before_fill": int(diag_before_fill),
                "selected_old_token_count_final": int(diag_final),
                "used_direct_safety_fill": bool(any_direct_safety_fill),
                "probe_response_residual_l2": last_probe_res,
                "coverage_residual_l2": last_cov,
                "feature_level": feat_level_last,
            }
        )

    return torch.stack(out_rows, dim=0)


def _bicriteria_select_indices(
    key_states_to_compress: torch.Tensor,
    value_states_to_compress: torch.Tensor,
    num_heads: int,
    token_per_frame: int,
    current_frame_num: int,
    compress_frame_num: int,
    tar_ratio: float,
    query_ratio: float,
    avg_pooling_nd: int,
    attn_pool_size: int,
    model,
    coreset_num_iters: int,
    compression_method: str,
    selector_mode: str,
    bicriteria_mode: str,
    bicriteria_alpha: float,
    recent_aware: bool,
    recent_summary_k: int,
    diagnostics_out: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Protected recent + old-region subset via true K/V bicriteria (mixed min-distance or
    separate-merge). One index set prunes K and V together. In-method V-norm safety fill only.
    """
    device = key_states_to_compress.device
    total_length = key_states_to_compress.shape[2]
    total_tokens_to_keep = int(round(compress_frame_num * token_per_frame))

    query_length = 0
    if tar_ratio > 0:
        query_frame = int(current_frame_num * query_ratio)
        query_length = int(query_frame * token_per_frame)
    query_length = min(query_length, total_length)
    split = total_length - query_length
    if split < 0:
        split = 0
        query_length = total_length

    if query_length > 0:
        protected = torch.arange(split, total_length, device=device, dtype=torch.long)
    else:
        protected = torch.zeros(0, dtype=torch.long, device=device)

    old_budget = max(0, total_tokens_to_keep - int(protected.numel()))
    if old_budget > split:
        raise RuntimeError("legacy-coreset-bicriteria: old_budget exceeds non-protected length")

    val_norm_score = _val_norm_pooled(
        value_states_to_compress,
        token_per_frame,
        avg_pooling_nd,
        attn_pool_size,
        model,
    )

    alpha = min(1.0, max(0.0, float(bicriteria_alpha)))
    bmode = str(bicriteria_mode)
    if bmode not in ("mixed", "separate_merge"):
        raise ValueError(f"Unknown bicriteria_mode: {bicriteria_mode}")

    out_rows: List[torch.Tensor] = []
    last_cov_k: Optional[float] = None
    last_cov_v: Optional[float] = None
    last_cov_j: Optional[float] = None
    any_direct_safety_fill = False
    diag_before_fill = 0
    diag_final = 0
    feat_level_last = "token"

    if selector_mode == "kmeans":
        per_space = "kmeans"
    elif selector_mode == "greedy":
        per_space = "greedy"
    else:
        per_space = "d2"

    for h in range(num_heads):
        key_h = key_states_to_compress[0, h, :, :].float()
        value_h = value_states_to_compress[0, h, :, :].float()

        need_means = recent_aware and protected.numel() > 0
        if need_means:
            k_mean_prot, v_mean_prot = _protected_kv_means(
                key_h, value_h, protected, recent_summary_k
            )
        else:
            k_mean_prot, v_mean_prot = None, None

        if old_budget == 0:
            pn = int(protected.numel())
            if pn == total_tokens_to_keep:
                row = protected
            elif pn > total_tokens_to_keep:
                sc = val_norm_score[h, protected]
                row = protected[
                    torch.topk(sc, k=total_tokens_to_keep, largest=True).indices
                ].sort().values
            else:
                raise RuntimeError("legacy-coreset-bicriteria: protected shorter than budget")
            diag_before_fill = 0
            diag_final = 0
            out_rows.append(row)
            continue
        if split <= 0:
            raise RuntimeError("no old tokens for bicriteria coreset")

        x_old = key_h[:split]
        v_old = value_h[:split]
        Kp, Vp = _bicriteria_apply_residual(x_old, v_old, k_mean_prot, v_mean_prot, recent_aware)

        use_frame_path = token_per_frame > 1 and split % token_per_frame == 0
        feat_level = "frame" if use_frame_path else "token"
        feat_level_last = feat_level

        if use_frame_path:
            F_old = split // token_per_frame
            n_frames = min(F_old, int(math.ceil(old_budget / float(token_per_frame))))
            k_old_f = Kp.reshape(F_old, token_per_frame, -1).mean(dim=1)
            v_old_f = Vp.reshape(F_old, token_per_frame, -1).mean(dim=1)
            if bmode == "separate_merge":
                idx_f = select_bicriteria_separate_merge(
                    k_old_f, v_old_f, n_frames, alpha, per_space, coreset_num_iters
                )
            elif selector_mode == "kmeans":
                idx_f = select_bicriteria_kmeans_mixed(
                    k_old_f, v_old_f, n_frames, coreset_num_iters, alpha
                )
            elif selector_mode == "greedy":
                idx_f = select_bicriteria_greedy_mixed(k_old_f, v_old_f, n_frames, alpha)
            else:
                idx_f = select_bicriteria_farthest_mixed(k_old_f, v_old_f, n_frames, alpha)
            flat: List[int] = []
            for f in idx_f.detach().cpu().tolist():
                base = int(f) * token_per_frame
                flat.extend(range(base, base + token_per_frame))
            abs_cand = torch.tensor(flat, device=device, dtype=torch.long)
            abs_cand = torch.unique(abs_cand, sorted=True)
            if int(abs_cand.numel()) > old_budget:
                sel_old = _trim_token_indices_bicriteria_mixed(
                    Kp, Vp, abs_cand, old_budget, alpha
                )
            else:
                sel_old = abs_cand
        else:
            if bmode == "separate_merge":
                sel_old = select_bicriteria_separate_merge(
                    Kp, Vp, old_budget, alpha, per_space, coreset_num_iters
                )
            elif selector_mode == "kmeans":
                sel_old = select_bicriteria_kmeans_mixed(
                    Kp, Vp, old_budget, coreset_num_iters, alpha
                )
            elif selector_mode == "greedy":
                sel_old = select_bicriteria_greedy_mixed(Kp, Vp, old_budget, alpha)
            else:
                sel_old = select_bicriteria_farthest_mixed(Kp, Vp, old_budget, alpha)

        selected_old_token_count_before_fill = int(sel_old.numel())

        if selected_old_token_count_before_fill < old_budget:
            n_pre_vfill = selected_old_token_count_before_fill
            mask_old = torch.ones(split, dtype=torch.bool, device=device)
            mask_old[sel_old] = False
            rest = mask_old.nonzero(as_tuple=False).squeeze(-1)
            if rest.numel() > 0:
                srest = val_norm_score[h, rest]
                need = old_budget - int(sel_old.numel())
                extra = rest[
                    torch.topk(srest, k=min(need, int(rest.numel())), largest=True).indices
                ]
                sel_old = torch.unique(torch.cat([sel_old, extra]), sorted=True)
            if int(sel_old.numel()) > n_pre_vfill:
                any_direct_safety_fill = True

        if int(sel_old.numel()) < old_budget:
            raise RuntimeError("bicriteria coreset could not fill old_budget")

        if int(sel_old.numel()) > old_budget:
            sel_old = _trim_token_indices_bicriteria_mixed(Kp, Vp, sel_old, old_budget, alpha)

        selected_old_token_count_final = int(sel_old.numel())

        k_sum_all = x_old.sum(dim=0)
        v_sum_all = v_old.sum(dim=0)
        k_sum_sel = x_old[sel_old.long()].sum(dim=0)
        v_sum_sel = v_old[sel_old.long()].sum(dim=0)
        last_cov_k = float((k_sum_all - k_sum_sel).norm().item())
        last_cov_v = float((v_sum_all - v_sum_sel).norm().item())
        last_cov_j = float(
            torch.cat([k_sum_all - k_sum_sel, v_sum_all - v_sum_sel], dim=0).norm().item()
        )

        row = torch.unique(torch.cat([sel_old, protected]), sorted=True)
        if int(row.numel()) != total_tokens_to_keep:
            raise RuntimeError("bicriteria coreset row length mismatch")
        out_rows.append(row)

        diag_before_fill = selected_old_token_count_before_fill
        diag_final = selected_old_token_count_final

    if diagnostics_out is not None:
        diagnostics_out.clear()
        diagnostics_out.update(
            {
                "method": str(compression_method),
                "selector_mode": str(selector_mode),
                "bicriteria_alpha": float(alpha),
                "bicriteria_mode": str(bmode),
                "selected_old_token_count_before_fill": int(diag_before_fill),
                "selected_old_token_count_final": int(diag_final),
                "used_direct_safety_fill": bool(any_direct_safety_fill),
                "coverage_residual_k_l2": last_cov_k,
                "coverage_residual_v_l2": last_cov_v,
                "coverage_residual_joint_l2": last_cov_j,
                "feature_level": feat_level_last,
            }
        )

    return torch.stack(out_rows, dim=0)


def _direct_select_indices(
    key_states_to_compress: torch.Tensor,
    value_states_to_compress: torch.Tensor,
    num_heads: int,
    token_per_frame: int,
    current_frame_num: int,
    compress_frame_num: int,
    tar_ratio: float,
    query_ratio: float,
    avg_pooling_nd: int,
    attn_pool_size: int,
    model,
    coreset_num_iters: int,
    selector_mode: str = "d2",
    feature_mode: str = "kv_joint",
    recent_aware: bool = False,
    recent_summary_k: int = -1,
    direct_trim_mode: str = "farthest",
    diagnostics_out: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    this work split: protected recent tokens; old region gets direct coreset (no candidate
    multiplier, no V-norm rerank for the primary selection).

    Direct safety fill (not part of the pure coreset rule, and not a fallback to other
    compression methods): if the selector returns fewer than old_budget distinct old
    indices (e.g. degenerate features), remaining slots are filled by pooled V-norm rank
    among still-unselected old tokens. Fail loudly if that cannot reach old_budget.
    Diagnostics: ``used_direct_safety_fill`` is True iff that in-method V-fill actually
    added tokens on at least one head; pair with ``selected_old_token_count_before_fill``.
    """
    device = key_states_to_compress.device
    total_length = key_states_to_compress.shape[2]
    total_tokens_to_keep = int(round(compress_frame_num * token_per_frame))

    query_length = 0
    if tar_ratio > 0:
        query_frame = int(current_frame_num * query_ratio)
        query_length = int(query_frame * token_per_frame)
    query_length = min(query_length, total_length)
    split = total_length - query_length
    if split < 0:
        split = 0
        query_length = total_length

    if query_length > 0:
        protected = torch.arange(split, total_length, device=device, dtype=torch.long)
    else:
        protected = torch.zeros(0, dtype=torch.long, device=device)

    old_budget = max(0, total_tokens_to_keep - int(protected.numel()))
    if old_budget > split:
        raise RuntimeError("legacy-coreset-direct: old_budget exceeds non-protected length")

    val_norm_score = value_states_to_compress.norm(dim=-1)[0, :, :].clone()

    if avg_pooling_nd > 0:
        pool_size = attn_pool_size
        head_num, _ = val_norm_score.shape
        frame_num = val_norm_score.shape[-1] // token_per_frame

        if avg_pooling_nd == 1:
            avg_pool = torch.nn.AvgPool1d(
                kernel_size=pool_size, stride=1, padding=pool_size // 2
            )
        elif avg_pooling_nd == 2:
            avg_pool = torch.nn.AvgPool2d(
                kernel_size=(pool_size, pool_size),
                stride=(1, 1),
                padding=(pool_size // 2, pool_size // 2),
            )
        elif avg_pooling_nd == 3:
            avg_pool = torch.nn.AvgPool3d(
                kernel_size=(pool_size, pool_size, pool_size),
                stride=(1, 1, 1),
                padding=(pool_size // 2, pool_size // 2, pool_size // 2),
            )
        else:
            raise NotImplementedError("We support 1D - 3D Pooling")

        patch_size = (
            model.config.height_width if hasattr(model.config, "height_width") else 14
        )

        if avg_pooling_nd > 1:
            original_numel = val_norm_score.numel()
            val_norm_reshaped = val_norm_score.reshape(head_num, frame_num, patch_size, -1)

            T = val_norm_reshaped.shape[-3]
            H = val_norm_reshaped.shape[-2]
            W = val_norm_reshaped.shape[-1]

            pool_size_eff = min(pool_size, T, H, W)

            if pool_size_eff % 2 == 0:
                pool_size_eff -= 1

            if pool_size_eff < 2:
                pass
            else:
                if pool_size_eff != pool_size:
                    pool_size = pool_size_eff
                    if avg_pooling_nd == 2:
                        avg_pool = torch.nn.AvgPool2d(
                            kernel_size=(pool_size, pool_size),
                            stride=(1, 1),
                            padding=(pool_size // 2, pool_size // 2),
                        )
                    elif avg_pooling_nd == 3:
                        avg_pool = torch.nn.AvgPool3d(
                            kernel_size=(pool_size, pool_size, pool_size),
                            stride=(1, 1, 1),
                            padding=(pool_size // 2, pool_size // 2, pool_size // 2),
                        )

                val_norm_pooled = avg_pool(val_norm_reshaped)
                val_norm_score = val_norm_pooled.reshape(head_num, -1)
                assert original_numel == val_norm_score.numel()
        else:
            val_norm_score = avg_pool(val_norm_score)

    out_rows: List[torch.Tensor] = []
    last_cov: Optional[float] = None
    recent_aware_effective = bool(
        recent_aware or feature_mode in ("k_recent_residual", "kv_recent_residual")
    )
    any_direct_safety_fill = False
    diag_before_fill = 0
    diag_final = 0

    for h in range(num_heads):
        key_h = key_states_to_compress[0, h, :, :].float()
        value_h = value_states_to_compress[0, h, :, :].float()

        need_means = recent_aware_effective and protected.numel() > 0
        if need_means:
            k_mean_prot, v_mean_prot = _protected_kv_means(
                key_h, value_h, protected, recent_summary_k
            )
        else:
            k_mean_prot, v_mean_prot = None, None

        if old_budget == 0:
            pn = int(protected.numel())
            if pn == total_tokens_to_keep:
                row = protected
            elif pn > total_tokens_to_keep:
                sc = val_norm_score[h, protected]
                row = protected[
                    torch.topk(sc, k=total_tokens_to_keep, largest=True).indices
                ].sort().values
            else:
                raise RuntimeError("legacy-coreset-direct: protected shorter than budget")
            diag_before_fill = 0
            diag_final = 0
            out_rows.append(row)
            continue
        if split <= 0:
            raise RuntimeError("no old tokens for direct coreset")

        x_old = key_h[:split]
        v_old = value_h[:split]

        use_frame_path = token_per_frame > 1 and split % token_per_frame == 0
        if use_frame_path:
            F_old = split // token_per_frame
            n_frames = min(F_old, int(math.ceil(old_budget / float(token_per_frame))))
            k_old_f = x_old.reshape(F_old, token_per_frame, -1).mean(dim=1)
            v_old_f = v_old.reshape(F_old, token_per_frame, -1).mean(dim=1)
            X_f = _build_direct_token_features(
                k_old_f,
                v_old_f,
                feature_mode,
                recent_aware,
                k_mean_prot,
                v_mean_prot,
            )
            idx_f = _direct_coreset_pick_indices(
                X_f.float(), n_frames, selector_mode, coreset_num_iters
            )
            flat: List[int] = []
            for f in idx_f.detach().cpu().tolist():
                base = int(f) * token_per_frame
                flat.extend(range(base, base + token_per_frame))
            abs_cand = torch.tensor(flat, device=device, dtype=torch.long)
            abs_cand = torch.unique(abs_cand, sorted=True)

            tok_feat = _build_direct_token_features(
                x_old[abs_cand],
                v_old[abs_cand],
                feature_mode,
                recent_aware,
                k_mean_prot,
                v_mean_prot,
            )
            if int(abs_cand.numel()) > old_budget:
                if direct_trim_mode == "farthest":
                    sel_old = _trim_expanded_token_indices_farthest(
                        tok_feat, abs_cand, old_budget
                    )
                elif direct_trim_mode == "first":
                    sel_old = _trim_expanded_token_indices_first(abs_cand, old_budget)
                else:
                    raise ValueError(f"Unknown direct_trim_mode: {direct_trim_mode}")
            else:
                sel_old = abs_cand
        else:
            X = _build_direct_token_features(
                x_old, v_old, feature_mode, recent_aware, k_mean_prot, v_mean_prot
            )
            sel_old = _direct_coreset_pick_indices(
                X.float(), old_budget, selector_mode, coreset_num_iters
            )

        selected_old_token_count_before_fill = int(sel_old.numel())

        # Direct safety fill (in-method only): coreset selection above is authoritative;
        # pooled V-norm ranks unselected old tokens to complete the budget (see docstring).
        if selected_old_token_count_before_fill < old_budget:
            n_pre_vfill = selected_old_token_count_before_fill
            mask_old = torch.ones(split, dtype=torch.bool, device=device)
            mask_old[sel_old] = False
            rest = mask_old.nonzero(as_tuple=False).squeeze(-1)
            if rest.numel() > 0:
                srest = val_norm_score[h, rest]
                need = old_budget - int(sel_old.numel())
                extra = rest[
                    torch.topk(srest, k=min(need, int(rest.numel())), largest=True).indices
                ]
                sel_old = torch.unique(torch.cat([sel_old, extra]), sorted=True)
            if int(sel_old.numel()) > n_pre_vfill:
                any_direct_safety_fill = True

        if int(sel_old.numel()) < old_budget:
            raise RuntimeError("direct coreset could not fill old_budget")

        if int(sel_old.numel()) > old_budget:
            tok_f = _build_direct_token_features(
                x_old[sel_old],
                v_old[sel_old],
                feature_mode,
                recent_aware,
                k_mean_prot,
                v_mean_prot,
            ).float()
            if direct_trim_mode == "farthest":
                sub = select_coreset_indices_d2(tok_f, old_budget)
                sel_old = sel_old[sub.long()].sort().values
            elif direct_trim_mode == "first":
                sel_old = sel_old.sort().values[:old_budget]
            else:
                raise ValueError(f"Unknown direct_trim_mode: {direct_trim_mode}")

        selected_old_token_count_final = int(sel_old.numel())

        v_sum_all = v_old.sum(dim=0)
        v_sum_sel = v_old[sel_old.long()].sum(dim=0)
        last_cov = float((v_sum_all - v_sum_sel).norm().item())

        row = torch.unique(torch.cat([sel_old, protected]), sorted=True)
        if int(row.numel()) != total_tokens_to_keep:
            raise RuntimeError("direct coreset row length mismatch")
        out_rows.append(row)

        diag_before_fill = selected_old_token_count_before_fill
        diag_final = selected_old_token_count_final

    if diagnostics_out is not None:
        diagnostics_out.clear()
        diagnostics_out.update(
            {
                "old_budget": int(old_budget),
                "protected_token_count": int(protected.numel()),
                "used_direct_safety_fill": bool(any_direct_safety_fill),
                "selected_old_token_count_before_fill": int(diag_before_fill),
                "selected_old_token_count_final": int(diag_final),
                "selector_mode": selector_mode,
                "feature_mode": feature_mode,
                "recent_aware": bool(recent_aware),
                "recent_aware_effective": recent_aware_effective,
                "coverage_residual_l2": last_cov,
            }
        )

    return torch.stack(out_rows, dim=0)


def _fit_weights_ls_signed(VS: torch.Tensor, v_full: torch.Tensor) -> torch.Tensor:
    """Ordinary least squares: VS @ w ≈ v_full (signed weights, linear hull)."""
    sol = torch.linalg.lstsq(VS, v_full.unsqueeze(-1), rcond=None)
    return sol.solution.squeeze(-1)


def _fit_weights_ls_nonneg(VS: torch.Tensor, v_full: torch.Tensor) -> torch.Tensor:
    """Non-negative least-squares weights: VS @ w ≈ v_full, VS is [D, t], v_full is [D]."""
    sol = torch.linalg.lstsq(VS, v_full.unsqueeze(-1), rcond=None)
    w = sol.solution.squeeze(-1)
    w = torch.clamp(w, min=0.0)
    if float(w.sum()) <= 1e-12:
        w = torch.ones_like(w)
    w = w / (w.sum() + 1e-12)
    return w


def _greedy_linear_coreset_indices(V: torch.Tensor, k: int) -> torch.Tensor:
    """
    Training-free coreset: greedily pick rows of V so their conic hull approximates sum(V).
    V: [N, D], typically row-normalized. Returns sorted LongTensor of length k on V's device.
    """
    device = V.device
    dtype = V.dtype
    N = V.shape[0]
    k = min(max(k, 0), N)
    if k == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    if k == N:
        return torch.arange(N, device=device, dtype=torch.long)

    v_full = V.sum(dim=0)
    r = v_full.clone()
    selected: List[int] = []
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)

    for _ in range(k):
        scores = torch.mv(V, r)
        if selected:
            scores = scores.clone()
            scores[torch.tensor(selected, device=device, dtype=torch.long)] = neg_inf
        j = int(torch.argmax(scores).item())
        if selected and j in selected:
            break
        selected.append(j)
        VS = V[selected].T
        w = _fit_weights_ls_nonneg(VS, v_full)
        r = v_full - VS @ w

    sel_set = set(selected)
    c = 0
    while len(selected) < k and c < N:
        if c not in sel_set:
            selected.append(c)
            sel_set.add(c)
        c += 1

    idx = torch.tensor(sorted(selected[:k]), device=device, dtype=torch.long)
    return idx


def _greedy_linear_coreset_indices_signed(V: torch.Tensor, k: int) -> torch.Tensor:
    """Greedy subset + signed LS residual (linear / non-conic ablation)."""
    device = V.device
    dtype = V.dtype
    N = V.shape[0]
    k = min(max(k, 0), N)
    if k == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    if k == N:
        return torch.arange(N, device=device, dtype=torch.long)

    v_full = V.sum(dim=0)
    r = v_full.clone()
    selected: List[int] = []
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)

    for _ in range(k):
        scores = torch.mv(V, r)
        if selected:
            scores = scores.clone()
            scores[torch.tensor(selected, device=device, dtype=torch.long)] = neg_inf
        j = int(torch.argmax(scores).item())
        if selected and j in selected:
            break
        selected.append(j)
        VS = V[selected].T
        w = _fit_weights_ls_signed(VS, v_full)
        r = v_full - VS @ w

    sel_set = set(selected)
    c = 0
    while len(selected) < k and c < N:
        if c not in sel_set:
            selected.append(c)
            sel_set.add(c)
        c += 1

    return torch.tensor(sorted(selected[:k]), device=device, dtype=torch.long)


def _stream_grad_row_per_frame(
    raw_frames: torch.Tensor,
    q_embed: torch.Tensor,
    probe: nn.Module,
    q_head: nn.Module,
) -> torch.Tensor:
    """
    Stream-style: one scalar loss per frame, gradient w.r.t. probe params only.
    target = q_head(q_embed).detach() — question enters the target, not via similarity to KV.
    """
    device = raw_frames.device
    dtype = torch.float32
    x = raw_frames.to(dtype=dtype, device=device)
    q = q_embed.to(dtype=dtype, device=device)
    probe.train(True)
    q_head.train(True)
    target = q_head(q).detach()
    params = [p for p in probe.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("stream probe has no trainable parameters")

    rows: List[torch.Tensor] = []
    F = x.shape[0]
    for f in range(F):
        probe.zero_grad(set_to_none=True)
        out = probe(x[f : f + 1]).squeeze(0)
        loss = (out - target).pow(2).sum()
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
        rows.append(torch.cat([g.reshape(-1) for g in grads]).detach())
    return torch.stack(rows, dim=0)


def _ensure_stream_modules(stream_modules: Dict, d_frame: int, d_q: int, device: torch.device) -> None:
    """Lazy-init small MLP probe + q head (fixed init for reproducibility across steps)."""
    if stream_modules.get("probe") is not None:
        return
    hid = 64
    torch.manual_seed(0)
    probe = nn.Sequential(
        nn.Linear(d_frame, hid),
        nn.Tanh(),
        nn.Linear(hid, hid),
    ).to(device=device, dtype=torch.float32)
    q_head = nn.Linear(d_q, hid).to(device=device, dtype=torch.float32)
    for p in list(probe.parameters()) + list(q_head.parameters()):
        p.requires_grad_(True)
    stream_modules["probe"] = probe
    stream_modules["q_head"] = q_head


def _greedy_linear_coreset_indices_mean_target(V: torch.Tensor, k: int) -> torch.Tensor:
    """Same greedy NNLS but target v_full = mean(rows) instead of sum (ablation)."""
    device = V.device
    dtype = V.dtype
    N = V.shape[0]
    k = min(max(k, 0), N)
    if k == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    if k == N:
        return torch.arange(N, device=device, dtype=torch.long)

    v_full = V.mean(dim=0)
    r = v_full.clone()
    selected: List[int] = []
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)

    for _ in range(k):
        scores = torch.mv(V, r)
        if selected:
            scores = scores.clone()
            scores[torch.tensor(selected, device=device, dtype=torch.long)] = neg_inf
        j = int(torch.argmax(scores).item())
        if selected and j in selected:
            break
        selected.append(j)
        VS = V[selected].T
        w = _fit_weights_ls_nonneg(VS, v_full)
        r = v_full - VS @ w

    sel_set = set(selected)
    c = 0
    while len(selected) < k and c < N:
        if c not in sel_set:
            selected.append(c)
            sel_set.add(c)
        c += 1

    return torch.tensor(sorted(selected[:k]), device=device, dtype=torch.long)


def _kcenter_frame_indices(V: torch.Tensor, k: int) -> torch.Tensor:
    """Farthest-first on L2 distance in feature space (rows need not be normalized)."""
    device = V.device
    N = V.shape[0]
    k = min(max(k, 0), N)
    if k == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    if k == N:
        return torch.arange(N, device=device, dtype=torch.long)

    centroid = V.mean(dim=0, keepdim=True)
    j0 = int(torch.argmax((V - centroid).norm(dim=1)).item())
    sel = [j0]
    for _ in range(k - 1):
        S = torch.tensor(sel, device=device, dtype=torch.long)
        d2 = (V.unsqueeze(1) - V[S].unsqueeze(0)).pow(2).sum(dim=-1).min(dim=1).values
        d2[sel] = -1.0
        j = int(torch.argmax(d2).item())
        if d2[j] < 0:
            break
        sel.append(j)
    return torch.tensor(sorted(sel[:k]), device=device, dtype=torch.long)


def _random_frame_indices(F: int, k: int, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(int(seed) % (2**31))
    perm = torch.randperm(F, generator=g)[:k]
    return torch.sort(perm.long().to(device)).values


def _raw_frame_features(kh_f: torch.Tensor, vh_f: torch.Tensor, feat_mode: str) -> torch.Tensor:
    """kh_f, vh_f: [H, F, D]. Unnormalized frame features for k-center / inspection."""
    if feat_mode == "kv":
        return torch.cat([kh_f.mean(0), vh_f.mean(0)], dim=-1)
    if feat_mode == "k":
        return kh_f.mean(0)
    if feat_mode == "v":
        return vh_f.mean(0)
    raise ValueError(feat_mode)


def _coreset_frame_features(
    kh_f: torch.Tensor,
    vh_f: torch.Tensor,
    feat_mode: str,
) -> torch.Tensor:
    """Row-normalized features for cosine / greedy coreset."""
    feat = _raw_frame_features(kh_f, vh_f, feat_mode)
    return feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-9)


_CORESET_ABLATIONS = {
    "coreset": ("kv", "greedy_sum"),
    "coreset_conic": ("kv", "greedy_sum"),  # explicit: nonnegative weights / conic hull
    "coreset_linear": ("kv", "greedy_linear_signed"),  # signed LS residual (linear hull)
    "coreset_k": ("k", "greedy_sum"),
    "coreset_v": ("v", "greedy_sum"),
    "coreset_kcenter": ("kv", "kcenter"),
    "coreset_k_kcenter": ("k", "kcenter"),
    "coreset_v_kcenter": ("v", "kcenter"),
    "coreset_random": ("kv", "random"),
    "coreset_mean": ("kv", "greedy_mean"),
}


def process_kv_cache(
    past_key_values,
    model,
    system_size: int,
    inst_size: int,
    token_per_frame: int,
    compress_frame_num: int,
    method: str = "uniform",
    tar_ratio: float = 0.5,
    query_ratio: float = 0.25,
    adaptive_pooling: bool = False,
    is_first_block: bool = False,
    is_last_block: bool = False,
    per_frame: bool = False,
    stream_question_embed: Optional[torch.Tensor] = None,
    stream_modules: Optional[Dict] = None,
    coreset_num_iters: int = 4,
    coreset_on_value: bool = False,
    coreset_candidate_multiplier: float = 1.5,
    coreset_bicrit_alpha: float = 0.5,
    coreset_feature_mode: str = "kv_joint",
    coreset_recent_aware: bool = False,
    coreset_recent_summary_k: int = -1,
    coreset_direct_trim_mode: str = "farthest",
    coreset_bicriteria_alpha: float = 0.5,
    coreset_bicriteria_mode: str = "mixed",
    probe_mode: str = "centroid",
    probe_feature_mode: str = "kv_joint",
    num_probe_vectors: int = 8,
    coreset_orthbicrit_alpha: float = 0.25,
    coreset_orthbicrit_beta: float = 0.0,
    coreset_orthbonus_lambda: float = 1.0,
    coreset_orthbonus_eta: float = 0.25,
    coreset_orthbonus_mode: str = "maxcos",
    coreset_orthbonus_beta: float = 0.0,
    coreset_orthbonus_prefilter_ratio: float = 1.0,
    coreset_orthbonus_score_norm: str = "none",
    compression_layer_mode: str = "all",
    compression_layer_indices: Optional[Any] = None,
) -> Tuple:
    """
    Compress KV cache by keeping only specified number of frames in vision tokens
    
    Args:
        past_key_values: Current KV cache (DynamicCache)
        model: The model instance
        system_size: Size of system tokens
        inst_size: Size of instruction tokens
        token_per_frame: Number of tokens per frame
        compress_frame_num: Number of frames to keep after compression
        method: Compression method ('swa', 'uniform', 'legacy-baseline', 'coreset')
        tar_ratio: Ratio for tar vs other methods (used in 'tar_val')
        query_ratio: Ratio of query frames for tar method
        adaptive_pooling: Whether to use adaptive pooling
        is_first_block: Whether this is the first block
        is_last_block: Whether this is the last block
        per_frame: Whether to select complete frames (for val_norm method)
        
    Returns:
        tuple: (compressed_past_key_values, cap_list)
    """
    
    pooling_func_list = [3, 2, 1, -1]
    pool_size_list = [7, 5, 3, 1]

    if compress_frame_num <= 0:
        return past_key_values, None
    
    # Get current sequence length
    current_seq_len = past_key_values.get_seq_length()
    
    # Calculate vision token range
    vision_start = system_size
    vision_end = current_seq_len
    vision_length = vision_end - vision_start
    current_frame_num = vision_length // token_per_frame
    assert vision_length % token_per_frame == 0, "Vision length must be divisible by token_per_frame"
    
    if current_frame_num <= compress_frame_num:
        return past_key_values, None
    
    # Get key and value states for processing
    num_layers = len(model.model.language_model.layers)
    
    active_layers = resolve_compression_layer_indices(
        num_layers,
        str(compression_layer_mode),
        compression_layer_indices,
    )
    active_layer_set = set(active_layers)
    setattr(
        model,
        "_compression_layer_diag",
        {
            "compression_layer_mode": str(compression_layer_mode),
            "compression_layer_indices_resolved": [int(i) for i in active_layers],
            "num_active_layers": int(len(active_layers)),
        },
    )

    # Process each layer
    for layer_idx in range(num_layers):
        if layer_idx not in active_layer_set:
            continue

        # Layer-Adaptive Pooling
        if adaptive_pooling:
            num_layers = len(model.model.language_model.layers)
            idx = layer_idx // max(1, (num_layers // len(pooling_func_list)))
            #idx = layer_idx // (model.config.num_hidden_layers // len(pooling_func_list))
            avg_pooling_nd = pooling_func_list[idx]
            attn_pool_size = pool_size_list[idx]
        else:
            avg_pooling_nd = -1
            # Unused when avg_pooling_nd < 0; must exist for legacy-baseline* helper calls.
            attn_pool_size = pool_size_list[0]

        key_states = past_key_values.layers[layer_idx].keys
        value_states = past_key_values.layers[layer_idx].values
        bsz, num_heads, q_len, head_dim = key_states.shape
        
        # Extract vision tokens to compress
        key_states_to_compress = key_states[:, :, system_size:, :]
        value_states_to_compress = value_states[:, :, system_size:, :]
        
        # Select compression method
        # Calculate total tokens to keep
        total_tokens_to_keep = int(round(compress_frame_num * token_per_frame))
        
        if method == "swa":
            # SWA: Keep the most recent tokens
            start_idx = key_states_to_compress.shape[2] - total_tokens_to_keep
            all_indices = torch.arange(start_idx, key_states_to_compress.shape[2], device=key_states.device)
            all_indices = all_indices.unsqueeze(0).expand(num_heads, -1)
            
        elif method == "uniform":
            # Uniform: Sample tokens uniformly across the sequence
            total_tokens = key_states_to_compress.shape[2]
            step = total_tokens / total_tokens_to_keep
            selected_token_indices = [int(i * step) for i in range(total_tokens_to_keep)]
            all_indices = torch.tensor(selected_token_indices, device=key_states.device)
            all_indices = all_indices.unsqueeze(0).expand(num_heads, -1)
                        
        elif method == "legacy-baseline":
            all_indices = _baseline_select_indices(
                key_states_to_compress,
                value_states_to_compress,
                num_heads,
                token_per_frame,
                current_frame_num,
                compress_frame_num,
                tar_ratio,
                query_ratio,
                avg_pooling_nd,
                attn_pool_size,
                model,
            )

        elif method == "legacy-coreset":
            try:
                all_indices = _coreset_select_indices(
                    key_states_to_compress,
                    value_states_to_compress,
                    num_heads,
                    token_per_frame,
                    current_frame_num,
                    compress_frame_num,
                    tar_ratio,
                    query_ratio,
                    avg_pooling_nd,
                    attn_pool_size,
                    model,
                    coreset_num_iters,
                    coreset_on_value,
                    coreset_candidate_multiplier,
                    selector_mode="kmeans",
                    use_value_rerank=True,
                )
            except Exception:
                all_indices = _baseline_select_indices(
                    key_states_to_compress,
                    value_states_to_compress,
                    num_heads,
                    token_per_frame,
                    current_frame_num,
                    compress_frame_num,
                    tar_ratio,
                    query_ratio,
                    avg_pooling_nd,
                    attn_pool_size,
                    model,
                )

        elif method == "legacy-coreset-d2":
            try:
                all_indices = _coreset_select_indices(
                    key_states_to_compress,
                    value_states_to_compress,
                    num_heads,
                    token_per_frame,
                    current_frame_num,
                    compress_frame_num,
                    tar_ratio,
                    query_ratio,
                    avg_pooling_nd,
                    attn_pool_size,
                    model,
                    coreset_num_iters,
                    coreset_on_value,
                    coreset_candidate_multiplier,
                    selector_mode="d2",
                    use_value_rerank=False,
                )
            except Exception:
                all_indices = _baseline_select_indices(
                    key_states_to_compress,
                    value_states_to_compress,
                    num_heads,
                    token_per_frame,
                    current_frame_num,
                    compress_frame_num,
                    tar_ratio,
                    query_ratio,
                    avg_pooling_nd,
                    attn_pool_size,
                    model,
                )

        elif method == "legacy-coreset-greedy":
            try:
                all_indices = _coreset_select_indices(
                    key_states_to_compress,
                    value_states_to_compress,
                    num_heads,
                    token_per_frame,
                    current_frame_num,
                    compress_frame_num,
                    tar_ratio,
                    query_ratio,
                    avg_pooling_nd,
                    attn_pool_size,
                    model,
                    coreset_num_iters,
                    coreset_on_value,
                    coreset_candidate_multiplier,
                    selector_mode="greedy",
                    use_value_rerank=True,
                )
            except Exception:
                all_indices = _baseline_select_indices(
                    key_states_to_compress,
                    value_states_to_compress,
                    num_heads,
                    token_per_frame,
                    current_frame_num,
                    compress_frame_num,
                    tar_ratio,
                    query_ratio,
                    avg_pooling_nd,
                    attn_pool_size,
                    model,
                )

        elif method == "legacy-coreset-d2v":
            try:
                all_indices = _coreset_select_indices(
                    key_states_to_compress,
                    value_states_to_compress,
                    num_heads,
                    token_per_frame,
                    current_frame_num,
                    compress_frame_num,
                    tar_ratio,
                    query_ratio,
                    avg_pooling_nd,
                    attn_pool_size,
                    model,
                    coreset_num_iters,
                    coreset_on_value,
                    coreset_candidate_multiplier,
                    selector_mode="d2",
                    use_value_rerank=True,
                )
            except Exception:
                all_indices = _baseline_select_indices(
                    key_states_to_compress,
                    value_states_to_compress,
                    num_heads,
                    token_per_frame,
                    current_frame_num,
                    compress_frame_num,
                    tar_ratio,
                    query_ratio,
                    avg_pooling_nd,
                    attn_pool_size,
                    model,
                )

        elif method == "legacy-coreset-kvjoint":
            try:
                all_indices = _coreset_select_indices(
                    key_states_to_compress,
                    value_states_to_compress,
                    num_heads,
                    token_per_frame,
                    current_frame_num,
                    compress_frame_num,
                    tar_ratio,
                    query_ratio,
                    avg_pooling_nd,
                    attn_pool_size,
                    model,
                    coreset_num_iters,
                    coreset_on_value,
                    coreset_candidate_multiplier,
                    selector_mode="kmeans",
                    use_value_rerank=True,
                    frame_feature="kv_joint",
                    rerank_mode="value",
                )
            except Exception:
                all_indices = _baseline_select_indices(
                    key_states_to_compress,
                    value_states_to_compress,
                    num_heads,
                    token_per_frame,
                    current_frame_num,
                    compress_frame_num,
                    tar_ratio,
                    query_ratio,
                    avg_pooling_nd,
                    attn_pool_size,
                    model,
                )

        elif method == "legacy-coreset-bicrit":
            try:
                all_indices = _coreset_select_indices(
                    key_states_to_compress,
                    value_states_to_compress,
                    num_heads,
                    token_per_frame,
                    current_frame_num,
                    compress_frame_num,
                    tar_ratio,
                    query_ratio,
                    avg_pooling_nd,
                    attn_pool_size,
                    model,
                    coreset_num_iters,
                    coreset_on_value,
                    coreset_candidate_multiplier,
                    selector_mode="kmeans",
                    use_value_rerank=True,
                    frame_feature="k",
                    rerank_mode="bicrit",
                    bicrit_alpha=coreset_bicrit_alpha,
                )
            except Exception:
                all_indices = _baseline_select_indices(
                    key_states_to_compress,
                    value_states_to_compress,
                    num_heads,
                    token_per_frame,
                    current_frame_num,
                    compress_frame_num,
                    tar_ratio,
                    query_ratio,
                    avg_pooling_nd,
                    attn_pool_size,
                    model,
                )

        elif method == "legacy-coreset-kvjoint-onv":
            try:
                all_indices = _coreset_select_indices(
                    key_states_to_compress,
                    value_states_to_compress,
                    num_heads,
                    token_per_frame,
                    current_frame_num,
                    compress_frame_num,
                    tar_ratio,
                    query_ratio,
                    avg_pooling_nd,
                    attn_pool_size,
                    model,
                    coreset_num_iters,
                    True,
                    coreset_candidate_multiplier,
                    selector_mode="kmeans",
                    use_value_rerank=True,
                    frame_feature="kv_joint",
                    rerank_mode="value",
                )
            except Exception:
                all_indices = _baseline_select_indices(
                    key_states_to_compress,
                    value_states_to_compress,
                    num_heads,
                    token_per_frame,
                    current_frame_num,
                    compress_frame_num,
                    tar_ratio,
                    query_ratio,
                    avg_pooling_nd,
                    attn_pool_size,
                    model,
                )

        elif _is_direct_method(method):
            _rp = _resolve_direct_params(
                method,
                coreset_feature_mode,
                coreset_recent_aware,
                coreset_recent_summary_k,
                coreset_direct_trim_mode,
            )
            _direct_diag: Dict[str, Any] = {}
            all_indices = _direct_select_indices(
                key_states_to_compress,
                value_states_to_compress,
                num_heads,
                token_per_frame,
                current_frame_num,
                compress_frame_num,
                tar_ratio,
                query_ratio,
                avg_pooling_nd,
                attn_pool_size,
                model,
                coreset_num_iters,
                selector_mode=_rp["selector_mode"],
                feature_mode=_rp["feature_mode"],
                recent_aware=_rp["recent_aware"],
                recent_summary_k=_rp["recent_summary_k"],
                direct_trim_mode=_rp["direct_trim_mode"],
                diagnostics_out=_direct_diag,
            )
            setattr(model, "_coreset_direct_diag", _direct_diag)

        elif _is_bicriteria_method(method):
            if method == "legacy-coreset-bicriteria-d2":
                _sel_b = "d2"
                _ra_b = bool(coreset_recent_aware)
            elif method == "legacy-coreset-bicriteria-kmeans":
                _sel_b = "kmeans"
                _ra_b = bool(coreset_recent_aware)
            elif method == "legacy-coreset-bicriteria-greedy":
                _sel_b = "greedy"
                _ra_b = bool(coreset_recent_aware)
            elif method == "legacy-coreset-bicriteria-recent":
                _sel_b = "d2"
                _ra_b = True
            else:
                raise ValueError(f"Unknown bicriteria method: {method}")
            _bic_diag: Dict[str, Any] = {}
            all_indices = _bicriteria_select_indices(
                key_states_to_compress,
                value_states_to_compress,
                num_heads,
                token_per_frame,
                current_frame_num,
                compress_frame_num,
                tar_ratio,
                query_ratio,
                avg_pooling_nd,
                attn_pool_size,
                model,
                coreset_num_iters,
                compression_method=method,
                selector_mode=_sel_b,
                bicriteria_mode=str(coreset_bicriteria_mode),
                bicriteria_alpha=float(coreset_bicriteria_alpha),
                recent_aware=_ra_b,
                recent_summary_k=int(coreset_recent_summary_k),
                diagnostics_out=_bic_diag,
            )
            setattr(model, "_coreset_bicriteria_diag", _bic_diag)

        elif _is_probe_method(method):
            _probe_diag: Dict[str, Any] = {}
            all_indices = _probe_select_indices(
                key_states_to_compress,
                value_states_to_compress,
                num_heads,
                token_per_frame,
                current_frame_num,
                compress_frame_num,
                tar_ratio,
                query_ratio,
                avg_pooling_nd,
                attn_pool_size,
                model,
                coreset_num_iters,
                compression_method=method,
                probe_mode=str(probe_mode),
                probe_feature_mode=str(probe_feature_mode),
                num_probe_vectors=int(num_probe_vectors),
                recent_aware=bool(coreset_recent_aware),
                recent_summary_k=int(coreset_recent_summary_k),
                direct_trim_mode=str(coreset_direct_trim_mode),
                diagnostics_out=_probe_diag,
            )
            setattr(model, "_coreset_probe_diag", _probe_diag)

        elif _is_orthbicrit_method(method):
            _orth_diag: Dict[str, Any] = {}
            all_indices = _orthbicrit_select_indices(
                key_states_to_compress,
                value_states_to_compress,
                num_heads,
                token_per_frame,
                current_frame_num,
                compress_frame_num,
                tar_ratio,
                query_ratio,
                avg_pooling_nd,
                attn_pool_size,
                model,
                compression_method=method,
                orthbicrit_alpha=float(coreset_orthbicrit_alpha),
                orthbicrit_beta=float(coreset_orthbicrit_beta),
                recent_aware=bool(coreset_recent_aware),
                recent_summary_k=int(coreset_recent_summary_k),
                diagnostics_out=_orth_diag,
            )
            setattr(model, "_coreset_orthbicrit_diag", _orth_diag)

        elif is_cords_method(method):
            _ob_diag: Dict[str, Any] = {}
            all_indices = _cords_select_indices(
                key_states_to_compress,
                value_states_to_compress,
                num_heads,
                token_per_frame,
                current_frame_num,
                compress_frame_num,
                tar_ratio,
                query_ratio,
                avg_pooling_nd,
                attn_pool_size,
                model,
                coreset_num_iters,
                compression_method=str(method),
                bicriteria_alpha=float(coreset_bicriteria_alpha),
                orthbonus_lambda=float(coreset_orthbonus_lambda),
                orthbonus_mode=str(coreset_orthbonus_mode),
                orthbonus_eta=float(coreset_orthbonus_eta),
                orthbonus_beta=float(coreset_orthbonus_beta),
                orthbonus_prefilter_ratio=float(coreset_orthbonus_prefilter_ratio),
                orthbonus_score_norm=str(coreset_orthbonus_score_norm),
                recent_aware=bool(coreset_recent_aware),
                recent_summary_k=int(coreset_recent_summary_k),
                diagnostics_out=_ob_diag,
            )
            setattr(model, "_coreset_bicrit_orthbonus_diag", _ob_diag)

        elif method in ("coreset_stream", "coreset_stream_linear"):
            # Gradient-based coreset on probe params (stream-style); question -> target vector, not KV similarity.
            # coreset_stream: conic (nonnegative LS) greedy on gradient rows; coreset_stream_linear: signed LS ablation.
            use_signed = method == "coreset_stream_linear"
            seq_len = key_states_to_compress.shape[2]
            kh = key_states_to_compress[0].float()
            vh = value_states_to_compress[0].float()
            F = seq_len // token_per_frame
            k_frames = min(compress_frame_num, F)
            kh_f = kh.reshape(num_heads, F, token_per_frame, -1).mean(dim=2)
            vh_f = vh.reshape(num_heads, F, token_per_frame, -1).mean(dim=2)
            raw = _raw_frame_features(kh_f, vh_f, "kv")

            def _pick_from_rows(M: torch.Tensor) -> torch.Tensor:
                M = M / M.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                if use_signed:
                    return _greedy_linear_coreset_indices_signed(M, k_frames)
                return _greedy_linear_coreset_indices(M, k_frames)

            if (
                stream_question_embed is None
                or stream_modules is None
                or stream_question_embed.numel() == 0
            ):
                feat = _coreset_frame_features(kh_f, vh_f, "kv")
                idx_f = _pick_from_rows(feat)
            else:
                d_f = raw.shape[1]
                d_q = int(stream_question_embed.shape[0])
                _ensure_stream_modules(stream_modules, d_f, d_q, raw.device)
                probe = stream_modules["probe"]
                q_head = stream_modules["q_head"]
                G = _stream_grad_row_per_frame(
                    raw, stream_question_embed.to(raw.device), probe, q_head
                )
                idx_f = _pick_from_rows(G)
            flat: List[int] = []
            for f in sorted(idx_f.detach().cpu().tolist()):
                base = int(f) * token_per_frame
                flat.extend(range(base, base + token_per_frame))
            all_indices = torch.tensor(flat, device=key_states.device, dtype=torch.long)
            all_indices = all_indices.unsqueeze(0).expand(num_heads, -1)
            all_indices, _ = all_indices.sort(dim=-1)

        elif method in _CORESET_ABLATIONS:
            # Frame-level only; ablations: feat (kv|k|v) × selector (greedy sum/mean, k-center, random)
            feat_mode, sel_mode = _CORESET_ABLATIONS[method]
            seq_len = key_states_to_compress.shape[2]
            kh = key_states_to_compress[0].float()
            vh = value_states_to_compress[0].float()
            F = seq_len // token_per_frame
            k_frames = min(compress_frame_num, F)
            kh_f = kh.reshape(num_heads, F, token_per_frame, -1).mean(dim=2)
            vh_f = vh.reshape(num_heads, F, token_per_frame, -1).mean(dim=2)

            if sel_mode == "random":
                idx_f = _random_frame_indices(
                    F,
                    k_frames,
                    key_states.device,
                    seed=layer_idx * 1_000_003 + seq_len,
                )
            elif sel_mode == "kcenter":
                raw = _raw_frame_features(kh_f, vh_f, feat_mode)
                idx_f = _kcenter_frame_indices(raw, k_frames)
            elif sel_mode == "greedy_sum":
                feat = _coreset_frame_features(kh_f, vh_f, feat_mode)
                idx_f = _greedy_linear_coreset_indices(feat, k_frames)
            elif sel_mode == "greedy_mean":
                feat = _coreset_frame_features(kh_f, vh_f, feat_mode)
                idx_f = _greedy_linear_coreset_indices_mean_target(feat, k_frames)
            elif sel_mode == "greedy_linear_signed":
                feat = _coreset_frame_features(kh_f, vh_f, feat_mode)
                idx_f = _greedy_linear_coreset_indices_signed(feat, k_frames)
            else:
                raise RuntimeError(f"coreset sel_mode {sel_mode}")

            flat: List[int] = []
            for f in sorted(idx_f.detach().cpu().tolist()):
                base = int(f) * token_per_frame
                flat.extend(range(base, base + token_per_frame))
            all_indices = torch.tensor(flat, device=key_states.device, dtype=torch.long)

            all_indices = all_indices.unsqueeze(0).expand(num_heads, -1)
            all_indices, _ = all_indices.sort(dim=-1)

        else:
            raise ValueError(
                f"Unknown compression method: {method}. Available: cords, uniform, swa."
            )
        
        # Apply compression using advanced indexing
        batch_indices = torch.zeros_like(all_indices)
        head_indices = torch.arange(num_heads, device=key_states.device).unsqueeze(1).expand(-1, all_indices.size(1))
        
        # Ensure indices are within bounds
        assert all_indices.max() <= key_states_to_compress.shape[2], \
            f"Selected index max {all_indices.max()} exceeds key length {key_states_to_compress.shape[2]}"
        
        # Gather selected tokens
        new_key_states_to_compress = key_states_to_compress[batch_indices, head_indices, all_indices].unsqueeze(0)
        new_value_states_to_compress = value_states_to_compress[batch_indices, head_indices, all_indices].unsqueeze(0)
        
        # Concatenate with system tokens
        new_k = torch.cat([key_states[:, :, :system_size, :], new_key_states_to_compress], dim=2)
        new_v = torch.cat([value_states[:, :, :system_size, :], new_value_states_to_compress], dim=2)

        past_key_values.layers[layer_idx].keys = new_k
        past_key_values.layers[layer_idx].values = new_v

    return past_key_values, None