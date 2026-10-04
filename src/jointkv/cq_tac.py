"""Hard-forward task-adjoint calibration primitives for grouped CQ.

The cache representation is the native CQ representation: one fixed-width
centroid index per contiguous channel group.  Assignments are recomputed on
every forward pass, while gradients reach only the selected FP16-roundtripped
centroids.  An identity straight-through edge preserves the closed-loop
transformer trajectory across quantized layers without changing the deployed
forward values.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .closed_loop_vq import QuantizedKV


def _gather_group_words(table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[P,N]`` indices from ``[P,K,S]`` tables."""

    if table.ndim != 3 or indices.ndim != 2 or table.shape[0] != indices.shape[0]:
        raise ValueError("CQ table/index geometry differs")
    return torch.gather(
        table,
        1,
        indices[..., None].expand(-1, -1, table.shape[-1]),
    )


def hard_grouped_cq(
    states: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    selector_whiteners: torch.Tensor | None = None,
    token_chunk_size: int = 64,
    straight_through_states: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact hard CQ roundtrip for ``[B,H,T,D]`` and ``[H,G,K,S]``.

    The literal forward uses the persistent FP16 codebook values and detached
    hard assignments.  The gather remains differentiable with respect to the
    selected centroid, implementing the fixed-assignment task-adjoint update.
    """

    if states.ndim != 4 or codebooks.ndim != 4:
        raise ValueError("states/codebooks must be [B,H,T,D]/[H,G,K,S]")
    batch, heads, tokens, dimension = map(int, states.shape)
    cb_heads, groups, clusters, subdimension = map(int, codebooks.shape)
    if cb_heads != heads or groups * subdimension != dimension:
        raise ValueError("CQ codebook geometry differs from states")
    if token_chunk_size <= 0:
        raise ValueError("token chunk size must be positive")
    if selector_whiteners is not None and tuple(selector_whiteners.shape) != (
        heads,
        groups,
        subdimension,
        subdimension,
    ):
        raise ValueError("CQ selector whitener geometry differs")

    problems = heads * groups
    rows = batch * tokens
    raw = (
        states.detach()
        .float()
        .permute(1, 0, 2, 3)
        .reshape(heads, rows, groups, subdimension)
        .permute(0, 2, 1, 3)
        .reshape(problems, rows, subdimension)
        .contiguous()
    )
    # Codebook files are FP16.  Keep an FP32 master for SGD, but make both the
    # search and decoded forward exactly use its FP16 deployment roundtrip.
    deployed = codebooks + (codebooks.half().float() - codebooks).detach()
    search_tables = deployed.detach().float().reshape(problems, clusters, subdimension)
    search_rows = raw
    if selector_whiteners is not None:
        factors = selector_whiteners.detach().float().reshape(
            problems, subdimension, subdimension
        )
        search_rows = torch.bmm(raw, factors.transpose(1, 2))
        search_tables = torch.bmm(search_tables, factors.transpose(1, 2))
    norms = search_tables.square().sum(-1).unsqueeze(1)
    indices = torch.empty(problems, rows, dtype=torch.long, device=states.device)
    for start in range(0, rows, token_chunk_size):
        stop = min(start + token_chunk_size, rows)
        current = search_rows[:, start:stop]
        distances = (
            current.square().sum(-1, keepdim=True)
            + norms
            - 2.0 * torch.bmm(current, search_tables.transpose(1, 2))
        )
        indices[:, start:stop] = distances.argmin(-1)

    live_tables = deployed.reshape(problems, clusters, subdimension)
    decoded = _gather_group_words(live_tables, indices)
    output = (
        decoded.reshape(heads, groups, batch, tokens, subdimension)
        .permute(2, 0, 3, 1, 4)
        .reshape(batch, heads, tokens, dimension)
        .to(states.dtype)
    )
    if straight_through_states:
        output = output + (states - states.detach())
    shaped_indices = (
        indices.reshape(heads, groups, batch, tokens)
        .permute(2, 0, 3, 1)
        .contiguous()
        .to(torch.uint8)
    )
    return output, shaped_indices


class ClosedLoopGroupedCQ(nn.Module):
    """All-layer hard CQ codec with frozen K and trainable V centroids."""

    def __init__(
        self,
        key_codebooks: Sequence[torch.Tensor],
        value_codebooks: Sequence[torch.Tensor],
        key_selector_whiteners: Sequence[torch.Tensor],
        *,
        dynamic_bpa: float,
        token_chunk_size: int = 64,
    ) -> None:
        super().__init__()
        if not (
            len(key_codebooks)
            == len(value_codebooks)
            == len(key_selector_whiteners)
        ) or not key_codebooks:
            raise ValueError("CQ layer payloads differ")
        geometry = tuple(key_codebooks[0].shape)
        if len(geometry) != 4 or any(
            tuple(row.shape) != geometry
            for row in (*key_codebooks, *value_codebooks)
        ):
            raise ValueError("CQ codebook layers have inconsistent geometry")
        if any(
            tuple(row.shape) != (geometry[0], geometry[1], geometry[3], geometry[3])
            for row in key_selector_whiteners
        ):
            raise ValueError("CQ key selector layers have inconsistent geometry")
        self.key_tables = nn.ParameterList(
            [nn.Parameter(row.detach().float().contiguous(), requires_grad=False) for row in key_codebooks]
        )
        self.value_tables = nn.ParameterList(
            [nn.Parameter(row.detach().float().contiguous()) for row in value_codebooks]
        )
        for layer, whitener in enumerate(key_selector_whiteners):
            self.register_buffer(
                f"key_selector_whitener_{layer:02d}",
                whitener.detach().float().contiguous(),
                persistent=True,
            )
        self.dynamic_bpa = float(dynamic_bpa)
        self.token_chunk_size = int(token_chunk_size)

    def key_whitener(self, layer: int) -> torch.Tensor:
        return getattr(self, f"key_selector_whitener_{layer:02d}")

    def forward(
        self,
        layer: int,
        raw_key: torch.Tensor,
        raw_value: torch.Tensor,
        *,
        mode: str = "uniform",
    ) -> QuantizedKV:
        if mode != "uniform":
            raise ValueError("grouped CQ supports one fixed-rate uniform mode")
        key, key_indices = hard_grouped_cq(
            raw_key,
            self.key_tables[layer],
            selector_whiteners=self.key_whitener(layer),
            token_chunk_size=self.token_chunk_size,
        )
        value, value_indices = hard_grouped_cq(
            raw_value,
            self.value_tables[layer],
            token_chunk_size=self.token_chunk_size,
        )
        return QuantizedKV(
            key=key,
            value=value,
            key_indices=key_indices,
            value_indices=value_indices,
            projected_depths=None,
        )
