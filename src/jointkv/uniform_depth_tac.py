"""Task-adaptive closed-loop residual VQ at an arbitrary uniform depth.

This is intentionally smaller than the research ``PureUniform16Codec``: the
rate-transfer experiment needs one deployable hard U32 path, frozen key
tables/whiteners, and trainable value tables.  There is no selector, mixed
precision action, recent-token exception, or soft-assignment surrogate.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .closed_loop_vq import QuantizedKV, hard_residual_vq_exits


class ClosedLoopUniformResidualVQ(nn.Module):
    """Literal hard uniform residual VQ with value-only task adaptation."""

    def __init__(
        self,
        key_tables: Sequence[torch.Tensor],
        value_tables: Sequence[torch.Tensor],
        key_whiteners: Sequence[torch.Tensor | None],
        *,
        depth: int,
        dynamic_bpa: float,
        token_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        if depth <= 0 or dynamic_bpa <= 0 or token_chunk_size <= 0:
            raise ValueError("invalid uniform residual-VQ configuration")
        if not key_tables or not (
            len(key_tables) == len(value_tables) == len(key_whiteners)
        ):
            raise ValueError("uniform residual-VQ layer payloads differ")
        geometry = tuple(key_tables[0].shape)
        if len(geometry) != 4 or geometry[1] != depth:
            raise ValueError("uniform residual-VQ tables have the wrong depth")
        if geometry[2] != 256:
            raise ValueError("this deployment contract requires 8-bit stage indices")
        if any(tuple(table.shape) != geometry for table in (*key_tables, *value_tables)):
            raise ValueError("uniform residual-VQ layer geometries differ")
        self.key_tables = nn.ParameterList(
            [nn.Parameter(table.detach().float().contiguous(), requires_grad=False) for table in key_tables]
        )
        self.value_tables = nn.ParameterList(
            [nn.Parameter(table.detach().float().contiguous()) for table in value_tables]
        )
        heads, _stages, _clusters, dimension = geometry
        for layer, whitener in enumerate(key_whiteners):
            if whitener is not None and tuple(whitener.shape) != (heads, dimension, dimension):
                raise ValueError("key assignment whitener geometry differs")
            self.register_buffer(
                f"key_whitener_{layer:02d}",
                torch.empty(0, dtype=torch.float32)
                if whitener is None
                else whitener.detach().float().contiguous(),
                persistent=True,
            )
        self.depth = int(depth)
        self.dynamic_bpa = float(dynamic_bpa)
        self.token_chunk_size = int(token_chunk_size)

    def whitener(self, layer: int) -> torch.Tensor | None:
        value = getattr(self, f"key_whitener_{layer:02d}")
        return None if value.numel() == 0 else value

    def forward(
        self,
        layer: int,
        raw_key: torch.Tensor,
        raw_value: torch.Tensor,
        *,
        mode: str = "uniform",
    ) -> QuantizedKV:
        if mode != "uniform":
            raise ValueError("uniform residual VQ exposes only uniform mode")
        key_exits, key_indices = hard_residual_vq_exits(
            raw_key,
            self.key_tables[layer],
            prefixes=(self.depth,),
            assignment_whiteners=self.whitener(layer),
            token_chunk_size=self.token_chunk_size,
        )
        value_exits, value_indices = hard_residual_vq_exits(
            raw_value,
            self.value_tables[layer],
            prefixes=(self.depth,),
            assignment_whiteners=None,
            token_chunk_size=self.token_chunk_size,
        )
        return QuantizedKV(
            key=key_exits[self.depth],
            value=value_exits[self.depth],
            key_indices=key_indices,
            value_indices=value_indices,
            projected_depths=None,
        )
