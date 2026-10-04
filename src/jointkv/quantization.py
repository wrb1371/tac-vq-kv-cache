"""Reference (de)quantization used for quality diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizationDiagnostics:
    mse: float
    relative_mse: float
    max_abs_error: float
    statistic_pairs: int


def affine_roundtrip(
    tensor: torch.Tensor,
    *,
    bits: int,
    axis: int,
    group_size: int,
) -> tuple[torch.Tensor, QuantizationDiagnostics]:
    """Groupwise asymmetric affine quantize/dequantize without hidden state.

    ``axis`` is the dimension divided into groups. Other dimensions receive an
    independent scale/zero pair. The function returns floating-point replay so
    model-quality tests remain independent of a particular packing kernel.
    """

    if bits not in {1, 2, 3, 4, 8, 16}:
        raise ValueError(f"unsupported bits={bits}")
    if tensor.numel() == 0 or not tensor.is_floating_point():
        raise ValueError("tensor must be a non-empty floating tensor")
    axis %= tensor.ndim
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if bits == 16:
        replay = tensor.clone()
        return replay, QuantizationDiagnostics(0.0, 0.0, 0.0, 0)

    levels = (1 << bits) - 1
    replay = torch.empty_like(tensor)
    statistic_pairs = 0
    work = tensor.float()
    for start in range(0, tensor.shape[axis], group_size):
        stop = min(start + group_size, tensor.shape[axis])
        index = [slice(None)] * tensor.ndim
        index[axis] = slice(start, stop)
        chunk = work[tuple(index)]
        minimum = chunk.amin(dim=axis, keepdim=True)
        maximum = chunk.amax(dim=axis, keepdim=True)
        scale = (maximum - minimum) / levels
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = torch.round((chunk - minimum) / safe_scale).clamp_(0, levels)
        dequantized = codes * safe_scale + minimum
        dequantized = torch.where(scale > 0, dequantized, minimum)
        replay[tuple(index)] = dequantized.to(dtype=tensor.dtype)
        statistic_pairs += minimum.numel()

    error = replay.float() - work
    mse = float(error.square().mean().item())
    energy = float(work.square().mean().item())
    return replay, QuantizationDiagnostics(
        mse=mse,
        relative_mse=mse / max(energy, torch.finfo(torch.float32).tiny),
        max_abs_error=float(error.abs().max().item()),
        statistic_pairs=statistic_pairs,
    )


def kivi_roundtrip(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    bits: int,
    group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, QuantizationDiagnostics]]:
    """Replay KIVI's key-per-channel/value-per-token asymmetry.

    Expected layout is ``[batch, kv_head, token, channel]``. Key groups run
    along tokens (one range per channel); value groups run along channels (one
    range per token).
    """

    if key.shape != value.shape or key.ndim != 4:
        raise ValueError("key and value must share [B,H,T,D] shape")
    q_key, key_diag = affine_roundtrip(key, bits=bits, axis=2, group_size=group_size)
    q_value, value_diag = affine_roundtrip(value, bits=bits, axis=3, group_size=group_size)
    return q_key, q_value, {"key": key_diag, "value": value_diag}
