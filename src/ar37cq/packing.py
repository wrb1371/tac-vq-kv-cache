"""Little-endian packed CQ indices, 4--10 bits, no byte per-index fiction."""

import torch


def pack(ids, bits):
    flat = ids.reshape(-1).long()
    n = flat.numel()
    offset = torch.arange(n, device=ids.device) * bits
    byte = offset // 8
    shift = offset % 8
    v = flat << shift
    out = torch.zeros((n * bits + 7) // 8, dtype=torch.int32, device=ids.device)
    out.scatter_add_(0, byte, (v & 255).int())
    ix = byte + 1
    mask = ix < len(out)
    out.scatter_add_(0, ix[mask], ((v[mask] >> 8) & 255).int())
    ix = byte + 2
    mask = ix < len(out)
    out.scatter_add_(0, ix[mask], ((v[mask] >> 16) & 255).int())
    assert bool((out <= 255).all())
    return out.byte()


def unpack(data, bits, n):
    offset = torch.arange(n, device=data.device) * bits
    byte = offset // 8
    shift = offset % 8
    data = torch.nn.functional.pad(data.long(), (0, 2))
    v = data[byte] | (data[byte + 1] << 8) | (data[byte + 2] << 16)
    return (v >> shift) & ((1 << bits) - 1)
