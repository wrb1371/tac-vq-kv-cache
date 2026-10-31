"""Exact CQ payload accounting, including codec, protected window and query reserve."""

import math

GEOMETRY = {"Omni": (36, 2, 128), "VL": (28, 4, 128)}
COUPLINGS = [1, 2, 4, 8, 16, 32, 64, 128]


def cost(kind, C, c, M, t, Q):
    L, H, D = GEOMETRY[kind]
    W = min(32, M)
    b = C.bit_length() - 1
    assert C == 1 << b and D % c == 0
    n = (H * (M - W + Q) * (D // c) * b + 7) // 8
    codec = 4 * L * H * D * C
    # Separate allocations avoid hidden arena alignment padding.
    state = L * (2 * n + 4 * H * D * (t + W)) + 12 * (M + t + Q) + 4 * (M + t) + 2 * D
    return {
        "codec_bytes": codec,
        "state_bytes": state,
        "resident_bytes": codec + state,
        "packed_bytes_per_side_layer": n,
        "M": M,
        "t": t,
        "Q": Q,
        "C": C,
        "c": c,
        "bits": b,
    }


def admit(kind, C, geometries, budget):
    for c in COUPLINGS:
        rows = [cost(kind, C, c, *g) for g in geometries]
        if max(r["resident_bytes"] for r in rows) <= budget:
            return {
                "C": C,
                "c": c,
                "codec_bytes": rows[0]["codec_bytes"],
                "maximum_resident_bytes": max(r["resident_bytes"] for r in rows),
                "budget_bytes": budget,
                "videos": len(rows),
            }
    return None
