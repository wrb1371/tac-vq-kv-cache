"""RVQ stage accumulation, exact inverse GEMM, and fused temporal epilogue.

The epilogue restores gains/scales, rounds decoded anchors and innovations to
BF16, composes temporal features, applies native MRoPE, and scatters protected
text. No full BF16 anchor/innovation intermediate is written. All scratch is
request workspace; the packed state and codec are read-only.
"""

import types
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "IS0", "IS1", "START"])
def _lookup(
    I,
    Books,
    Z,
    N,
    IS0,
    IS1,
    START,
    R: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = x // 128
    d = x % 128
    h = row // N
    t = row % N + START
    valid = row < HEADS * N
    z = tl.full((BLOCK,), 0, tl.float32)
    for r in tl.static_range(R):
        code = tl.load(I + h * IS0 + t * IS1 + r * 8 + d // 16, valid, 0).to(tl.int32)
        z += tl.load(Books + (r * 256 + code) * 16 + d % 16, valid, 0).to(tl.float32)
    tl.store(Z + x, z, valid)


@triton.jit
def _restore(
    U, Norm, Scale, Mean, head, slot, d, N, NS0, INDEPENDENT: tl.constexpr, mask
):
    z = tl.load(U + (head * N + slot) * 128 + d, mask, 0)
    g = tl.load(Norm + head * NS0 + slot, mask, 0).to(tl.float32)
    a = tl.load(Scale + head * 128 + d, mask, 0).to(tl.float32)
    result = (z * g) * a
    if INDEPENDENT:
        result += tl.load(Mean + head * 128 + d, mask, 0).to(tl.float32)
    return result.to(tl.bfloat16).to(tl.float32)


@triton.jit(
    do_not_specialize=[
        "END",
        "M",
        "NT",
        "A",
        "NI",
        "CS0",
        "KS0",
        "VS0",
        "HEIGHT",
        "WIDTH",
        "GOP",
    ]
)
def _compose(
    AK,
    AV,
    IK,
    IV,
    KN,
    VN,
    KS,
    VS,
    KM,
    VM,
    TK,
    TV,
    Vid,
    Text,
    Code,
    Rank,
    Cos,
    Sin,
    OK,
    OV,
    END,
    M,
    NT,
    A,
    NI,
    CS0,
    KS0,
    VS0,
    HEIGHT,
    WIDTH,
    GOP,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    head = tl.program_id(1)
    work = x // 128
    d = x % 128
    dh = (d + 64) % 128
    valid = work < M + NT
    video = work < M
    text = (work >= M) & valid
    seq = tl.load(Vid + work, video, 0)
    seq = tl.where(text, tl.load(Text + work - M, text, 0), seq)
    code = tl.load(Code + head * CS0 + seq, video, 4).to(tl.int32)
    frame = work // (HEIGHT * WIDTH)
    anchor = frame % GOP == 0
    yy = tl.minimum(
        tl.maximum(work // WIDTH % HEIGHT + (code & 127) // 3 - 1, 0), HEIGHT - 1
    )
    xx = tl.minimum(tl.maximum(work % WIDTH + (code & 127) % 3 - 1, 0), WIDTH - 1)
    ref = frame // GOP * HEIGHT * WIDTH + yy * WIDTH + xx
    ref = tl.where(anchor, frame // GOP * HEIGHT * WIDTH + work % (HEIGHT * WIDTH), ref)
    rank = tl.load(Rank + head * M + work, video, 0)
    active = video & ~anchor & ((code & 128) != 0)
    k = _restore(AK, KN, KS, KM, head, ref, d, A, KS0, True, video)
    kh = _restore(AK, KN, KS, KM, head, ref, dh, A, KS0, True, video)
    v = _restore(AV, VN, VS, VM, head, ref, d, A, VS0, True, video)
    # Innovation GEMM rows start at0; their gain slots start after the anchors.
    ki = tl.load(IK + (head * NI + rank) * 128 + d, active, 0)
    kih = tl.load(IK + (head * NI + rank) * 128 + dh, active, 0)
    vv = tl.load(IV + (head * NI + rank) * 128 + d, active, 0)
    kg = tl.load(KN + head * KS0 + A + rank, active, 0).to(tl.float32)
    vg = tl.load(VN + head * VS0 + A + rank, active, 0).to(tl.float32)
    ki = (
        ((ki * kg) * tl.load(KS + head * 128 + d, active, 0).to(tl.float32))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    kih = (
        ((kih * kg) * tl.load(KS + head * 128 + dh, active, 0).to(tl.float32))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    vv = (
        ((vv * vg) * tl.load(VS + head * 128 + d, active, 0).to(tl.float32))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    k = (k + ki).to(tl.bfloat16).to(tl.float32)
    kh = (kh + kih).to(tl.bfloat16).to(tl.float32)
    v = (v + vv).to(tl.bfloat16)
    axis = tl.where(d % 64 < 16, 0, tl.where(d % 64 < 40, 1, 2))
    phase_index = (axis * END + seq) * 64 + d % 64
    co = tl.load(Cos + phase_index, video, 0).to(tl.float32)
    si = tl.load(Sin + phase_index, video, 0).to(tl.float32)
    left = (k * co).to(tl.bfloat16).to(tl.float32)
    right = (tl.where(d < 64, -kh, kh) * si).to(tl.bfloat16).to(tl.float32)
    k = (left + right).to(tl.bfloat16)
    k = tl.where(text, tl.load(TK + (head * NT + work - M) * 128 + d, text, 0), k)
    v = tl.where(text, tl.load(TV + (head * NT + work - M) * 128 + d, text, 0), v)
    tl.store(OK + (head * END + seq) * 128 + d, k, valid)
    tl.store(OV + (head * END + seq) * 128 + d, v, valid)


@torch.inference_mode()
def decode(codec, layer):
    if codec.plan["hybrid_dense"]:
        return codec._ar35_reference_decode(layer)
    assert codec.name == "STI-RVQ" and codec.table["H"].shape == (128, 128)
    st = codec.states[layer]
    table = codec.table
    heads = st["K_index"].shape[0]
    a = codec.plan["anchors"]
    ni = codec.plan["layer_choices"][layer]["innovations"]
    arrays = []
    for start, n in [(0, a), (a, ni)]:
        for side in ["K", "V"]:
            ix = st[side + "_index"]
            z = torch.empty((heads, n, 128), dtype=torch.float32, device=ix.device)
            if n:
                _lookup[(triton.cdiv(heads * n * 128, 256),)](
                    ix,
                    table[side + "_books"],
                    z,
                    n,
                    ix.stride(0),
                    ix.stride(1),
                    start,
                    ix.shape[2],
                    heads,
                    256,
                    enable_fp_fusion=False,
                )
                z = z @ table["H"].T
            arrays.append(z)
    vi = codec.global_maps["video"]
    ti = codec.global_maps["text"]
    m = len(vi)
    nt, h, w = codec.grid
    g = codec.plan["gop"]
    ordinal = torch.arange(m, device=vi.device)
    code = st["T_code"][:, vi.long()]
    active = ((code & 128) != 0) & (((ordinal // (h * w)) % g) != 0)[None]
    rank = active.int().cumsum(1, dtype=torch.int32) - 1
    phase = (
        codec.global_maps["positions"][:, : codec.end].float()[:, :, None]
        * codec.global_maps["freq"][None, None, :]
    )
    co = (phase.cos() * codec.scale).bfloat16()
    si = (phase.sin() * codec.scale).bfloat16()
    out = [
        torch.empty((heads, codec.end, 128), dtype=torch.bfloat16, device=vi.device)
        for _ in range(2)
    ]
    _compose[(triton.cdiv((m + len(ti)) * 128, 256), heads)](
        *arrays,
        st["K_norm"],
        st["V_norm"],
        table["K_scale"][layer],
        table["V_scale"][layer],
        table["K_mean"][layer],
        table["V_mean"][layer],
        st["text_FP_K"],
        st["text_FP_V"],
        vi,
        ti,
        st["T_code"],
        rank,
        co,
        si,
        *out,
        codec.end,
        m,
        len(ti),
        a,
        ni,
        st["T_code"].stride(0),
        st["K_norm"].stride(0),
        st["V_norm"].stride(0),
        h,
        w,
        g,
        256,
        enable_fp_fusion=False
    )
    return tuple(out)


def install(codec):
    assert not hasattr(codec, "_ar35_reference_decode")
    codec._ar35_reference_decode = codec.decode
    codec.decode = types.MethodType(decode, codec)
    codec.ar35_reader = "stage_lookup_inverse_GEMM_fused_gain_temporal_MRoPE_epilogue"
    return codec
