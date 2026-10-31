"""AR21 runtime layout kernels; same numeric math, bounded layout specialization."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=['N', 'IS0', 'IS1'])
def _lookup(I, B, S, Z, N, IS0, IS1, R: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = x // 128
    dim = x % 128
    h = row // N
    t = row % N
    slot = tl.load(S + h * N + t, row < 2 * N, 0)
    value = tl.full((BLOCK,), 0, tl.float32)
    for r in tl.static_range(R):
        code = tl.load(I + h * IS0 + slot * IS1 + r * 8 + dim // 16, row < 2 * N, 0).to(tl.int32)
        value += tl.load(B + (r * 256 + code) * 16 + dim % 16, row < 2 * N, 0).to(tl.float32)
    tl.store(Z + x, value, row < 2 * N)


@triton.jit(do_not_specialize=['N', 'NS0'])
def _epilogue(Z, Norm, Scale, Mean, S, O, N, NS0, INDEPENDENT: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = x // 128
    dim = x % 128
    h = row // N
    slot = tl.load(S + row, row < 2 * N, 0)
    z = tl.load(Z + x, row < 2 * N, 0)
    norm = tl.load(Norm + h * NS0 + slot, row < 2 * N, 0).to(tl.float32)
    scale = tl.load(Scale + h * 128 + dim, row < 2 * N, 0).to(tl.float32)
    v = (z * norm) * scale
    if INDEPENDENT:
        v = v + tl.load(Mean + h * 128 + dim, row < 2 * N, 0).to(tl.float32)
    tl.store(O + x, v, row < 2 * N)


def read_slots(l, state, kind, slots, independent, tables):
    slots = slots.contiguous()
    n = slots.shape[1]
    ix = state[kind + '_index']
    z = torch.empty((2, n, 128), device=ix.device, dtype=torch.float32)
    _lookup[(triton.cdiv(2 * n * 128, 256),)](ix, tables[kind + '_books'], slots, z, n, ix.stride(0), ix.stride(1), ix.shape[2], 256)
    z = z @ tables['H'].T
    out = torch.empty((2, n, 128), device=ix.device, dtype=torch.bfloat16)
    norm = state[kind + '_norm']
    _epilogue[(triton.cdiv(2 * n * 128, 256),)](z, norm, tables[kind + '_scale'][l], tables[kind + '_mean'][l], slots, out, n, norm.stride(0), independent, 256, enable_fp_fusion=False)
    return out


@triton.jit(do_not_specialize=['ROWS', 'NT', 'OS0', 'OS1'])
def _rvq(X, B, O, Slots, ROWS, NT, OS0, OS1, R: tl.constexpr, BM: tl.constexpr, PRECISION: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    d = tl.arange(0, 16)
    c = tl.arange(0, 256)
    z = tl.load(X + row[:, None] * 16 + d[None, :], row[:, None] < ROWS, 0)
    head = row // (NT * 8)
    tok = row // 8 % NT
    group = row % 8
    slot = tl.load(Slots + tok, row < ROWS, 0)
    for r in tl.static_range(R):
        book = tl.load(B + r * 4096 + c[None, :] * 16 + d[:, None]).to(tl.float32)
        norm = tl.sum(book * book, axis=0)
        xx = tl.sum(z * z, axis=1)
        dot = tl.dot(z, book, input_precision=PRECISION)
        dist = (xx[:, None] + norm[None, :]) - 2.0 * dot
        chosen = tl.argmin(dist, axis=1, tie_break_left=True)
        tl.store(O + head * OS0 + slot * OS1 + r * 8 + group, chosen.to(tl.uint8), row < ROWS)
        reconstruction = tl.load(B + r * 4096 + chosen[:, None] * 16 + d[None, :]).to(tl.float32)
        z = z - reconstruction


def encode_rvq(z, books, out, slots, block=32, precision='ieee'):
    assert z.is_contiguous() and z.shape[-1] == 16
    nt = z.numel() // (2 * 128)
    _rvq[(triton.cdiv(2 * nt * 8, block),)](z, books, out, slots.contiguous(), 2 * nt * 8, nt, out.stride(0), out.stride(1), out.shape[2], block, precision, num_warps=4, enable_fp_fusion=False)


@triton.jit(do_not_specialize=['N', 'T', 'H', 'W', 'GOP'])
def _match9(K, V, AK, AV, Vid, SK, SV, Targets, Codes, Best, N, T, H, W, GOP):
    target = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, 128)
    ordinal = tl.load(Targets + target)
    seq = tl.load(Vid + ordinal)
    frame = ordinal // (H * W)
    y = ordinal // W % H
    x = ordinal % W
    k = tl.load(K + (head * N + seq) * 128 + d)
    v = tl.load(V + (head * N + seq) * 128 + d)
    sk = tl.load(SK + head * 128 + d).to(tl.float32)
    sv = tl.load(SV + head * 128 + d).to(tl.float32)
    best = float('inf')
    code = 4
    for c in tl.static_range(9):
        yy = tl.minimum(tl.maximum(y + c // 3 - 1, 0), H - 1)
        xx = tl.minimum(tl.maximum(x + c % 3 - 1, 0), W - 1)
        reford = frame // GOP * GOP * H * W + yy * W + xx
        refseq = tl.load(Vid + reford)
        pk = tl.load(AK + (head * N + refseq) * 128 + d).to(tl.float32)
        pv = tl.load(AV + (head * N + refseq) * 128 + d).to(tl.float32)
        dk = (k - pk) / sk
        dv = (v - pv) / sv
        dist = tl.sum(dk * dk, 0) / 128 + tl.sum(dv * dv, 0) / 128
        code = tl.where(dist < best, c, code)
        best = tl.minimum(best, dist)
    tl.store(Codes + head * T + target, code)
    tl.store(Best + head * T + target, best)


def match9(raw, decoded, vid, targets, scales, height, width, gop):
    n = raw['K'].shape[1]
    count = targets.numel()
    codes = torch.empty((2, count), device=vid.device, dtype=torch.uint8)
    best = torch.empty((2, count), device=vid.device, dtype=torch.float32)
    _match9[(count, 2)](raw['K'].contiguous(), raw['V'].contiguous(), decoded['K'], decoded['V'], vid, scales[0], scales[1], targets, codes, best, n, count, height, width, gop, num_warps=4, enable_fp_fusion=False)
    return codes, best


@triton.jit(do_not_specialize=['N', 'M', 'NT', 'A', 'NR', 'NQ', 'END', 'CS0', 'PS0', 'H', 'W', 'GOP'])
def _compose(AK, AV, RK, RV, QK, QV, TK, TV, Vid, Text, Code, Rank, Pos, Freq, OK, OV,
             N, M, NT, A, NR,
             NQ, END, CS0, PS0,
             H, W, GOP, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    xx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    head = tl.program_id(1)
    work = xx // 128
    d = xx % 128
    dh = (d + 64) % 128
    valid = work < M + NT + NQ
    video = work < M
    text = (work >= M) & (work < M + NT)
    query = (work >= M + NT) & valid
    seq = tl.load(Vid + work, video, 0)
    seq = tl.where(text, tl.load(Text + work - M, text, 0), seq)
    seq = tl.where(query, END + work - M - NT, seq)
    code = tl.load(Code + head * CS0 + seq, video, 4).to(tl.int32)
    frame = work // (H * W)
    anchor = frame % GOP == 0
    # only lower seven code bits identify correspondence (0..8).
    yy = tl.minimum(tl.maximum(work // W % H + (code & 127) // 3 - 1, 0), H - 1)
    xpos = tl.minimum(tl.maximum(work % W + (code & 127) % 3 - 1, 0), W - 1)
    refslot = frame // GOP * H * W + yy * W + xpos
    refslot = tl.where(anchor, frame // GOP * H * W + work % (H * W), refslot)
    rank = tl.load(Rank + head * M + work, video, 0)
    active = video & ~anchor & ((code & 128) != 0)
    ai = (head * A + refslot) * 128
    ri = (head * NR + rank) * 128
    qi = (head * NQ + work - M - NT) * 128
    ti = (head * NT + work - M) * 128
    k = tl.load(AK + ai + d, video, 0).to(tl.float32)
    kh = tl.load(AK + ai + dh, video, 0).to(tl.float32)
    v = tl.load(AV + ai + d, video, 0).to(tl.float32)
    k = (k + tl.load(RK + ri + d, active, 0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    kh = (kh + tl.load(RK + ri + dh, active, 0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    v = (v + tl.load(RV + ri + d, active, 0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    k = tl.where(query, tl.load(QK + qi + d, query, 0).to(tl.float32), k)
    kh = tl.where(query, tl.load(QK + qi + dh, query, 0).to(tl.float32), kh)
    v = tl.where(query, tl.load(QV + qi + d, query, 0).to(tl.float32), v)
    axis = tl.where(d % 64 < 16, 0, tl.where(d % 64 < 40, 1, 2))
    position = tl.load(Pos + axis * PS0 + seq, valid & ~text, 0).to(tl.float32)
    phase = position * tl.load(Freq + d % 64)
    co = (libdevice.cos(phase) * SCALE).to(tl.bfloat16).to(tl.float32)
    si = (libdevice.sin(phase) * SCALE).to(tl.bfloat16).to(tl.float32)
    x = (k * co).to(tl.bfloat16).to(tl.float32)
    y = (tl.where(d < 64, -kh, kh) * si).to(tl.bfloat16).to(tl.float32)
    k = (x + y).to(tl.bfloat16)
    k = tl.where(text, tl.load(TK + ti + d, text, 0), k)
    v = tl.where(text, tl.load(TV + ti + d, text, 0), v)
    tl.store(OK + (head * N + seq) * 128 + d, k, valid)
    tl.store(OV + (head * N + seq) * 128 + d, v, valid)


def compose(codec, state, positions, inv_freq, attention_scaling, arrays):
    n = state['n']; m = len(codec._video_list); nt = len(codec._text_list)
    a = codec.plan['anchor_tokens']; nr = codec.plan['innovation_slots']; nq = n - codec._end
    order = torch.arange(m, device='cuda')
    codes = state['T_code'][:, codec._vid_gpu.long()]
    active = ((codes & 128) != 0) & ((order // (codec._h * codec._w)) % int(codec.policy['gop']) != 0)[None]
    rank = active.int().cumsum(1, dtype=torch.int32) - 1
    out = [torch.empty((1, 2, n, 128), device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    _compose[(triton.cdiv((m + nt + nq) * 128, 256), 2)](*arrays, state['text_FP_K'], state['text_FP_V'], codec._vid_gpu, codec._text_gpu, state['T_code'], rank, positions, inv_freq, *out,
        n, m, nt, a, nr, nq, codec._end, state['T_code'].stride(0), positions.stride(0), codec._h, codec._w, int(codec.policy['gop']), attention_scaling, 256, enable_fp_fusion=False)
    return tuple(out)


@triton.jit(do_not_specialize=['ROWS', 'NT', 'OS0', 'OS1', 'STAGE'])
def _tail(Dot,XX,CN,Book,Z,Out,Slots,ROWS,NT,OS0,OS1,STAGE):
    row=tl.program_id(0);c=tl.arange(0,256);d=tl.arange(0,16)
    dot=tl.load(Dot+row*256+c);xx=tl.load(XX+row);cn=tl.load(CN+c)
    distance=(xx+cn)-2.0*dot
    index=tl.argmin(distance,0,tie_break_left=True)
    head=row//(NT*8);token=row//8%NT;group=row%8;slot=tl.load(Slots+token)
    tl.store(Out+head*OS0+slot*OS1+STAGE*8+group,index.to(tl.uint8))
    z=tl.load(Z+row*16+d)
    reconstruction=tl.load(Book+index*16+d).to(tl.float32)
    tl.store(Z+row*16+d,z-reconstruction)

