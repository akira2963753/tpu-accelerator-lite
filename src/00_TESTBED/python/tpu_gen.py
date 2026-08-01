#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# tpu_gen.py : TPU-top input + golden generator (library + single-shot CLI).
#
# golden = C = sum_k (2A+1)*(W|1)  (then OP : ReLU + quant).
# activation gets the RPE guard bit ({a,1'b1} = 2a+1) ; weight's LSB is the
# expected-value 1 (w|1). Matches the reduced MAC (== RSA full_mac_contrib).
#
# Weights are kept MSR-4 by default (small magnitude) so no compensation fires;
# non_msr4_per_col injects a few large weights to exercise the compensation
# path. Injection respects the hardware limit of <= 1 Non-MSR-4 per column per
# 16-row tile (the CMEM/compensation limit) ; allow_tile_overflow deliberately
# violates it (2 in one tile) to exercise the CMEM-overflow corner.
#
# The three .dat files hold only the UNIQUE data ; PATTERN.sv replays them in
# the order the hardware asks for (see the index maps below). That keeps the
# weight file MT times smaller -- at 1024^3 the naive full stream would be
# 4.2M lines / 138 MB.
#
#   *_weight.dat      nt -> kt -> 16 k-rows          beat = 16 x 8-bit
#                     hw stream mt->nt->kt->r  ==>  idx = i % (NT*KT*16)
#   *_activation.dat  mt -> kt -> m=0..15            beat = 16 x 7-bit
#                     hw stream mt->nt->kt->m  ==>  idx = mt*KT*16 + kt*16 + m
#   *_golden.dat      mt -> nt -> m=0..15            beat = 16 x 7-bit
#                     hw stream mt->nt->m      ==>  idx = i
#
# All hex, one beat per line, lane 0 in the LSB (matches bus [w*i +: w]).
# ---------------------------------------------------------------------------
import os
import argparse
import random

# ---- design parameters (match define.vh) ----
ARRAY_S = 16
W_W     = 8
A_W     = 7
SAT_MAX = (1 << (A_W - 1)) - 1  # 63 : max positive A_W signed (matches OP.sv)


def mask(v, bits):
    return v & ((1 << bits) - 1)


def signed(v, bits):
    v = mask(v, bits)
    return v - (1 << bits) if (v & (1 << (bits - 1))) else v


def is_msr4(w8):
    """W[7:4] all-0 or all-1 -> MSR-4 (no compensation needed)."""
    top = (mask(w8, 8) >> 4) & 0xF
    non_msr4 = (1 if top == 0xF else 0) ^ (1 if top != 0x0 else 0)
    return non_msr4 == 0


def rand_non_msr4(rng):
    """full-range signed 8-bit weight that is NOT MSR-4 (needs compensation)."""
    while True:
        w = rng.randrange(-128, 128) & 0xFF
        if not is_msr4(w):
            return w


# ---------------------------------------------------------------------------
# data generation (standard library only)
# ---------------------------------------------------------------------------
def gen_A(rng, M, K):
    # full-range signed 7-bit activations
    lo, hi = -(1 << (A_W - 1)), (1 << (A_W - 1))
    return [[rng.randrange(lo, hi) for _ in range(K)] for _ in range(M)]


def gen_W(rng, K, N, non_msr4_per_col, allow_tile_overflow=False):
    # default : MSR-4-only weights (top nibble all-0 or all-1) -> signed [-16, 15],
    # so no compensation is needed and golden == hardware for every weight.
    # stored as the raw unsigned byte the host would send.
    W = [[rng.randrange(-16, 16) & 0xFF for _ in range(N)] for _ in range(K)]
    if non_msr4_per_col <= 0 and not allow_tile_overflow:
        return W

    n_tiles = K // ARRAY_S
    if allow_tile_overflow:
        # CMEM-overflow corner : force 2 Non-MSR-4 into the SAME 16-row tile of a
        # column. Hardware keeps only 1 -> compensation is wrong -> golden mismatch.
        for n in range(N):
            for r in rng.sample(range(ARRAY_S), 2):
                W[r][n] = rand_non_msr4(rng)
    else:
        # legal path : at most 1 Non-MSR-4 per column per 16-row tile. Inject into
        # `k` distinct tiles (capped by tile count), one random row within each.
        k = min(non_msr4_per_col, n_tiles)
        for n in range(N):
            for kt in rng.sample(range(n_tiles), k):
                r = kt * ARRAY_S + rng.randrange(ARRAY_S)
                W[r][n] = rand_non_msr4(rng)
    return W


def op_model(C, qmult, qshift):
    """OP.sv model : ReLU -> * M0 -> >> n (truncate) -> saturate to [0, SAT_MAX]."""
    qmult, qshift = int(qmult), int(qshift)
    return [[min(SAT_MAX, (max(0, v) * qmult) >> qshift) for v in row]
            for row in C]


# ---------------------------------------------------------------------------
# packing / writing
# ---------------------------------------------------------------------------
def pack_rows(mat, lane_w):
    """(R, ARRAY_S) matrix -> list of R Python ints, lane 0 in the LSB."""
    lane_mask = (1 << lane_w) - 1
    out = []
    for row in mat:
        assert len(row) == ARRAY_S, "each packed row must have ARRAY_S lanes"
        v = 0
        for i, lane in enumerate(row):
            v |= (int(lane) & lane_mask) << (lane_w * i)
        out.append(v)
    return out


def hex_str(value, total_bits):
    return format(value, "0{}x".format((total_bits + 3) // 4))


def write_dat(path, rows, total_bits):
    # newline="\n" : force LF on every platform ($readmemh-friendly, no CRLF)
    nib = (total_bits + 3) // 4
    with open(path, "w", newline="\n") as f:
        f.write("".join(format(v, "0{}x".format(nib)) + "\n" for v in rows))


def _tile_stack(mat, row_tiles, col_tiles):
    """slice `mat` into 16x16 blocks in (row_tile major, col_tile minor) order and
    stack them vertically -> (row_tiles*col_tiles*16, 16)."""
    rows = []
    for rt in range(row_tiles):
        for ct in range(col_tiles):
            base_r, base_c = rt * ARRAY_S, ct * ARRAY_S
            for r in range(ARRAY_S):
                rows.append(mat[base_r + r][base_c:base_c + ARRAY_S])
    return rows


# ---------------------------------------------------------------------------
# one full case : returns packed stream-order rows + metadata (no file I/O)
# ---------------------------------------------------------------------------
def auto_qshift(C, qmult):
    """Pick the right-shift that lands the 99th-percentile positive accumulator
    near SAT_MAX, so a randomised case actually exercises the requant range
    instead of coming out all-zero or all-saturated."""
    pos = sorted(v for row in C for v in row if v > 0)
    if pos:
        # NumPy's historical default percentile is linear interpolation.  The
        # integer formulation below is its value truncated to int, without
        # using floating point.
        lo, rem = divmod(99 * (len(pos) - 1), 100)
        hi = min(lo + 1, len(pos) - 1)
        ref = (pos[lo] * (100 - rem) + pos[hi] * rem) // 100
    else:
        ref = 1
    t = max(1, (ref * int(qmult)) // SAT_MAX)
    return max(0, min(63, t.bit_length() - 1))


def _matmul_model(A, W):
    """Exact integer C = (2*A+1) @ signed(W|1), using only Python built-ins.

    Keeping W transposed makes each dot-product walk contiguous tuples.  This
    is intentionally dependency-free; large 512^3 and 1024^3 cases are much
    slower than the previous NumPy/BLAS implementation.
    """
    w_cols = [tuple(signed(row[n] | 1, W_W) for row in W)
              for n in range(len(W[0]))]
    C = []
    for a_row in A:
        a_ext = tuple(2 * a + 1 for a in a_row)
        C.append([sum(a * w for a, w in zip(a_ext, w_col)) for w_col in w_cols])
    return C


def gen_case(M, K, N, non_msr4_per_col, qmult, qshift, seed,
             allow_tile_overflow=False, A_in=None):
    """Generate one matmul case. Seeds the RNG so the case is reproducible.
    A_in  : supply the activation matrix instead of randomising it -- used to chain
            a fused job onto the quantised output of the job before it.
    qshift: None -> auto-picked (see auto_qshift) and reported back in meta.
    Returns (w_rows, a_rows, g_rows, meta, Cp)."""
    assert M % ARRAY_S == 0 and K % ARRAY_S == 0 and N % ARRAY_S == 0, \
        "M/K/N must be multiples of ARRAY_S (16)"
    rng = random.Random(seed)
    MT, KT, NT = M // ARRAY_S, K // ARRAY_S, N // ARRAY_S

    if A_in is None:
        A = gen_A(rng, M, K)
    else:
        assert len(A_in) == M and all(len(row) == K for row in A_in), \
            "fused activation shape must be (%d,%d)" % (M, K)
        A = [[int(v) for v in row] for row in A_in]
    W = gen_W(rng, K, N, non_msr4_per_col, allow_tile_overflow)

    # golden : sum_k (2A+1)*(W|1), then OP (ReLU + quant).
    # Python integers are arbitrary-precision, so the reference is exact.
    C = _matmul_model(A, W)
    if qshift is None:
        qshift = auto_qshift(C, qmult)
    Cp = op_model(C, qmult, qshift)

    # ---- stream-order rows (unique data only ; PATTERN replays) ----
    # weight : nt -> kt -> 16 k-rows, each row = W[kt*16+r, nt*16 .. +15]
    wblocks = []
    for nt in range(NT):
        for kt in range(KT):
            for r in range(ARRAY_S):
                wblocks.append(W[kt * ARRAY_S + r][nt * ARRAY_S:(nt + 1) * ARRAY_S])
    w_rows = pack_rows(wblocks, W_W)
    # activation : mt -> kt -> m=0..15
    a_rows = pack_rows(_tile_stack(A, MT, KT), A_W)
    # golden : mt -> nt -> m=0..15
    g_rows = pack_rows(_tile_stack(Cp, MT, NT), A_W)

    raw_min = min(min(row) for row in C)
    raw_max = max(max(row) for row in C)
    relu0 = sum(v <= 0 for row in C for v in row)
    sat = sum(((max(0, v) * int(qmult)) >> int(qshift)) > SAT_MAX
              for row in C for v in row)
    meta = dict(M=M, K=K, N=N, MT=MT, KT=KT, NT=NT,
                w_rows=len(w_rows), a_rows=len(a_rows), r_rows=len(g_rows),
                w_beats=MT * NT * KT * ARRAY_S,     # what the DUT actually consumes
                a_beats=MT * NT * KT * ARRAY_S,
                r_beats=MT * NT * ARRAY_S,
                raw_min=raw_min, raw_max=raw_max, relu0=relu0, saturated=sat,
                non_msr4_per_col=non_msr4_per_col, qmult=qmult, qshift=qshift,
                overflow=allow_tile_overflow)
    return w_rows, a_rows, g_rows, meta, Cp


# ---------------------------------------------------------------------------
# main : ad-hoc single-shot generator (kept for convenience)
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="TPU single-case golden generator")
    ap.add_argument("-M", type=int, default=16, help="rows of A/C (multiple of 16)")
    ap.add_argument("-K", type=int, default=16, help="contraction dim (multiple of 16)")
    ap.add_argument("-N", type=int, default=16, help="cols of W/C (multiple of 16)")
    ap.add_argument("-s", "--seed", type=int, default=42)
    ap.add_argument("--non-msr4-per-col", type=int, default=0,
                    help="Non-MSR-4 tiles per column (<=1 per 16-row tile) ; 0 = pure main path")
    ap.add_argument("--qmult", type=int, default=1,
                    help="per-tensor requant multiplier M0 (match PATTERN QMULT)")
    ap.add_argument("--qshift", type=int, default=0,
                    help="per-tensor requant right-shift n (match PATTERN QSHIFT)")
    ap.add_argument("-o", "--outdir", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir or os.path.normpath(os.path.join(here, "..", "pattern"))
    os.makedirs(outdir, exist_ok=True)

    w_rows, a_rows, g_rows, meta, _ = gen_case(
        args.M, args.K, args.N, args.non_msr4_per_col, args.qmult, args.qshift, args.seed)

    write_dat(os.path.join(outdir, "weight_raw.dat"), w_rows, W_W * ARRAY_S)
    write_dat(os.path.join(outdir, "activation.dat"), a_rows, A_W * ARRAY_S)
    write_dat(os.path.join(outdir, "golden.dat"), g_rows, A_W * ARRAY_S)

    print("=" * 60)
    print("  TPU golden : M={M} K={K} N={N} (tiles {MT}x{KT}x{NT})".format(**meta))
    print("  requant    : M0={qmult} n={qshift}".format(**meta))
    print("  raw C range: [{raw_min}, {raw_max}]".format(**meta))
    print("  ReLU'd to 0: {}/{}".format(meta["relu0"], meta["M"] * meta["N"]))
    print("  saturated  : {}/{}".format(meta["saturated"], meta["M"] * meta["N"]))
    print("  file rows  : weight={w_rows} activation={a_rows} golden={r_rows}".format(**meta))
    print("  DUT beats  : weight={w_beats} activation={a_beats} result={r_beats}".format(**meta))
    print("  wrote -> {}".format(outdir))
    print("=" * 60)


if __name__ == "__main__":
    main()
