#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# tpu_gen.py : simple smoke-test input + golden generator for the TPU top.
#
# golden = C = sum_k (2A+1)*(W|1)  (then OP : ReLU + quant).
# activation gets the RPE guard bit ({a,1'b1} = 2a+1) ; weight's LSB is the
# expected-value 1 (w|1). Matches the reduced MAC (== RSA gen full_mac_contrib).
#
# Weights are kept MSR-4 by default (small magnitude) so no compensation fires;
# --non-msr4-per-col injects a few large weights (<= 1 per 64 rows per column,
# the CMEM/compensation limit) to exercise the compensation path later.
#
# Emits three .dat (hex, one beat per line, lane 0 in LSB -> matches bus [w*i +: w]):
#   weight_raw.dat  : stream order mt->nt->kt->16 k-rows ; beat = 16 x 8-bit  (W_RAW_BW=128)
#   activation.dat  : stream order mt->kt->m=0..15       ; beat = 16 x 7-bit  (A_BW=112)
#   golden.dat      : output order mt->nt->m=0..15        ; beat = 16 x 7-bit  (A_BW=112)
# ---------------------------------------------------------------------------
import os
import random
import argparse

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


def lsb1(v, bits):
    """expected-value compensation : force LSB to 1, return signed value."""
    return signed(mask(v, bits) | 1, bits)


def act_ext(a):
    """RPE activation extension {a,1'b1} = 2a+1, signed (A_W+1 bits)."""
    return signed((mask(a, A_W) << 1) | 1, A_W + 1)


def is_msr4(w8):
    """W[7:4] all-0 or all-1 -> MSR-4 (no compensation needed)."""
    top = (mask(w8, 8) >> 4) & 0xF
    non_msr4 = (1 if top == 0xF else 0) ^ (1 if top != 0x0 else 0)
    return non_msr4 == 0


def op_process(v, qmult, qshift):
    """OP.sv model : ReLU -> * M0 -> >> n (truncate) -> saturate to [0, SAT_MAX].
    Python `>>` on signed ints floors toward -inf, matching SV arithmetic `>>>`."""
    if v < 0:
        v = 0
    v = (v * qmult) >> qshift
    return SAT_MAX if v > SAT_MAX else v


# ---------------------------------------------------------------------------
# data generation
# ---------------------------------------------------------------------------
def gen_A(M, K):
    # full-range signed 7-bit activations
    return [[random.randint(-(1 << (A_W - 1)), (1 << (A_W - 1)) - 1) for _ in range(K)]
            for _ in range(M)]


def gen_W(K, N, non_msr4_per_col):
    # default : MSR-4-only weights (top nibble all-0 or all-1) -> signed [-16, 15],
    # so no compensation is needed and golden == hardware for every weight.
    W = [[random.randint(-16, 15) & 0xFF for _ in range(N)] for _ in range(K)]
    # optional : inject full-range Non-MSR-4 weights, at most 1 per 64 rows per column
    # (the CMEM/compensation limit) to exercise the compensation path.
    if non_msr4_per_col > 0:
        n_inj = min(non_msr4_per_col, max(1, K // 64))
        for n in range(N):
            for r in random.sample(range(K), n_inj):
                w = random.randint(-128, 127) & 0xFF
                while is_msr4(w):
                    w = random.randint(-128, 127) & 0xFF
                W[r][n] = w
    return W


# ---------------------------------------------------------------------------
# packing / writing
# ---------------------------------------------------------------------------
def pack_row(values, lane_w):
    packed = 0
    for i, v in enumerate(values):
        packed |= mask(v, lane_w) << (lane_w * i)
    return packed


def hex_str(value, total_bits):
    return format(value, "0{}x".format((total_bits + 3) // 4))


def write_dat(path, rows, total_bits):
    with open(path, "w") as f:
        for row in rows:
            f.write(hex_str(row, total_bits) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="TPU smoke-test golden generator")
    ap.add_argument("-M", type=int, default=16, help="rows of A/C (multiple of 16)")
    ap.add_argument("-K", type=int, default=16, help="contraction dim (multiple of 16)")
    ap.add_argument("-N", type=int, default=16, help="cols of W/C (multiple of 16)")
    ap.add_argument("-s", "--seed", type=int, default=42)
    ap.add_argument("--non-msr4-per-col", type=int, default=0,
                    help="Non-MSR-4 weights per column (<=1 per 64 rows) ; 0 = pure main path")
    ap.add_argument("--qmult", type=int, default=1,
                    help="per-tensor requant multiplier M0 (match PATTERN QMULT)")
    ap.add_argument("--qshift", type=int, default=0,
                    help="per-tensor requant right-shift n (match PATTERN QSHIFT)")
    ap.add_argument("-o", "--outdir", default=None)
    args = ap.parse_args()

    assert args.M % ARRAY_S == 0 and args.K % ARRAY_S == 0 and args.N % ARRAY_S == 0, \
        "M/K/N must be multiples of ARRAY_S (16)"
    random.seed(args.seed)
    M, K, N = args.M, args.K, args.N
    MT, KT, NT = M // ARRAY_S, K // ARRAY_S, N // ARRAY_S

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir or os.path.normpath(os.path.join(here, "..", "pattern"))
    os.makedirs(outdir, exist_ok=True)

    A = gen_A(M, K)
    W = gen_W(K, N, args.non_msr4_per_col)

    # golden : sum_k (2A+1)*(W|1), then OP (ReLU + quant)
    C = [[sum(act_ext(A[m][k]) * lsb1(W[k][n], W_W) for k in range(K))
          for n in range(N)] for m in range(M)]
    Cp = [[op_process(C[m][n], args.qmult, args.qshift) for n in range(N)] for m in range(M)]

    # ---- stream-order .dat ----
    w_rows, a_rows, g_rows = [], [], []
    # weight_raw : mt -> nt -> kt -> 16 k-rows (re-sent per mt)
    for mt in range(MT):
        for nt in range(NT):
            for kt in range(KT):
                for r in range(ARRAY_S):
                    vals = [W[kt * ARRAY_S + r][nt * ARRAY_S + c] for c in range(ARRAY_S)]
                    w_rows.append(pack_row(vals, W_W))
    # activation : mt -> kt -> m=0..15  (UB addr = kt*16 + m)
    for mt in range(MT):
        for kt in range(KT):
            for m in range(ARRAY_S):
                vals = [A[mt * ARRAY_S + m][kt * ARRAY_S + c] for c in range(ARRAY_S)]
                a_rows.append(pack_row(vals, A_W))
    # golden : mt -> nt -> m=0..15
    for mt in range(MT):
        for nt in range(NT):
            for m in range(ARRAY_S):
                vals = [Cp[mt * ARRAY_S + m][nt * ARRAY_S + c] for c in range(ARRAY_S)]
                g_rows.append(pack_row(vals, A_W))

    write_dat(os.path.join(outdir, "weight_raw.dat"), w_rows, W_W * ARRAY_S)
    write_dat(os.path.join(outdir, "activation.dat"), a_rows, A_W * ARRAY_S)
    write_dat(os.path.join(outdir, "golden.dat"), g_rows, A_W * ARRAY_S)

    # ---- report ----
    raw = [C[m][n] for m in range(M) for n in range(N)]
    sat = sum(1 for m in range(M) for n in range(N)
              if (max(0, C[m][n]) * args.qmult) >> args.qshift > SAT_MAX)  # post-requant clip
    relu0 = sum(1 for m in range(M) for n in range(N) if C[m][n] <= 0)
    distinct = len({Cp[m][n] for m in range(M) for n in range(N)})
    print("=" * 56)
    print("  TPU smoke golden : M={} K={} N={} (tiles {}x{}x{})".format(M, K, N, MT, KT, NT))
    print("  requant          : M0={} n={}  (C_q = sat((ReLU(acc)*M0)>>n))".format(
        args.qmult, args.qshift))
    print("  raw C range      : [{}, {}]".format(min(raw), max(raw)))
    print("  ReLU'd to 0      : {}/{}".format(relu0, M * N))
    print("  saturated (>{})  : {}/{}   (clipped by OP ; raise n / lower M0 to rescale)".format(
        SAT_MAX, sat, M * N))
    print("  distinct outputs : {}".format(distinct))
    print("  beats : weight={} activation={} golden={}".format(
        len(w_rows), len(a_rows), len(g_rows)))
    print("  wrote -> {}".format(outdir))
    print("=" * 56)


if __name__ == "__main__":
    main()
