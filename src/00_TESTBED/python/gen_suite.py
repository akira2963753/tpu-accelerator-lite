#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# gen_suite.py : table-driven verification-suite generator for the TPU top.
#
# Edit TEST_TABLE below (the single source of truth), then run once :
#     python3 python/gen_suite.py           # default tier (fast, ~3.5M cycles)
#     python3 python/gen_suite.py --big     # + the 1024^3 case (~22M cycles)
# It emits, for every row :
#     pattern/tNN_weight.dat  tNN_activation.dat  tNN_golden.dat
# plus a manifest the testbench includes :
#     test_suite.svh          (NUM_TESTS + per-test param arrays)
#     suite_summary.txt       (human-readable overview)
#
# The .dat files hold only UNIQUE data ; PATTERN.sv replays them in the order
# the DUT asks for (weight repeats per mt, activation repeats per nt). See the
# index maps at the top of tpu_gen.py.
#
# PATTERN.sv loops over all rows, drives each job, and self-checks r_data vs
# golden. Rows with overflow=True are CMEM-overflow corners : the runner expects
# a mismatch for them (reverse judged), every other row must match exactly.
# ---------------------------------------------------------------------------
import argparse
import glob
import os
import random

import tpu_gen as tg

ARRAY_S = tg.ARRAY_S
W_W, A_W = tg.W_W, tg.A_W

# ---- must track define.vh ----
M_MAX, N_MAX, K_MAX = 1024, 1024, 4096
OBUF_D = 128                       # layer-fusion buffer depth
OBUF_TILES = OBUF_D // ARRAY_S     # = 8


def T(name, M, K, N, nm4=0, qmult=1, qshift=0, seed=1, new_seq=1, overflow=False,
      fuse_in=False, fuse_out=False, big=False):
    return dict(name=name, M=M, K=K, N=N, nm4=nm4, qmult=qmult, qshift=qshift,
                seed=seed, new_seq=new_seq, overflow=overflow,
                fuse_in=fuse_in, fuse_out=fuse_out, big=big)


# ---------------------------------------------------------------------------
# TEST_TABLE : one row = one matmul job. Fields :
#   M,K,N     dims (multiples of 16 ; M,N <= 1024 ; K <= 4096)
#   nm4       Non-MSR-4 tiles / column      qmult   requant M0
#   qshift    requant >>n (None = auto)     seed    reproducible RNG seed
#   new_seq   1 = reset DUT before job ; 0 = back-to-back (no reset)
#   overflow  True = break CMEM limit on purpose (runner expects a mismatch)
#   fuse_out  True = also write results into OBUF   (needs M<=16, N<=128)
#   fuse_in   True = read activation from OBUF      (needs M<=16, K<=128)
#             a fuse_in row must directly follow its fuse_out producer and take
#             K == producer's N, M == producer's M.
#   big       True = only emitted with --big (long-running)
# ---------------------------------------------------------------------------
TEST_TABLE = [
    # ---- basic main path (no compensation, identity requant) ----
    T("basic_16",          16,   16,   16, seed=42),
    T("basic_32",          32,   32,   32, seed=43),
    T("basic_128",        128,  128,  128, seed=44),
    T("basic_16x128x16",   16,  128,   16, seed=45),
    T("basic_128x16x128", 128,   16,  128, seed=46),
    T("basic_64x48x32",    64,   48,   32, seed=47),
    # ---- large K : UB is a streaming window now, K is bounded by ACC_W only ----
    T("bigk_512",          16,  512,   16, qshift=None, seed=48),
    T("bigk_1024",         32, 1024,   32, qshift=None, seed=49),
    T("bigk_4096",         16, 4096,   16, qshift=None, seed=50),
    # ---- large M / N : free dimensions (descriptor port width only) ----
    T("bigm_1024",       1024,   16,   16, seed=51),
    T("bign_1024",         16,   16, 1024, seed=52),
    T("bigmn_512",        512,   16,  512, seed=53),
    # ---- compensation : Non-MSR-4, legal (<= 1 per 16-row tile per column) ----
    T("comp_16",           16,   16,   16, nm4=1, seed=54),
    T("comp_64",           64,   64,   64, nm4=1, seed=55),
    T("comp_bigk",         32,  512,   32, nm4=2, qshift=None, seed=56),
    # ---- CMEM-overflow corner : 2 Non-MSR-4 in one tile -> expected mismatch ----
    T("comp_ovf_16",       16,   16,   16, nm4=1, seed=57, overflow=True),
    # ---- requant : non-trivial scale / saturation / ReLU / >31 shift ----
    T("rq_scale_16",       16,   16,   16, qmult=13,     qshift=7,  seed=62),
    T("rq_sat_16",         16,   16,   16, qmult=1000,   qshift=0,  seed=63),
    T("rq_relu_16",        16,   16,   16, qmult=5,      qshift=4,  seed=64),
    T("rq_shift32",        16, 4096,   16, qmult=131071, qshift=32, seed=65),
    # ---- consecutive jobs : back-to-back, no reset (tests re-entry) ----
    T("consec_a",          32,   32,   32, seed=72, new_seq=1),
    T("consec_b",          16,   64,   48, seed=73, new_seq=0),
    T("consec_c",          48,   16,   16, seed=74, new_seq=0),
    # ---- on-chip layer fusion : producer writes OBUF, consumer reads it ----
    T("fuse_src",          16,   16,   32, qshift=7, seed=80, fuse_out=True),
    T("fuse_dst",          16,   32,   16, qshift=7, seed=81, new_seq=0, fuse_in=True),
    # ---- long-running (only with --big) ----
    T("big_512",          512,  512,  512, qshift=None, seed=90),
    T("big_1024",        1024, 1024, 1024, qshift=None, seed=91, big=True),
]

# ---- random cases (reproducible via meta-seed ; qshift auto-tuned) ----
_rng = random.Random(2026)
_SIZES = [16, 32, 48, 64, 96, 128]
_KSIZES = [16, 32, 64, 128, 256, 512]
for _i in range(6):
    TEST_TABLE.append(T("rand_%d" % _i,
                        _rng.choice(_SIZES), _rng.choice(_KSIZES), _rng.choice(_SIZES),
                        nm4=_rng.choice([0, 1]),
                        qmult=_rng.randint(1, 64), qshift=None,
                        seed=100 + _i, new_seq=1))


def validate(t, row, prev):
    n = row["name"]
    for d in ("M", "K", "N"):
        assert row[d] % ARRAY_S == 0, "%s : %s must be a multiple of %d" % (n, d, ARRAY_S)
    assert 0 < row["M"] <= M_MAX, "%s : M must be in (0,%d]" % (n, M_MAX)
    assert 0 < row["N"] <= N_MAX, "%s : N must be in (0,%d]" % (n, N_MAX)
    assert 0 < row["K"] <= K_MAX, "%s : K must be in (0,%d]" % (n, K_MAX)
    assert row["qshift"] is None or row["qshift"] >= 0, "%s : qshift must be >= 0" % n
    assert 0 <= row["qmult"] < (1 << 17), "%s : qmult must fit signed QMULT_W" % n
    if not row["overflow"] and row["nm4"] > 0:
        assert row["nm4"] <= row["K"] // ARRAY_S, \
            "%s : nm4 (%d) exceeds tile count (%d)" % (n, row["nm4"], row["K"] // ARRAY_S)
    # ---- fusion legality (OBUF is 8 x 16 rows and the write address ignores mt) ----
    if row["fuse_out"]:
        assert row["M"] <= ARRAY_S, "%s : fuse_out needs M <= %d" % (n, ARRAY_S)
        assert row["N"] <= OBUF_D, "%s : fuse_out needs N <= %d" % (n, OBUF_D)
    if row["fuse_in"]:
        assert row["M"] <= ARRAY_S, "%s : fuse_in needs M <= %d" % (n, ARRAY_S)
        assert row["K"] <= OBUF_D, "%s : fuse_in needs K <= %d" % (n, OBUF_D)
        assert prev is not None and prev["fuse_out"], \
            "%s : fuse_in must directly follow a fuse_out row" % n
        assert row["K"] == prev["N"], \
            "%s : fuse_in K (%d) must equal producer N (%d)" % (n, row["K"], prev["N"])
        assert row["M"] == prev["M"], \
            "%s : fuse_in M (%d) must equal producer M (%d)" % (n, row["M"], prev["M"])
        assert row["new_seq"] == 0, "%s : fuse_in must not reset the DUT" % n


def _arr(name, vals):
    return "    localparam int %s [NUM_TESTS] = '{%s};\n" % (name, ", ".join(str(v) for v in vals))


def _emit_svh(path, metas):
    mw = max(m["w_rows"] for m in metas)
    ma = max(m["a_rows"] for m in metas)
    mr = max(m["r_rows"] for m in metas)
    with open(path, "w", newline="\n") as f:
        f.write("// AUTO-GENERATED by gen_suite.py -- do not edit by hand.\n")
        f.write("// Regenerate : python3 python/gen_suite.py [--big]\n\n")
        f.write("    localparam int NUM_TESTS  = %d;\n" % len(metas))
        f.write("    // .dat file lengths (unique data ; PATTERN replays them)\n")
        f.write("    localparam int MAX_W_ROWS = %d;\n" % mw)
        f.write("    localparam int MAX_A_ROWS = %d;\n" % ma)
        f.write("    localparam int MAX_R_ROWS = %d;\n" % mr)
        f.write(_arr("TEST_M", [m["M"] for m in metas]))
        f.write(_arr("TEST_K", [m["K"] for m in metas]))
        f.write(_arr("TEST_N", [m["N"] for m in metas]))
        f.write(_arr("TEST_QMULT", [m["qmult"] for m in metas]))
        f.write(_arr("TEST_QSHIFT", [m["qshift"] for m in metas]))
        f.write(_arr("TEST_NEWSEQ", [m["new_seq"] for m in metas]))
        f.write(_arr("TEST_XMISS", [1 if m["overflow"] else 0 for m in metas]))  # 1 = expect mismatch
        f.write(_arr("TEST_FIN", [1 if m["fuse_in"] else 0 for m in metas]))
        f.write(_arr("TEST_FOUT", [1 if m["fuse_out"] else 0 for m in metas]))
        names = ", ".join('"%s"' % m["name"] for m in metas)
        f.write("    localparam string TEST_NAME [NUM_TESTS] = '{%s};\n" % names)


def _cycles(m):
    """rough DUT cycle estimate : per k-tile 17 LOADING + 17 PRELOAD + 52 CAL."""
    per_tile = 17 + 17 + (ARRAY_S + 2 * ARRAY_S + 4)
    return m["MT"] * m["NT"] * (m["KT"] * per_tile + ARRAY_S)


def _emit_summary(path, metas):
    total = 0
    with open(path, "w", newline="\n") as f:
        f.write("TPU verification suite  (%d cases)\n" % len(metas))
        f.write("=" * 100 + "\n")
        f.write("%-3s %-18s %5s %5s %5s %4s %7s %6s %4s %4s %7s %7s %10s %-8s\n" %
                ("id", "name", "M", "K", "N", "nm4", "qmult", "qshift", "seq",
                 "fuse", "beats", "sat", "cycles", "expect"))
        for t, m in enumerate(metas):
            c = _cycles(m)
            total += c
            fuse = ("O" if m["fuse_out"] else "-") + ("I" if m["fuse_in"] else "-")
            f.write("%-3d %-18s %5d %5d %5d %4d %7d %6d %4d %4s %7d %7d %10d %-8s\n" % (
                t, m["name"], m["M"], m["K"], m["N"], m["non_msr4_per_col"],
                m["qmult"], m["qshift"], m["new_seq"], fuse, m["r_beats"],
                m["saturated"], c, "MISMATCH" if m["overflow"] else "pass"))
        f.write("=" * 100 + "\n")
        f.write("total estimated DUT cycles : %d  (~%.1f ms sim time @ 10ns)\n"
                % (total, total * 10 / 1e6))
    return total


def main():
    ap = argparse.ArgumentParser(description="TPU verification suite generator")
    ap.add_argument("--big", action="store_true",
                    help="also emit the long-running cases (1024^3, ~22M cycles)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    pat = os.path.normpath(os.path.join(here, "..", "pattern"))
    svh = os.path.normpath(os.path.join(here, "..", "test_suite.svh"))
    summ = os.path.normpath(os.path.join(here, "..", "suite_summary.txt"))
    os.makedirs(pat, exist_ok=True)

    # drop stale tNN_*.dat from a previous (differently sized) suite
    for old in glob.glob(os.path.join(pat, "t[0-9][0-9]_*.dat")):
        os.remove(old)

    table = [r for r in TEST_TABLE if args.big or not r["big"]]

    metas = []
    prev_out = None          # quantised output of the last fuse_out job
    prev_row = None
    for t, row in enumerate(table):
        validate(t, row, prev_row)
        A_in = prev_out if row["fuse_in"] else None
        w_rows, a_rows, g_rows, meta, Cp = tg.gen_case(
            row["M"], row["K"], row["N"], row["nm4"],
            row["qmult"], row["qshift"], row["seed"], row["overflow"], A_in)
        tg.write_dat(os.path.join(pat, "t%02d_weight.dat" % t), w_rows, W_W * ARRAY_S)
        tg.write_dat(os.path.join(pat, "t%02d_activation.dat" % t), a_rows, A_W * ARRAY_S)
        tg.write_dat(os.path.join(pat, "t%02d_golden.dat" % t), g_rows, A_W * ARRAY_S)
        meta.update(name=row["name"], new_seq=row["new_seq"],
                    fuse_in=row["fuse_in"], fuse_out=row["fuse_out"])
        metas.append(meta)
        prev_out = Cp if row["fuse_out"] else None
        prev_row = row
        print("  [%2d] %-18s M=%-5d K=%-5d N=%-5d qshift=%-2d rows w/a/g = %d/%d/%d"
              % (t, meta["name"], meta["M"], meta["K"], meta["N"], meta["qshift"],
                 meta["w_rows"], meta["a_rows"], meta["r_rows"]))

    _emit_svh(svh, metas)
    total = _emit_summary(summ, metas)
    print("=" * 64)
    print("  generated %d cases -> %s" % (len(metas), pat))
    print("  manifest  -> %s" % svh)
    print("  summary   -> %s" % summ)
    print("  estimated DUT cycles : %d (~%.1f ms @ 10ns)" % (total, total * 10 / 1e6))
    if not any(r["big"] for r in table):
        print("  (run with --big to add the 1024^3 case)")
    print("=" * 64)


if __name__ == "__main__":
    main()
