#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# check_replay.py : verify the .dat replay contract between the generator and
# PATTERN.sv, without needing a simulator.
#
#     cd src/00_TESTBED && python3 python/check_replay.py
#
# The .dat files hold only unique data, but the DUT re-fetches the same weight
# tile once per mt and the same activation tile once per nt. PATTERN.sv folds
# that back with two index maps. This script re-implements those maps exactly
# and checks EVERY fed beat against the matrix the DUT should be seeing, so a
# stream-order bug is caught in a second instead of after a 30-minute VCS run.
#
# Run it after touching either tpu_gen.gen_case's row ordering or PATTERN's
# feed_w / feed_a.
# ---------------------------------------------------------------------------
import sys
import random

import tpu_gen as tg

ARRAY_S = tg.ARRAY_S


def unpack(v, lane_w, lanes):
    m = (1 << lane_w) - 1
    return [(v >> (lane_w * i)) & m for i in range(lanes)]


def check(M, K, N, seed, nm4=0):
    MT, KT, NT = M // ARRAY_S, K // ARRAY_S, N // ARRAY_S
    w_rows, a_rows, g_rows, meta, Cp = tg.gen_case(M, K, N, nm4, 1, 0, seed)

    # re-derive the same A / W the generator used (same seed, same call order)
    rng = random.Random(seed)
    A = tg.gen_A(rng, M, K)
    W = tg.gen_W(rng, K, N, nm4)

    err = 0

    # ---- weight : PATTERN does mem_w[i % (NT*KT*16)] over MT*NT*KT*16 beats ----
    period = NT * KT * ARRAY_S
    assert len(w_rows) == period, "weight file %d rows, expected %d" % (len(w_rows), period)
    i = 0
    for mt in range(MT):
        for nt in range(NT):
            for kt in range(KT):
                for r in range(ARRAY_S):
                    got = unpack(w_rows[i % period], tg.W_W, ARRAY_S)
                    exp = [int(W[kt * ARRAY_S + r][nt * ARRAY_S + c]) & 0xFF
                           for c in range(ARRAY_S)]
                    if got != exp:
                        err += 1
                        if err < 4:
                            print("    W beat %d (mt%d nt%d kt%d r%d) mismatch"
                                  % (i, mt, nt, kt, r))
                    i += 1
    w_beats = i

    # ---- activation : idx = mt*KT*16 + kt*16 + m over MT*NT*KT*16 beats ----
    assert len(a_rows) == MT * KT * ARRAY_S, \
        "activation file %d rows, expected %d" % (len(a_rows), MT * KT * ARRAY_S)
    i = 0
    for mt in range(MT):
        for nt in range(NT):
            for kt in range(KT):
                for m in range(ARRAY_S):
                    # exactly PATTERN.sv feed_a's arithmetic
                    mm = i % ARRAY_S
                    ktv = (i // ARRAY_S) % KT
                    mtv = i // (ARRAY_S * KT * NT)
                    idx = mtv * KT * ARRAY_S + ktv * ARRAY_S + mm
                    assert (mm, ktv, mtv) == (m, kt, mt), \
                        "index decode broke at beat %d" % i
                    got = unpack(a_rows[idx], tg.A_W, ARRAY_S)
                    exp = [int(A[mt * ARRAY_S + m][kt * ARRAY_S + c]) & 0x7F
                           for c in range(ARRAY_S)]
                    if got != exp:
                        err += 1
                        if err < 4:
                            print("    A beat %d idx %d (mt%d nt%d kt%d m%d) mismatch"
                                  % (i, idx, mt, nt, kt, m))
                    i += 1
    a_beats = i

    # ---- golden : straight sequential, mt -> nt -> m ----
    assert len(g_rows) == MT * NT * ARRAY_S
    i = 0
    for mt in range(MT):
        for nt in range(NT):
            for m in range(ARRAY_S):
                got = unpack(g_rows[i], tg.A_W, ARRAY_S)
                exp = [int(Cp[mt * ARRAY_S + m][nt * ARRAY_S + c]) & 0x7F
                       for c in range(ARRAY_S)]
                if got != exp:
                    err += 1
                i += 1
    r_beats = i

    print("  [%s] M=%-5d K=%-5d N=%-5d  fed w/a/r = %d/%d/%d   file rows = %d/%d/%d"
          % ("OK " if err == 0 else "ERR", M, K, N, w_beats, a_beats, r_beats,
             len(w_rows), len(a_rows), len(g_rows)))
    return err


def main():
    print("=== .dat replay contract check ===")
    total = 0
    total += check(16, 16, 16, 42)
    total += check(32, 32, 32, 43)
    total += check(64, 48, 32, 47)
    total += check(16, 128, 16, 45)
    total += check(128, 16, 128, 46)
    total += check(16, 16, 128, 52)
    total += check(128, 16, 16, 51)
    total += check(48, 64, 32, 55, nm4=1)
    total += check(32, 256, 48, 56, nm4=1)
    print("==================================")
    if total == 0:
        print("  PASS")
        return 0
    print("  FAIL : %d mismatches" % total)
    return 1


if __name__ == "__main__":
    sys.exit(main())
