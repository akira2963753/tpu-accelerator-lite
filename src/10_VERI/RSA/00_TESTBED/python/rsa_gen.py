import os
import random
import argparse
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Design parameters
# ---------------------------------------------------------------------------
ARRAY_S = 16      # Array size
W_W     = 8       # Weight width
A_W     = 7       # Activation width
RW_W    = 5       # Reduced weight width
PSUM_W  = 8 + (A_W + 1) + 4   # Accumulator width

MUL_W   = (RW_W - 1) + (A_W + 1)   # = 12
RES_W   = W_W + (A_W + 1)          # = 16


# ---------------------------------------------------------------------------
# Bit helpers
# ---------------------------------------------------------------------------
def mask(v, bits):
    return v & ((1 << bits) - 1)


def signed(v, bits):
    """Interpret an unsigned value as signed."""
    v = mask(v, bits)
    if v & (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def sext(v, from_bits, to_bits):
    """Sign-extend an unsigned value."""
    v = mask(v, from_bits)
    if v & (1 << (from_bits - 1)):
        v |= (((1 << (to_bits - from_bits)) - 1) << from_bits)
    return mask(v, to_bits)


# ---------------------------------------------------------------------------
# Weight encoding
# ---------------------------------------------------------------------------
def is_msr4(w8):
    """Return whether a weight is MSR-4."""
    w8 = mask(w8, 8)
    top = (w8 >> 4) & 0xF
    and4 = 1 if top == 0xF else 0
    or4 = 1 if top != 0x0 else 0
    non_msr4 = and4 ^ or4
    return non_msr4 == 0


def wpu_encode(w8):
    """Encode one reduced weight."""
    w8 = mask(w8, 8)
    if is_msr4(w8):
        # MSR-4 path
        return (0 << 4) | ((w8 >> 1) & 0xF)
    else:
        # Non-MSR-4 path
        return (1 << 4) | ((w8 >> 4) & 0xF)


# ---------------------------------------------------------------------------
# Reduced MAC
# ---------------------------------------------------------------------------
def reduced_mac_contrib(rw5, act7):
    rw5 = mask(rw5, RW_W)
    # Add guard bit
    act8 = mask((mask(act7, A_W) << 1) | 1, 8)
    act_sign = (act8 >> 7) & 1

    w_bit4 = (rw5 >> 4) & 1
    w_bit3 = (rw5 >> 3) & 1
    w4 = rw5 & 0xF   # Low bits

    weight_sign_add = w_bit3 & (1 - w_bit4)
    if w_bit3:
        weight_i = mask((~w4) + weight_sign_add, 4)   # Signed magnitude
    else:
        weight_i = w4

    if act_sign:
        activation_i = mask((~act8) + 1, 8)            # Absolute value
    else:
        activation_i = act8

    mul_result = mask(activation_i * weight_i, MUL_W)  # Magnitude

    if act_sign ^ w_bit3:
        sign_result = mask((~mul_result) + 1, MUL_W)   # Negate
    else:
        sign_result = mul_result

    shift_result = mask(sign_result << 1, MUL_W + 1)   # x2
    act8_sext13 = sext(act8, 8, MUL_W + 1)             # Sign-extend
    msr4_result = mask(shift_result + act8_sext13, MUL_W + 1)   # MSR-4 result
    no_msr4_result = mask(shift_result << 3, RES_W)    # Non-MSR-4 result

    if w_bit4:                                         # Non-MSR-4
        result = no_msr4_result
    else:                                              # MSR-4
        result = sext(msr4_result, MUL_W + 1, RES_W)   # Sign-extend

    return signed(result, RES_W)                       # PE contribution


def full_mac_contrib(w8, act7):
    """Reference 8-bit MAC."""
    a8s = signed((mask(act7, A_W) << 1) | 1, 8)        # Guard bit
    w8s = signed(mask(w8, 8) | 1, 8)                   # Force LSB
    return a8s * w8s


# ---------------------------------------------------------------------------
# Array model
# ---------------------------------------------------------------------------
def reduced_array(RW, A):
    out = [[0] * ARRAY_S for _ in range(ARRAY_S)]
    for m in range(ARRAY_S):
        for c in range(ARRAY_S):
            acc = 0
            for r in range(ARRAY_S):
                acc += reduced_mac_contrib(RW[r][c], A[m][r])
            out[m][c] = signed(mask(acc, PSUM_W), PSUM_W)   # Hardware width
    return out


def full_array(W, A):
    out = [[0] * ARRAY_S for _ in range(ARRAY_S)]
    for m in range(ARRAY_S):
        for c in range(ARRAY_S):
            acc = 0
            for r in range(ARRAY_S):
                acc += full_mac_contrib(W[r][c], A[m][r])
            out[m][c] = acc                                 # Full precision
    return out


# ---------------------------------------------------------------------------
# Random data
# ---------------------------------------------------------------------------
def rand_weight(non_msr4_prob):
    """Generate mostly MSR-4 weights."""
    if random.random() < non_msr4_prob:
        w = random.randint(-128, 127)           # Likely Non-MSR-4
        while is_msr4(mask(w, 8)):              # Enforce Non-MSR-4
            w = random.randint(-128, 127)
    else:
        w = random.randint(-16, 15)             # MSR-4 range
    return mask(w, 8)


def rand_activation():
    return mask(random.randint(-64, 63), A_W)   # Signed 7-bit


def gen_matrices(non_msr4_prob):
    W = [[rand_weight(non_msr4_prob) for _ in range(ARRAY_S)] for _ in range(ARRAY_S)]
    A = [[rand_activation() for _ in range(ARRAY_S)] for _ in range(ARRAY_S)]
    RW = [[wpu_encode(W[r][c]) for c in range(ARRAY_S)] for r in range(ARRAY_S)]
    return W, A, RW


# ---------------------------------------------------------------------------
# File I/O
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
# Reports
# ---------------------------------------------------------------------------
def per_mac_report(all_W, all_A):
    """Report MAC accuracy."""
    msr4_n = msr4_exact = 0
    non_n = 0
    non_abs = 0
    non_max = 0
    for W, A in zip(all_W, all_A):
        acts = [v for row in A for v in row]
        for row in W:
            for w8 in row:
                rw = wpu_encode(w8)
                for act in acts[:ARRAY_S]:   # Sample one row
                    f = full_mac_contrib(w8, act)
                    r = reduced_mac_contrib(rw, act)
                    if is_msr4(w8):
                        msr4_n += 1
                        if f == r:
                            msr4_exact += 1
                    else:
                        non_n += 1
                        non_abs += abs(r - f)
                        non_max = max(non_max, abs(r - f))
    print("-" * 60)
    print("  Per-MAC breakdown")
    print("-" * 60)
    if msr4_n:
        print("  MSR-4     : {}/{} exact ({:.2f}%)".format(
            msr4_exact, msr4_n, 100.0 * msr4_exact / msr4_n))
    if non_n:
        print("  Non-MSR-4 : {} macs, MAE {:.2f}, max abs {}".format(
            non_n, non_abs / non_n, non_max))
    else:
        print("  Non-MSR-4 : (none generated)")


def accuracy_report(all_full, all_reduced):
    diffs = []
    for of, orr in zip(all_full, all_reduced):
        for row_f, row_r in zip(of, orr):
            for vf, vr in zip(row_f, row_r):
                diffs.append(vr - vf)
    n = len(diffs)
    mae = sum(abs(d) for d in diffs) / n
    max_abs = max(abs(d) for d in diffs)
    sum_abs_full = sum(abs(v) for of in all_full for row in of for v in row)
    rel = (sum(abs(d) for d in diffs) / sum_abs_full) if sum_abs_full else 0.0
    exact = sum(1 for d in diffs if d == 0)

    print("=" * 60)
    print("  Full vs Reduced Accuracy Report")
    print("=" * 60)
    print("  output elements : {}".format(n))
    print("  exact match     : {} ({:.2f}%)".format(exact, 100.0 * exact / n))
    print("  MAE             : {:.4f}".format(mae))
    print("  max abs error   : {}".format(max_abs))
    print("  relative error  : {:.4f}%".format(100.0 * rel))
    print("=" * 60)
    return {"mae": mae, "max_abs": max_abs, "rel": rel, "exact": exact, "n": n}


def plot_error(all_full, all_reduced, stats):
    """Plot reduced-MAC error."""

    full = [v for of in all_full for row in of for v in row]
    reduced = [v for orr in all_reduced for row in orr for v in row]
    err = [r - f for f, r in zip(full, reduced)]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    lo = min(min(full), min(reduced))
    hi = max(max(full), max(reduced))
    ax0.scatter(full, reduced, s=6, alpha=0.35, edgecolors="none")
    ax0.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x (ideal)")
    ax0.set_xlabel("Full (8b x 8b, LSB=1)")
    ax0.set_ylabel("Reduced (MSR-4)")
    ax0.set_title("Full vs Reduced output")
    ax0.legend(loc="upper left")
    ax0.grid(True, alpha=0.3)

    # Error histogram
    ax1.hist(err, bins=61, color="#4472c4", edgecolor="white", linewidth=0.3)
    ax1.axvline(0, color="r", linestyle="--", linewidth=1)
    ax1.set_xlabel("Error  (reduced - full)")
    ax1.set_ylabel("Count")
    ax1.set_title("Error distribution")
    ax1.grid(True, alpha=0.3)

    fig.suptitle(
        "RSA Reduced Systolic Array : {} outputs, exact {:.1f}%, "
        "MAE {:.1f}, max |err| {}, rel {:.2f}%".format(
            stats["n"], 100.0 * stats["exact"] / stats["n"],
            stats["mae"], stats["max_abs"], 100.0 * stats["rel"]),
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="RSA reduced systolic array golden generator")
    ap.add_argument("-n", "--num", type=int, default=500, help="組數 N (預設 10)")
    ap.add_argument("-s", "--seed", type=int, default=42, help="亂數種子")
    ap.add_argument("--non-msr4-prob", type=float, default=0.02,
                    help="Non-MSR-4 weight 注入機率 (預設 0.02, 偏 MSR-4 貼近真實)")
    ap.add_argument("-o", "--outdir", default=None,
                    help="輸出 .dat 目錄 (預設 ../00_TESTBED)")
    args = ap.parse_args()

    random.seed(args.seed)

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir or os.path.normpath(
        os.path.join(here, "..", "00_TESTBED", "pattern"))
    os.makedirs(outdir, exist_ok=True)

    w_rows, a_rows, g_rows = [], [], []
    all_full, all_reduced = [], []
    all_W, all_A = [], []

    for _ in range(args.num):
        W, A, RW = gen_matrices(args.non_msr4_prob)
        out_full = full_array(W, A)
        out_reduced = reduced_array(RW, A)
        all_full.append(out_full)
        all_reduced.append(out_reduced)
        all_W.append(W)
        all_A.append(A)

        # Reduced weights
        for r in range(ARRAY_S):
            w_rows.append(pack_row(RW[r], RW_W))
        # Activations
        for m in range(ARRAY_S):
            a_rows.append(pack_row(A[m], A_W))
        # Golden output
        for m in range(ARRAY_S):
            g_rows.append(pack_row([mask(v, PSUM_W) for v in out_reduced[m]], PSUM_W))

    stats = accuracy_report(all_full, all_reduced)
    per_mac_report(all_W, all_A)

    plot_error(all_full, all_reduced, stats)

    write_dat(os.path.join(outdir, "weight_reduced.dat"), w_rows, RW_W * ARRAY_S)
    write_dat(os.path.join(outdir, "activation.dat"), a_rows, A_W * ARRAY_S)
    write_dat(os.path.join(outdir, "golden.dat"), g_rows, PSUM_W * ARRAY_S)

    print("  wrote {} lines each -> {}".format(len(w_rows), outdir))
    print("    weight_reduced.dat  ({} hex digits / line)".format((RW_W * ARRAY_S + 3) // 4))
    print("    activation.dat      ({} hex digits / line)".format((A_W * ARRAY_S + 3) // 4))
    print("    golden.dat          ({} hex digits / line)".format((PSUM_W * ARRAY_S + 3) // 4))


if __name__ == "__main__":
    main()
