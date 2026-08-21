#!/usr/bin/env python3
"""Bit-accurate model and data packing for the reduced-precision TPU."""

import argparse
import os
import random

import tpu_command as tc


ARRAY_S = 32
W_W = 8
A_W = 4
RW_W = 5
R_W = 8
PSUM_W = 21
ACC_W = 29
QMULT_W = 18
QSHIFT_W = 6
SAT_MAX = (1 << (A_W - 1)) - 1


def mask(value, width):
    return int(value) & ((1 << width) - 1)


def signed(value, width):
    value = mask(value, width)
    return value - (1 << width) if value & (1 << (width - 1)) else value


def wrap_signed(value, width):
    return signed(value, width)


def wpu_encode(raw_weight):
    """Match WPU.sv and return {shift_flag, payload[3:0]}."""
    raw_weight = mask(raw_weight, W_W)
    top = (raw_weight >> 4) & 0xF
    non_msr = top not in (0x0, 0xF)
    payload = top if non_msr else ((raw_weight >> 1) & 0xF)
    return (int(non_msr) << 4) | payload


def decode_weight(raw_weight):
    code = wpu_encode(raw_weight)
    value_5b = signed(((code & 0xF) << 1) | 1, RW_W)
    shift = 6 if code & 0x10 else 3
    return value_5b, shift


def decode_activation(payload):
    return signed((mask(payload, A_W) << 1) | 1, A_W + 1)


def rpe_contribution(activation_payload, raw_weight):
    activation_5b = decode_activation(activation_payload)
    weight_5b, shift = decode_weight(raw_weight)
    return wrap_signed((activation_5b * weight_5b) << shift, 16)


def op_model(acc_value, qmult, qshift):
    """Match OP.sv for one lane and return a 4-bit UB payload."""
    acc_value = wrap_signed(acc_value, ACC_W)
    qmult = wrap_signed(qmult, QMULT_W)
    relu = 0 if acc_value < 0 else acc_value
    product = wrap_signed(relu * qmult, ACC_W + QMULT_W)
    quantized = product >> int(qshift)
    return SAT_MAX if quantized > SAT_MAX else mask(quantized, A_W)


def matmul_acc_model(activation, weight):
    """Model tiled RSA PSUM and ACC truncation before output processing."""
    m_dim = len(activation)
    k_dim = len(activation[0])
    n_dim = len(weight[0])
    assert all(len(row) == k_dim for row in activation)
    assert len(weight) == k_dim
    assert all(len(row) == n_dim for row in weight)
    assert k_dim % ARRAY_S == 0

    output = [[0 for _ in range(n_dim)] for _ in range(m_dim)]
    for m_idx in range(m_dim):
        for n_idx in range(n_dim):
            acc = 0
            for k_base in range(0, k_dim, ARRAY_S):
                psum = 0
                for offset in range(ARRAY_S):
                    k_idx = k_base + offset
                    term = rpe_contribution(
                        activation[m_idx][k_idx],
                        weight[k_idx][n_idx],
                    )
                    psum = wrap_signed(psum + term, PSUM_W)
                acc = wrap_signed(acc + psum, ACC_W)
            output[m_idx][n_idx] = acc
    return output


def matmul_model(activation, weight, qmult, qshift):
    acc = matmul_acc_model(activation, weight)
    return [
        [op_model(value, qmult, qshift) for value in row]
        for row in acc
    ]


def auto_qshift(accumulator, qmult):
    positive = [value for row in accumulator for value in row if value > 0]
    if not positive or qmult <= 0:
        return 0
    peak = max(positive) * int(qmult)
    shift = 0
    while shift < (1 << QSHIFT_W) - 1 and (peak >> shift) > SAT_MAX:
        shift += 1
    return shift


def generate_activation(rng, m_dim, k_dim, mode="random"):
    if mode == "ramp":
        values = list(range(-8, 8))
        return [
            [mask(values[(m_idx * 3 + k_idx) % len(values)], A_W)
             for k_idx in range(k_dim)]
            for m_idx in range(m_dim)
        ]
    return [
        [mask(rng.randrange(-4, 4), A_W) for _ in range(k_dim)]
        for _ in range(m_dim)
    ]


def generate_weight(rng, k_dim, n_dim, mode="msr"):
    if mode == "mixed":
        values = [-128, -96, -65, -33, -16, -7, -1, 0,
                  1, 7, 15, 31, 63, 95, 126, 127]
        return [
            [mask(values[(k_idx * 5 + n_idx * 3) % len(values)], W_W)
             for n_idx in range(n_dim)]
            for k_idx in range(k_dim)
        ]
    return [
        [mask(rng.randrange(-8, 8), W_W) for _ in range(n_dim)]
        for _ in range(k_dim)
    ]


def pack_row(row, lane_width):
    value = 0
    for lane, item in enumerate(row):
        value |= mask(item, lane_width) << (lane * lane_width)
    return value


def unpack_row(value, lane_width, lanes=ARRAY_S):
    return [mask(value >> (lane * lane_width), lane_width) for lane in range(lanes)]


def pack_weight_tile(weight, kt, nt):
    rows = []
    for row in range(ARRAY_S):
        k_idx = kt * ARRAY_S + row
        base_n = nt * ARRAY_S
        rows.append(pack_row(weight[k_idx][base_n:base_n + ARRAY_S], W_W))
    return rows


def pack_activation_tile(activation, mt, kt):
    rows = []
    base_m = mt * ARRAY_S
    base_k = kt * ARRAY_S
    for row in range(ARRAY_S):
        rows.append(pack_row(
            activation[base_m + row][base_k:base_k + ARRAY_S],
            A_W,
        ))
    return rows


def pack_output_tile(output, mt, nt):
    rows = []
    base_m = mt * ARRAY_S
    base_n = nt * ARRAY_S
    for row in range(ARRAY_S):
        expanded = [
            mask(output[base_m + row][base_n + lane], A_W) << (R_W - A_W)
            for lane in range(ARRAY_S)
        ]
        rows.append(pack_row(expanded, R_W))
    return rows


def expand_ub_rows(payload_rows):
    rows = []
    for payload_row in payload_rows:
        payload = unpack_row(payload_row, A_W)
        rows.append(pack_row([lane << (R_W - A_W) for lane in payload], R_W))
    return rows


def write_dat(path, rows, total_bits):
    nibbles = (total_bits + 3) // 4
    with open(path, "w", newline="\n") as data_file:
        for row in rows:
            data_file.write(format(row, "0%dx" % nibbles) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate one 32x32 TPU data set")
    parser.add_argument("-M", type=int, default=32)
    parser.add_argument("-K", type=int, default=32)
    parser.add_argument("-N", type=int, default=32)
    parser.add_argument("--qmult", type=int, default=1)
    parser.add_argument("--qshift", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-weight", action="store_true")
    parser.add_argument("--readback", action="store_true")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    assert args.M % ARRAY_S == 0
    assert args.K % ARRAY_S == 0
    assert args.N % ARRAY_S == 0
    rng = random.Random(args.seed)
    activation = generate_activation(rng, args.M, args.K)
    weight = generate_weight(
        rng, args.K, args.N, "mixed" if args.mixed_weight else "msr"
    )
    accumulator = matmul_acc_model(activation, weight)
    qshift = auto_qshift(accumulator, args.qmult) if args.qshift is None else args.qshift
    output = [
        [op_model(value, args.qmult, qshift) for value in row]
        for row in accumulator
    ]

    commands = tc.compile_gemm_trace(
        args.M, args.K, args.N, args.qmult, qshift, readback=args.readback
    )
    weight_rows = []
    activation_rows = []
    golden_rows = []
    for command in commands:
        if command.opcode == tc.CMD_OP_LOAD_W:
            weight_rows.extend(pack_weight_tile(weight, command.kt, command.nt))
        elif command.opcode == tc.CMD_OP_LOAD_A:
            activation_rows.extend(pack_activation_tile(
                activation, command.mt, command.kt
            ))
        elif command.opcode in (tc.CMD_OP_STORE_C, tc.CMD_OP_READ_UB):
            golden_rows.extend(pack_output_tile(output, command.mt, command.nt))

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir or os.path.normpath(os.path.join(here, "..", "pattern"))
    os.makedirs(outdir, exist_ok=True)

    write_dat(
        os.path.join(outdir, "command.dat"),
        [command.word for command in commands],
        tc.CMD_DESC_W,
    )
    write_dat(
        os.path.join(outdir, "weight.dat"),
        weight_rows,
        W_W * ARRAY_S,
    )
    write_dat(
        os.path.join(outdir, "activation.dat"),
        activation_rows,
        A_W * ARRAY_S,
    )
    write_dat(
        os.path.join(outdir, "golden.dat"),
        golden_rows,
        R_W * ARRAY_S,
    )
    print("M=%d K=%d N=%d qmult=%d qshift=%d" % (
        args.M, args.K, args.N, args.qmult, qshift
    ))
    print("commands=%d rows=%d/%d/%d" % (
        len(commands), len(weight_rows), len(activation_rows), len(golden_rows)
    ))
    print("wrote -> %s" % outdir)


if __name__ == "__main__":
    main()
