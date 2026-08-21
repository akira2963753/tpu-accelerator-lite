#!/usr/bin/env python3
"""Self-check the Python model, command compiler, and generated files."""

import os
import re
import sys
import tempfile

import gen_suite as gs
import tpu_command as tc
import tpu_gen as tg


def check_rtl_constants():
    here = os.path.dirname(os.path.abspath(__file__))
    define_path = os.path.normpath(os.path.join(here, "..", "..", "01_RTL", "define.vh"))
    with open(define_path, "r") as define_file:
        text = define_file.read()

    expected = {
        "ARRAY_S": 32,
        "W_W": 8,
        "A_W": 4,
        "RW_W": 5,
        "R_W": 8,
        "WMEM_D": 8192,
        "UB_D": 16384,
        "M_MAX": 1024,
        "N_MAX": 4096,
        "K_MAX": 8192,
        "CMD_DESC_W": 512,
        "CMD_OP_W": 3,
        "QMULT_W": 18,
        "QSHIFT_W": 6,
    }
    for name, value in expected.items():
        match = re.search(r"`define\s+%s\s+(\d+)" % name, text)
        assert match is not None, "missing RTL define %s" % name
        assert int(match.group(1)) == value, "%s differs between RTL and Python" % name

    assert tc.ARRAY_S == tg.ARRAY_S == expected["ARRAY_S"]
    assert tc.M_MAX == expected["M_MAX"]
    assert tc.N_MAX == expected["N_MAX"]
    assert tc.K_MAX == expected["K_MAX"]


def check_descriptor():
    command = tc.make_command(
        tc.CMD_OP_GEMM,
        acc_init=True,
        acc_final=True,
        qmult=-7,
        qshift=33,
        mt=31,
        kt=255,
        nt=127,
        w_slot=255,
        src_slot=511,
        dst_slot=510,
    )
    decoded = tc.decode_descriptor(command.word)
    assert decoded == {
        "opcode": tc.CMD_OP_GEMM,
        "acc_init": 1,
        "acc_final": 1,
        "qmult": -7,
        "qshift": 33,
        "mt": 31,
        "kt": 255,
        "nt": 127,
        "w_slot": 255,
        "src_slot": 511,
        "dst_slot": 510,
    }
    assert command.word >> tc.CMD_DEFINED_W == 0
    assert tc.field(command.word, 5, 3) == 0


def check_type2_primitives():
    assert tg.wpu_encode(0x00) == 0x00
    assert tg.wpu_encode(0xFF) == 0x0F
    assert tg.wpu_encode(0x40) == 0x14
    assert tg.wpu_encode(0x80) == 0x18
    assert tg.decode_weight(0x00) == (1, 3)
    assert tg.decode_weight(0xFF) == (-1, 3)
    assert tg.decode_weight(0x40) == (9, 6)
    assert tg.decode_weight(0x80) == (-15, 6)
    assert tg.decode_activation(0x0) == 1
    assert tg.decode_activation(0x7) == 15
    assert tg.decode_activation(0x8) == -15
    assert tg.decode_activation(0xF) == -1


def check_hand_calculation():
    activation = [[0 for _ in range(32)] for _ in range(1)]
    positive_weight = [[0] for _ in range(32)]
    negative_weight = [[0xFF] for _ in range(32)]

    positive_acc = tg.matmul_acc_model(activation, positive_weight)
    negative_acc = tg.matmul_acc_model(activation, negative_weight)
    assert positive_acc == [[256]]
    assert negative_acc == [[-256]]
    assert tg.op_model(positive_acc[0][0], 1, 6) == 4
    assert tg.op_model(negative_acc[0][0], 1, 0) == 0

    activation_64 = [[0 for _ in range(64)] for _ in range(1)]
    weight_64 = [[0] for _ in range(64)]
    assert tg.matmul_acc_model(activation_64, weight_64) == [[512]]
    assert tg.op_model(512, 1, 7) == 4


def check_independent_matrix():
    def direct_signed(value, width):
        value &= (1 << width) - 1
        return value - (1 << width) if value & (1 << (width - 1)) else value

    def direct_term(activation_payload, raw_weight):
        raw_weight &= 0xFF
        top = raw_weight >> 4
        if top == 0 or top == 15:
            weight_payload = (raw_weight >> 1) & 0xF
            shift = 3
        else:
            weight_payload = top
            shift = 6
        activation_5b = direct_signed(((activation_payload & 0xF) << 1) | 1, 5)
        weight_5b = direct_signed((weight_payload << 1) | 1, 5)
        return direct_signed((activation_5b * weight_5b) << shift, 16)

    activation = [
        [(m_idx * 7 + k_idx * 3) & 0xF for k_idx in range(64)]
        for m_idx in range(3)
    ]
    raw_values = [0x00, 0xFF, 0x40, 0x80, 0x07, 0xF1, 0x31, 0xA4]
    weight = [
        [raw_values[(k_idx * 5 + n_idx) % len(raw_values)] for n_idx in range(4)]
        for k_idx in range(64)
    ]

    expected = [[0 for _ in range(4)] for _ in range(3)]
    for m_idx in range(3):
        for n_idx in range(4):
            acc = 0
            for k_base in (0, 32):
                psum = 0
                for k_idx in range(k_base, k_base + 32):
                    psum = direct_signed(
                        psum + direct_term(activation[m_idx][k_idx], weight[k_idx][n_idx]),
                        21,
                    )
                acc = direct_signed(acc + psum, 29)
            expected[m_idx][n_idx] = acc

    assert tg.matmul_acc_model(activation, weight) == expected


def opcode_counts(commands):
    return {
        opcode: sum(command.opcode == opcode for command in commands)
        for opcode in tc.OPCODE_NAME
    }


def check_command_schedules():
    basic = tc.compile_gemm_trace(32, 32, 32, 1, 0)
    counts = opcode_counts(basic)
    assert len(basic) == 5
    assert counts[tc.CMD_OP_LOAD_W] == 1
    assert counts[tc.CMD_OP_LOAD_A] == 1
    assert counts[tc.CMD_OP_PRELOAD_W] == 1
    assert counts[tc.CMD_OP_GEMM] == 1
    assert counts[tc.CMD_OP_STORE_C] == 1

    reuse = tc.compile_gemm_trace(64, 32, 32, 1, 0)
    counts = opcode_counts(reuse)
    assert len(reuse) == 8
    assert counts[tc.CMD_OP_LOAD_W] == 1
    assert counts[tc.CMD_OP_LOAD_A] == 2
    assert counts[tc.CMD_OP_PRELOAD_W] == 1
    assert counts[tc.CMD_OP_GEMM] == 2
    assert counts[tc.CMD_OP_STORE_C] == 2

    multi_k = tc.compile_gemm_trace(32, 64, 32, 1, 0)
    gemm = [command for command in multi_k if command.opcode == tc.CMD_OP_GEMM]
    assert len(multi_k) == 9
    assert [command.acc_init for command in gemm] == [True, False]
    assert [command.acc_final for command in gemm] == [False, True]

    readback = tc.compile_gemm_trace(32, 32, 32, 1, 0, readback=True)
    assert len(readback) == 6
    assert readback[-1].opcode == tc.CMD_OP_READ_UB
    assert readback[-1].src_slot == readback[-2].dst_slot


def check_suite_contract():
    cases = gs.build_suite()
    assert len(cases) == 9
    names = [case["name"] for case in cases]
    assert names == [spec["name"] for spec in gs.TEST_TABLE]

    for case in cases:
        gs.validate_case(case)
        for row in case["golden_rows"]:
            assert all((lane & 0x0F) == 0 for lane in tg.unpack_row(row, tg.R_W))

    by_name = {case["name"]: case for case in cases}
    assert len(by_name["ub_load_read"]["weight_rows"]) == 0
    assert len(by_name["ub_load_read"]["golden_rows"]) == 32
    assert len(by_name["store_read_ub"]["golden_rows"]) == 64
    assert by_name["ready_stall"]["r_stall"]

    mixed_raw = by_name["gemm_signed"]["weight_rows"]
    flags = []
    for row in mixed_raw:
        flags.extend(tg.wpu_encode(raw) >> 4 for raw in tg.unpack_row(row, tg.W_W))
    assert 0 in flags and 1 in flags


def nonempty_lines(path):
    with open(path, "r") as data_file:
        return [line.strip() for line in data_file if line.strip()]


def check_generated_files():
    cases = gs.build_suite()
    with tempfile.TemporaryDirectory() as tempdir:
        gs.write_suite(tempdir, cases)
        assert os.path.isfile(os.path.join(tempdir, "test_suite.svh"))
        assert os.path.isfile(os.path.join(tempdir, "suite_summary.txt"))

        widths = {
            "command": tc.CMD_DESC_W // 4,
            "weight": tg.W_W * tg.ARRAY_S // 4,
            "activation": tg.A_W * tg.ARRAY_S // 4,
            "golden": tg.R_W * tg.ARRAY_S // 4,
        }
        keys = {
            "command": "commands",
            "weight": "weight_rows",
            "activation": "activation_rows",
            "golden": "golden_rows",
        }
        for index, case in enumerate(cases):
            for suffix, width in widths.items():
                path = os.path.join(tempdir, "pattern", "t%02d_%s.dat" % (index, suffix))
                lines = nonempty_lines(path)
                assert len(lines) == len(case[keys[suffix]])
                assert all(len(line) == width for line in lines)
                assert all(int(line, 16) >= 0 for line in lines)


def check_project_files():
    cases = gs.build_suite()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, ".."))
    keys = {
        "command": lambda case: [command.word for command in case["commands"]],
        "weight": lambda case: case["weight_rows"],
        "activation": lambda case: case["activation_rows"],
        "golden": lambda case: case["golden_rows"],
    }
    for index, case in enumerate(cases):
        for suffix, expected_fn in keys.items():
            path = os.path.join(root, "pattern", "t%02d_%s.dat" % (index, suffix))
            actual = [int(line, 16) for line in nonempty_lines(path)]
            assert actual == expected_fn(case), "stale generated file: %s" % path

    with tempfile.TemporaryDirectory() as tempdir:
        expected_manifest = os.path.join(tempdir, "test_suite.svh")
        expected_summary = os.path.join(tempdir, "suite_summary.txt")
        gs.write_manifest(expected_manifest, cases)
        gs.write_summary(expected_summary, cases)
        with open(expected_manifest, "r") as expected_file:
            expected = expected_file.read()
        with open(os.path.join(root, "test_suite.svh"), "r") as actual_file:
            assert actual_file.read() == expected, "stale test_suite.svh"
        with open(expected_summary, "r") as expected_file:
            expected = expected_file.read()
        with open(os.path.join(root, "suite_summary.txt"), "r") as actual_file:
            assert actual_file.read() == expected, "stale suite_summary.txt"


CHECKS = [
    ("RTL/Python constants", check_rtl_constants),
    ("descriptor encode/decode", check_descriptor),
    ("Type-2 primitives", check_type2_primitives),
    ("hand calculations", check_hand_calculation),
    ("independent matrix oracle", check_independent_matrix),
    ("command schedules", check_command_schedules),
    ("suite contract", check_suite_contract),
    ("generated files", check_generated_files),
    ("project vectors", check_project_files),
]


def main():
    failures = 0
    print("TPU Python self-check")
    print("=" * 64)
    for name, check in CHECKS:
        try:
            check()
            print("[PASS] %s" % name)
        except Exception as error:
            failures += 1
            print("[FAIL] %s: %s" % (name, error))
    print("=" * 64)
    if failures:
        print("FAIL: %d Python check(s) failed" % failures)
        return 1
    print("PASS: all Python checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
