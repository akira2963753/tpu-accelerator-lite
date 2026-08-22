#!/usr/bin/env python3
"""Self-check the model, command compiler, suite, and generated files."""

import argparse
import os
import re
import sys
import tempfile

COMMON_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "python"
))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

import gen_suite as gs
import tpu_command as tc
import tpu_gen as tg


PROFILE = "regression"
CASES = []


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
    assert tc.CMD_DEFINED_W == 88
    assert tc.K_VALID_W == 6
    assert tc.ACT_MODE_W == 2


def check_descriptor():
    command = tc.make_command(
        tc.CMD_OP_GEMM,
        acc_init=True,
        acc_final=True,
        qmult=131071,
        qshift=63,
        mt=31,
        kt=255,
        nt=127,
        w_slot=255,
        src_slot=511,
        dst_slot=510,
        k_valid=17,
        bias_en=True,
    )
    decoded = tc.decode_descriptor(command.word)
    assert decoded == {
        "opcode": tc.CMD_OP_GEMM,
        "acc_init": 1,
        "acc_final": 1,
        "qmult": 131071,
        "qshift": 63,
        "mt": 31,
        "kt": 255,
        "nt": 127,
        "w_slot": 255,
        "src_slot": 511,
        "dst_slot": 510,
        "k_valid": 17,
        "bias_en": 1,
        "act_mode": tc.ACT_NONE,
        "a_mask_en": 0,
    }
    assert command.word >> tc.CMD_DEFINED_W == 0
    assert tc.field(command.word, 5, 3) == 0

    try:
        tc.make_command(tc.CMD_OP_STORE_C, qmult=-1)
    except AssertionError:
        pass
    else:
        raise AssertionError("negative QMULT must be rejected")

    store = tc.make_command(tc.CMD_OP_STORE_C, act_mode=tc.ACT_RELU)
    load_a = tc.make_command(tc.CMD_OP_LOAD_A, a_mask_en=True)
    load_bias = tc.make_command(tc.CMD_OP_LOAD_BIAS)
    assert tc.decode_descriptor(store.word)["act_mode"] == tc.ACT_RELU
    assert tc.decode_descriptor(load_a.word)["a_mask_en"] == 1
    assert tc.opcode_name(load_bias.word) == "LOAD_BIAS"


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

    activation_mask = [[int((m_idx + k_idx) % 3 == 0)
                        for k_idx in range(32)] for m_idx in range(32)]
    assert tg.unpack_activation_mask(
        tg.pack_activation_mask(activation_mask)
    ) == activation_mask

    bias = [((lane * 37) - 500) for lane in range(32)]
    assert tg.unpack_bias(tg.pack_bias(bias)) == bias


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
    assert tg.op_model(256, 0, 0) == 0
    assert tg.op_model(256, 1, 5) == 7
    assert tg.op_model(256, 1, 63) == 0
    assert tg.op_model(-256, 1, 6, tc.ACT_NONE) == 0xC
    assert tg.op_model(-4096, 1, 0, tc.ACT_NONE) == 0x8

    activation_64 = [[0 for _ in range(64)] for _ in range(1)]
    weight_64 = [[0] for _ in range(64)]
    assert tg.matmul_acc_model(activation_64, weight_64) == [[512]]


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

    tile_activation = [[0 for _ in range(32)] for _ in range(32)]
    tile_weight = [[0 for _ in range(32)] for _ in range(32)]
    tile_mask = [[int(k_idx % 3 == 0) for k_idx in range(32)]
                 for _ in range(32)]
    tile_acc = tg.matmul_tile_acc_model(
        tile_activation, tile_weight, 17, tile_mask
    )
    assert all(value == 48 for row in tile_acc for value in row)


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
    assert next(command for command in basic
                if command.opcode == tc.CMD_OP_GEMM).k_valid == 32
    assert next(command for command in basic
                if command.opcode == tc.CMD_OP_STORE_C).act_mode == tc.ACT_RELU

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

    k_limit = tc.compile_gemm_trace(32, tc.K_MAX, 32, 1, 0)
    assert max(command.w_slot for command in k_limit) == 255
    assert max(command.src_slot for command in k_limit) == 255


def check_reference_replay():
    case = next(case for case in CASES if case["name"] == "gemm_basic") \
        if PROFILE != "stress" else CASES[0]
    golden = tg.replay_commands(
        case["commands"], case["weight_rows"], case["activation_rows"]
    )
    assert golden == case["golden_rows"]


def check_suite_contract():
    expected_count = {"regression": 28, "stress": 4, "all": 32}[PROFILE]
    assert len(CASES) == expected_count
    assert len({case["name"] for case in CASES}) == len(CASES)

    for case in CASES:
        gs.validate_case(case)
        for row in case["golden_rows"]:
            assert all((lane & 0x0F) == 0 for lane in tg.unpack_row(row, tg.R_W))

    if PROFILE in ("regression", "all"):
        by_name = {case["name"]: case for case in CASES}
        assert by_name["ready_stall"]["r_stall"] == 1
        assert by_name["read_ub_stall"]["r_stall"] == 2
        assert by_name["input_valid_gap"]["w_stall"] == 1
        assert by_name["input_valid_gap"]["a_stall"] == 2

        switch = by_name["preload_switch_reuse"]["golden_rows"]
        assert switch[0:32] == switch[64:96]
        assert switch[0:32] != switch[32:64]

        restart = by_name["acc_init_restart"]["golden_rows"]
        assert restart[0:32] == restart[64:96]
        assert restart[0:32] != restart[32:64]

        relu_values = {
            lane for row in by_name["relu_mixed_sign"]["golden_rows"]
            for lane in tg.unpack_row(row, tg.R_W)
        }
        assert {0x00, 0x40, 0x70}.issubset(relu_values)

        k_tail = {
            lane for row in by_name["k_valid_tail"]["golden_rows"]
            for lane in tg.unpack_row(row, tg.R_W)
        }
        assert k_tail == {0x50}

        act_none = by_name["act_mode_none"]["golden_rows"]
        assert all(lane == 0xC0 for row in act_none[:32]
                   for lane in tg.unpack_row(row, tg.R_W))
        assert all(lane == 0x00 for row in act_none[32:]
                   for lane in tg.unpack_row(row, tg.R_W))

        bias = by_name["bias_init"]["golden_rows"]
        assert bias[0:32] != bias[32:64]

        amask = by_name["activation_mask"]["golden_rows"]
        expected_rows = [((1 + row % 7) << 4) for row in range(32)]
        assert [tg.unpack_row(row, tg.R_W)[0] for row in amask] == expected_rows


def check_coverage_contract():
    coverage = gs.collect_coverage(CASES)
    if PROFILE in ("regression", "all"):
        assert coverage["opcodes"] == set(tc.OPCODE_NAME)
        assert coverage["activation_codes"] == set(range(16))
        assert coverage["weight_codes"] == set(range(256))
        assert {0, 15, 16, 255}.issubset(coverage["w_slots"])
        assert {0, 15, 16, 511}.issubset(coverage["ub_slots"])
        assert coverage["acc_init"] == {0, 1}
        assert coverage["acc_final"] == {0, 1}
        assert {5, 17, 32}.issubset(coverage["k_valid"])
        assert coverage["bias_en"] == {0, 1}
        assert coverage["act_mode"] == {tc.ACT_NONE, tc.ACT_RELU}
        assert coverage["a_mask_en"] == {0, 1}
        assert {0, 1, 3, 131071}.issubset(coverage["qmult"])
        assert {0, 63}.issubset(coverage["qshift"])

    if PROFILE in ("stress", "all"):
        by_name = {case["name"]: case for case in CASES}
        assert by_name["m_limit"]["M"] == tc.M_MAX
        assert by_name["n_limit"]["N"] == tc.N_MAX
        assert by_name["k_limit"]["K"] == tc.K_MAX


def nonempty_lines(path):
    with open(path, "r") as data_file:
        return [line.strip() for line in data_file if line.strip()]


def check_generated_files():
    with tempfile.TemporaryDirectory() as tempdir:
        stale_path = os.path.join(tempdir, "t99_command.dat")
        with open(stale_path, "w") as stale_file:
            stale_file.write("deadbeef\n")

        gs.write_suite(tempdir, CASES)
        assert not os.path.exists(stale_path)
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
        for index, case in enumerate(CASES):
            for suffix, width in widths.items():
                path = os.path.join(tempdir, "t%02d_%s.dat" % (index, suffix))
                lines = nonempty_lines(path)
                expected_count = len(case[keys[suffix]])
                assert len(lines) == expected_count
                assert all(len(line) == width for line in lines)
                assert all(int(line, 16) >= 0 for line in lines)


def check_project_files():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", "pattern"))
    keys = {
        "command": lambda case: [command.word for command in case["commands"]],
        "weight": lambda case: case["weight_rows"],
        "activation": lambda case: case["activation_rows"],
        "golden": lambda case: case["golden_rows"],
    }
    for index, case in enumerate(CASES):
        for suffix, expected_fn in keys.items():
            path = os.path.join(root, "t%02d_%s.dat" % (index, suffix))
            actual = [int(line, 16) for line in nonempty_lines(path)]
            assert actual == expected_fn(case), "stale generated file: %s" % path

    with tempfile.TemporaryDirectory() as tempdir:
        expected_manifest = os.path.join(tempdir, "test_suite.svh")
        expected_summary = os.path.join(tempdir, "suite_summary.txt")
        gs.write_manifest(expected_manifest, CASES)
        gs.write_summary(expected_summary, CASES)
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
    ("reference trace replay", check_reference_replay),
    ("suite contract", check_suite_contract),
    ("coverage contract", check_coverage_contract),
    ("generated files", check_generated_files),
    ("project vectors", check_project_files),
]


def main():
    global PROFILE, CASES

    parser = argparse.ArgumentParser(description="Check generated TPU vectors")
    parser.add_argument("--profile", choices=("regression", "stress", "all"),
                        default="regression")
    args = parser.parse_args()
    PROFILE = args.profile
    CASES = gs.build_suite(PROFILE)

    failures = 0
    print("TPU Python self-check (%s)" % PROFILE)
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
