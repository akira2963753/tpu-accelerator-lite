#!/usr/bin/env python3
"""Generate the primary directed verification suite."""

import argparse
import os
import random

import tpu_command as tc
import tpu_gen as tg


def gemm_spec(name, m_dim, k_dim, n_dim, *, seed, activation_mode="random",
              weight_mode="msr", qmult=1, qshift=None, readback=False,
              r_stall=False):
    return {
        "kind": "gemm",
        "name": name,
        "M": m_dim,
        "K": k_dim,
        "N": n_dim,
        "seed": seed,
        "activation_mode": activation_mode,
        "weight_mode": weight_mode,
        "qmult": qmult,
        "qshift": qshift,
        "readback": readback,
        "r_stall": r_stall,
    }


TEST_TABLE = [
    {
        "kind": "ub",
        "name": "ub_load_read",
        "M": 32,
        "K": 32,
        "N": 32,
        "seed": 10,
        "r_stall": False,
    },
    gemm_spec("gemm_basic", 32, 32, 32, seed=20),
    gemm_spec(
        "gemm_signed", 32, 32, 32, seed=21,
        activation_mode="ramp", weight_mode="mixed",
    ),
    gemm_spec("gemm_multi_k", 32, 64, 32, seed=22),
    gemm_spec("weight_reuse", 64, 32, 32, seed=23),
    gemm_spec(
        "multi_tile_slot", 64, 64, 64, seed=24,
        activation_mode="ramp", weight_mode="mixed",
    ),
    gemm_spec(
        "requant_relu", 32, 32, 32, seed=25,
        activation_mode="ramp", weight_mode="mixed", qmult=3,
    ),
    gemm_spec("store_read_ub", 32, 32, 32, seed=26, readback=True),
    gemm_spec("ready_stall", 32, 32, 32, seed=27, r_stall=True),
]


def _gemm_case(spec):
    rng = random.Random(spec["seed"])
    activation = tg.generate_activation(
        rng, spec["M"], spec["K"], spec["activation_mode"]
    )
    weight = tg.generate_weight(
        rng, spec["K"], spec["N"], spec["weight_mode"]
    )
    accumulator = tg.matmul_acc_model(activation, weight)
    qshift = spec["qshift"]
    if qshift is None:
        qshift = tg.auto_qshift(accumulator, spec["qmult"])
    output = [
        [tg.op_model(value, spec["qmult"], qshift) for value in row]
        for row in accumulator
    ]

    commands = tc.compile_gemm_trace(
        spec["M"], spec["K"], spec["N"],
        spec["qmult"], qshift,
        readback=spec["readback"],
    )
    weight_rows = []
    activation_rows = []
    golden_rows = []
    for command in commands:
        if command.opcode == tc.CMD_OP_LOAD_W:
            weight_rows.extend(tg.pack_weight_tile(
                weight, command.kt, command.nt
            ))
        elif command.opcode == tc.CMD_OP_LOAD_A:
            activation_rows.extend(tg.pack_activation_tile(
                activation, command.mt, command.kt
            ))
        elif command.opcode in (tc.CMD_OP_STORE_C, tc.CMD_OP_READ_UB):
            golden_rows.extend(tg.pack_output_tile(
                output, command.mt, command.nt
            ))

    result = dict(spec)
    result.update({
        "qshift": qshift,
        "commands": commands,
        "weight_rows": weight_rows,
        "activation_rows": activation_rows,
        "golden_rows": golden_rows,
        "positive_acc": sum(value > 0 for row in accumulator for value in row),
        "zero_output": sum(value == 0 for row in output for value in row),
        "sat_output": sum(value == tg.SAT_MAX for row in output for value in row),
    })
    return result


def _ub_case(spec):
    rng = random.Random(spec["seed"])
    activation = tg.generate_activation(rng, 32, 32, "ramp")
    activation_rows = tg.pack_activation_tile(activation, 0, 0)
    commands = tc.compile_ub_read_trace(slot=7)
    result = dict(spec)
    result.update({
        "qmult": 1,
        "qshift": 0,
        "commands": commands,
        "weight_rows": [],
        "activation_rows": activation_rows,
        "golden_rows": tg.expand_ub_rows(activation_rows),
        "positive_acc": 0,
        "zero_output": 0,
        "sat_output": 0,
    })
    return result


def build_suite():
    cases = []
    for spec in TEST_TABLE:
        case = _ub_case(spec) if spec["kind"] == "ub" else _gemm_case(spec)
        validate_case(case)
        cases.append(case)
    return cases


def validate_case(case):
    load_w = sum(command.opcode == tc.CMD_OP_LOAD_W for command in case["commands"])
    load_a = sum(command.opcode == tc.CMD_OP_LOAD_A for command in case["commands"])
    result_cmds = sum(
        command.opcode in (tc.CMD_OP_STORE_C, tc.CMD_OP_READ_UB)
        for command in case["commands"]
    )
    assert len(case["weight_rows"]) == load_w * tg.ARRAY_S
    assert len(case["activation_rows"]) == load_a * tg.ARRAY_S
    assert len(case["golden_rows"]) == result_cmds * tg.ARRAY_S

    for command in case["commands"]:
        decoded = tc.decode_descriptor(command.word)
        assert decoded["opcode"] == command.opcode
        assert decoded["mt"] == command.mt
        assert decoded["kt"] == command.kt
        assert decoded["nt"] == command.nt
        assert decoded["w_slot"] == command.w_slot
        assert decoded["src_slot"] == command.src_slot
        assert decoded["dst_slot"] == command.dst_slot
        assert command.word >> tc.CMD_DEFINED_W == 0
        assert tc.field(command.word, 5, 3) == 0


def _sv_int_array(name, values):
    return "    localparam int %-15s [NUM_TESTS] = '{%s};\n" % (
        name, ", ".join(str(value) for value in values)
    )


def write_manifest(path, cases):
    max_cmd = max(len(case["commands"]) for case in cases)
    max_w = max(len(case["weight_rows"]) for case in cases)
    max_a = max(len(case["activation_rows"]) for case in cases)
    max_r = max(len(case["golden_rows"]) for case in cases)
    with open(path, "w", newline="\n") as manifest:
        manifest.write("// Auto-generated by python/gen_suite.py.\n\n")
        manifest.write("    localparam int NUM_TESTS     = %d;\n" % len(cases))
        manifest.write("    localparam int MAX_CMD_COUNT = %d;\n" % max_cmd)
        manifest.write("    localparam int MAX_W_ROWS    = %d;\n" % max_w)
        manifest.write("    localparam int MAX_A_ROWS    = %d;\n" % max_a)
        manifest.write("    localparam int MAX_R_ROWS    = %d;\n\n" % max_r)
        manifest.write(_sv_int_array("TEST_M", [case["M"] for case in cases]))
        manifest.write(_sv_int_array("TEST_K", [case["K"] for case in cases]))
        manifest.write(_sv_int_array("TEST_N", [case["N"] for case in cases]))
        manifest.write(_sv_int_array(
            "TEST_CMD_COUNT", [len(case["commands"]) for case in cases]
        ))
        manifest.write(_sv_int_array(
            "TEST_W_ROWS", [len(case["weight_rows"]) for case in cases]
        ))
        manifest.write(_sv_int_array(
            "TEST_A_ROWS", [len(case["activation_rows"]) for case in cases]
        ))
        manifest.write(_sv_int_array(
            "TEST_R_ROWS", [len(case["golden_rows"]) for case in cases]
        ))
        manifest.write(_sv_int_array(
            "TEST_RSTALL", [int(case["r_stall"]) for case in cases]
        ))
        names = ", ".join('"%s"' % case["name"] for case in cases)
        manifest.write(
            "    localparam string TEST_NAME [NUM_TESTS] = '{%s};\n" % names
        )


def write_summary(path, cases):
    with open(path, "w", newline="\n") as summary:
        summary.write("TPU primary verification suite\n")
        summary.write("=" * 96 + "\n")
        summary.write(
            "%-2s %-18s %5s %5s %5s %6s %6s %6s %6s %6s %6s\n" %
            ("id", "name", "M", "K", "N", "qmult", "qshift",
             "cmds", "wrows", "arows", "rrows")
        )
        for index, case in enumerate(cases):
            summary.write(
                "%-2d %-18s %5d %5d %5d %6d %6d %6d %6d %6d %6d\n" %
                (index, case["name"], case["M"], case["K"], case["N"],
                 case["qmult"], case["qshift"], len(case["commands"]),
                 len(case["weight_rows"]), len(case["activation_rows"]),
                 len(case["golden_rows"]))
            )
        summary.write("=" * 96 + "\n")


def write_suite(root, cases):
    pattern_dir = os.path.join(root, "pattern")
    os.makedirs(pattern_dir, exist_ok=True)
    for index, case in enumerate(cases):
        prefix = os.path.join(pattern_dir, "t%02d_" % index)
        tg.write_dat(
            prefix + "command.dat",
            [command.word for command in case["commands"]],
            tc.CMD_DESC_W,
        )
        tg.write_dat(prefix + "weight.dat", case["weight_rows"], tg.W_W * tg.ARRAY_S)
        tg.write_dat(prefix + "activation.dat", case["activation_rows"], tg.A_W * tg.ARRAY_S)
        tg.write_dat(prefix + "golden.dat", case["golden_rows"], tg.R_W * tg.ARRAY_S)

    write_manifest(os.path.join(root, "test_suite.svh"), cases)
    write_summary(os.path.join(root, "suite_summary.txt"), cases)


def main():
    parser = argparse.ArgumentParser(description="Generate primary TPU directed cases")
    parser.add_argument("--outdir", default=None, help="00_TESTBED output directory")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = args.outdir or os.path.normpath(os.path.join(here, ".."))
    cases = build_suite()
    write_suite(root, cases)

    for index, case in enumerate(cases):
        print(
            "[%02d] %-18s M=%-3d K=%-3d N=%-3d cmds=%-3d rows=%d/%d/%d" %
            (index, case["name"], case["M"], case["K"], case["N"],
             len(case["commands"]), len(case["weight_rows"]),
             len(case["activation_rows"]), len(case["golden_rows"]))
        )
    print("generated %d cases -> %s" % (len(cases), root))


if __name__ == "__main__":
    main()
