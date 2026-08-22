#!/usr/bin/env python3
"""Generate directed and stress verification suites."""

import argparse
import glob
import os
import random
import sys

COMMON_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "python"
))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

import tpu_command as tc
import tpu_gen as tg


def gemm_spec(name, m_dim, k_dim, n_dim, *, seed, activation_mode="random",
              weight_mode="msr", qmult=1, qshift=None, readback=False,
              r_stall=0, w_stall=0, a_stall=0):
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
        "w_stall": w_stall,
        "a_stall": a_stall,
    }


def custom_spec(name, builder, *, m_dim=32, k_dim=32, n_dim=32,
                r_stall=0, w_stall=0, a_stall=0):
    return {
        "kind": "custom",
        "name": name,
        "builder": builder,
        "M": m_dim,
        "K": k_dim,
        "N": n_dim,
        "r_stall": r_stall,
        "w_stall": w_stall,
        "a_stall": a_stall,
    }


class TraceBuilder:
    def __init__(self):
        self.commands = []
        self.weight_rows = []
        self.activation_rows = []

    def nop(self):
        self.commands.append(tc.make_command(tc.CMD_OP_NOP))

    def load_w(self, slot, tile, *, kt=0, nt=0):
        self.commands.append(tc.make_command(
            tc.CMD_OP_LOAD_W, w_slot=slot, kt=kt, nt=nt,
        ))
        self.weight_rows.extend(tg.pack_weight_tile(tile, 0, 0))

    def preload_w(self, slot, *, kt=0, nt=0):
        self.commands.append(tc.make_command(
            tc.CMD_OP_PRELOAD_W, w_slot=slot, kt=kt, nt=nt,
        ))

    def load_a(self, slot, tile, *, mt=0, kt=0, activation_mask=None):
        self.commands.append(tc.make_command(
            tc.CMD_OP_LOAD_A,
            src_slot=slot,
            mt=mt,
            kt=kt,
            a_mask_en=(activation_mask is not None),
        ))
        self.activation_rows.extend(tg.pack_activation_tile(tile, 0, 0))
        if activation_mask is not None:
            self.activation_rows.extend(tg.pack_activation_mask(activation_mask))

    def load_bias(self, bias):
        self.commands.append(tc.make_command(tc.CMD_OP_LOAD_BIAS))
        self.weight_rows.extend(tg.pack_bias(bias))

    def gemm(self, src_slot, *, acc_init, acc_final=True, mt=0, kt=0, nt=0,
             k_valid=tg.ARRAY_S, bias_en=False):
        self.commands.append(tc.make_command(
            tc.CMD_OP_GEMM,
            acc_init=acc_init,
            acc_final=acc_final,
            src_slot=src_slot,
            mt=mt,
            kt=kt,
            nt=nt,
            k_valid=k_valid,
            bias_en=bias_en,
        ))

    def store(self, dst_slot, *, qmult=1, qshift=0, mt=0, nt=0,
              act_mode=tc.ACT_RELU):
        self.commands.append(tc.make_command(
            tc.CMD_OP_STORE_C,
            qmult=qmult,
            qshift=qshift,
            dst_slot=dst_slot,
            mt=mt,
            nt=nt,
            act_mode=act_mode,
        ))

    def read_ub(self, src_slot, *, mt=0, nt=0):
        self.commands.append(tc.make_command(
            tc.CMD_OP_READ_UB, src_slot=src_slot, mt=mt, nt=nt,
        ))


def activation_tile(tag=0):
    return [
        [((row * 3 + lane * 5 + tag) & 0xF) for lane in range(tg.ARRAY_S)]
        for row in range(tg.ARRAY_S)
    ]


def weight_tile(tag=0):
    values = [0x00, 0x01, 0x07, 0x0F, 0x10, 0x31, 0x40, 0x7F,
              0x80, 0xA4, 0xEF, 0xF0, 0xF1, 0xF8, 0xFE, 0xFF]
    return [
        [values[(row * 7 + lane * 3 + tag) % len(values)]
         for lane in range(tg.ARRAY_S)]
        for row in range(tg.ARRAY_S)
    ]


def tile_qshift(activation, weight, qmult=1):
    return tg.auto_qshift(tg.matmul_acc_model(activation, weight), qmult)


def finish_custom(spec, trace):
    golden_rows = tg.replay_commands(
        trace.commands, trace.weight_rows, trace.activation_rows
    )
    result = dict(spec)
    result.update({
        "qmult": 1,
        "qshift": 0,
        "commands": trace.commands,
        "weight_rows": trace.weight_rows,
        "activation_rows": trace.activation_rows,
        "golden_rows": golden_rows,
        "positive_acc": 0,
        "zero_output": sum(
            lane == 0
            for row in golden_rows
            for lane in tg.unpack_row(row, tg.R_W)
        ),
        "sat_output": sum(
            lane == (tg.SAT_MAX << (tg.R_W - tg.A_W))
            for row in golden_rows
            for lane in tg.unpack_row(row, tg.R_W)
        ),
    })
    return result


def build_nop(spec):
    trace = TraceBuilder()
    trace.nop()
    return finish_custom(spec, trace)


def build_ub_load_read(spec):
    trace = TraceBuilder()
    trace.load_a(7, activation_tile(1))
    trace.read_ub(7)
    return finish_custom(spec, trace)


def build_ub_bank_boundary(spec):
    trace = TraceBuilder()
    slots = [0, 15, 16, 511]
    for tag, slot in enumerate(slots):
        trace.load_a(slot, activation_tile(tag))
    for slot in slots:
        trace.read_ub(slot)
    return finish_custom(spec, trace)


def build_ub_overwrite(spec):
    trace = TraceBuilder()
    trace.load_a(15, activation_tile(2))
    trace.load_a(16, activation_tile(5))
    trace.load_a(16, activation_tile(11))
    trace.read_ub(16)
    trace.read_ub(15)
    return finish_custom(spec, trace)


def build_wmem_bank_boundary(spec):
    trace = TraceBuilder()
    activation = activation_tile(4)
    trace.load_a(42, activation)
    slots = [0, 15, 16, 255]
    for tag, slot in enumerate(slots):
        trace.load_w(slot, weight_tile(tag))
    for tag, slot in enumerate(slots):
        weight = weight_tile(tag)
        trace.preload_w(slot)
        trace.gemm(42, acc_init=True)
        trace.store(100 + tag, qshift=tile_qshift(activation, weight))
    return finish_custom(spec, trace)


def build_preload_switch(spec):
    trace = TraceBuilder()
    activation = activation_tile(3)
    weights = [weight_tile(1), weight_tile(8)]
    trace.load_a(2, activation)
    trace.load_w(15, weights[0])
    trace.load_w(16, weights[1])
    for index, select in enumerate((0, 1, 0)):
        trace.preload_w((15, 16)[select])
        trace.gemm(2, acc_init=True)
        trace.store(20 + index, qshift=tile_qshift(activation, weights[select]))
    return finish_custom(spec, trace)


def build_activation_reuse(spec):
    trace = TraceBuilder()
    activation = activation_tile(9)
    weights = [weight_tile(2), weight_tile(6)]
    trace.load_a(511, activation)
    for index, weight in enumerate(weights):
        trace.load_w(30 + index, weight)
        trace.preload_w(30 + index)
        trace.gemm(511, acc_init=True)
        trace.store(40 + index, qshift=tile_qshift(activation, weight))
    return finish_custom(spec, trace)


def build_dst_slot_isolation(spec):
    trace = TraceBuilder()
    weight = weight_tile(4)
    trace.load_w(7, weight)
    trace.preload_w(7)
    dst_slots = [0, 15, 16, 511]
    for tag, dst_slot in enumerate(dst_slots):
        activation = activation_tile(tag * 2)
        trace.load_a(60 + tag, activation)
        trace.gemm(60 + tag, acc_init=True)
        trace.store(dst_slot, qshift=tile_qshift(activation, weight))
    for dst_slot in reversed(dst_slots):
        trace.read_ub(dst_slot)
    return finish_custom(spec, trace)


def build_activation_sweep(spec):
    trace = TraceBuilder()
    activation = [
        [((row * tg.ARRAY_S + lane) & 0xF) for lane in range(tg.ARRAY_S)]
        for row in range(tg.ARRAY_S)
    ]
    weight = weight_tile(5)
    trace.load_w(8, weight)
    trace.load_a(8, activation)
    trace.preload_w(8)
    trace.gemm(8, acc_init=True)
    trace.store(8, qshift=tile_qshift(activation, weight))
    return finish_custom(spec, trace)


def build_weight_sweep(spec):
    trace = TraceBuilder()
    activation = activation_tile(7)
    weight = [
        [((row * tg.ARRAY_S + lane) & 0xFF) for lane in range(tg.ARRAY_S)]
        for row in range(tg.ARRAY_S)
    ]
    trace.load_w(9, weight)
    trace.load_a(9, activation)
    trace.preload_w(9)
    trace.gemm(9, acc_init=True)
    trace.store(9, qshift=tile_qshift(activation, weight))
    return finish_custom(spec, trace)


def build_relu_mixed_sign(spec):
    trace = TraceBuilder()
    activation = [[0 for _ in range(tg.ARRAY_S)] for _ in range(tg.ARRAY_S)]
    weight = []
    for row in range(tg.ARRAY_S):
        values = []
        for lane in range(tg.ARRAY_S):
            mode = lane % 4
            if mode == 0:
                values.append(0x00)
            elif mode == 1:
                values.append(0xFF)
            elif mode == 2:
                values.append(0x00 if row < 16 else 0xFF)
            else:
                values.append(0x40)
        weight.append(values)
    trace.load_w(10, weight)
    trace.load_a(10, activation)
    trace.preload_w(10)
    trace.gemm(10, acc_init=True)
    trace.store(10, qshift=6)
    return finish_custom(spec, trace)


def build_requant_boundary(spec):
    trace = TraceBuilder()
    activation = [[0 for _ in range(tg.ARRAY_S)] for _ in range(tg.ARRAY_S)]
    weight = [[0 for _ in range(tg.ARRAY_S)] for _ in range(tg.ARRAY_S)]
    trace.load_w(11, weight)
    trace.load_a(11, activation)
    trace.preload_w(11)
    trace.gemm(11, acc_init=True)
    settings = [
        (0, 0),
        (1, 6),
        (1, 5),
        (3, 7),
        (131071, 22),
        (131071, 21),
        (1, 63),
    ]
    for index, (qmult, qshift) in enumerate(settings):
        trace.store(70 + index, qmult=qmult, qshift=qshift)
    return finish_custom(spec, trace)


def build_acc_init_restart(spec):
    trace = TraceBuilder()
    activation = [[0 for _ in range(tg.ARRAY_S)] for _ in range(tg.ARRAY_S)]
    weight = [[0 for _ in range(tg.ARRAY_S)] for _ in range(tg.ARRAY_S)]
    trace.load_w(12, weight)
    trace.load_a(12, activation)
    trace.preload_w(12)
    trace.gemm(12, acc_init=True, acc_final=False)
    trace.store(80, qshift=6)
    trace.gemm(12, acc_init=False)
    trace.store(81, qshift=6)
    trace.gemm(12, acc_init=True)
    trace.store(82, qshift=6)
    return finish_custom(spec, trace)


def build_read_ub_stall(spec):
    trace = TraceBuilder()
    trace.load_a(511, activation_tile(13))
    trace.read_ub(511)
    return finish_custom(spec, trace)


def build_k_valid_tail(spec):
    trace = TraceBuilder()
    activation = [[0 for _ in range(tg.ARRAY_S)] for _ in range(tg.ARRAY_S)]
    weight = [[0 for _ in range(tg.ARRAY_S)] for _ in range(tg.ARRAY_S)]
    trace.load_w(13, weight)
    trace.load_a(13, activation)
    trace.preload_w(13)
    trace.gemm(13, acc_init=True, k_valid=5)
    trace.store(13, qshift=3)
    return finish_custom(spec, trace)


def build_act_mode_none(spec):
    trace = TraceBuilder()
    activation = [[0 for _ in range(tg.ARRAY_S)] for _ in range(tg.ARRAY_S)]
    weight = [[0xFF for _ in range(tg.ARRAY_S)] for _ in range(tg.ARRAY_S)]
    trace.load_w(14, weight)
    trace.load_a(14, activation)
    trace.preload_w(14)
    trace.gemm(14, acc_init=True)
    trace.store(14, qshift=6, act_mode=tc.ACT_NONE)
    trace.store(15, qshift=6, act_mode=tc.ACT_RELU)
    return finish_custom(spec, trace)


def build_bias_init(spec):
    trace = TraceBuilder()
    activation = [[0 for _ in range(tg.ARRAY_S)] for _ in range(tg.ARRAY_S)]
    weight = [[0 for _ in range(tg.ARRAY_S)] for _ in range(tg.ARRAY_S)]
    bias = [64 * ((lane % 16) - 8) for lane in range(tg.ARRAY_S)]
    trace.load_bias(bias)
    trace.load_w(15, weight)
    trace.load_a(15, activation)
    trace.preload_w(15)
    trace.gemm(15, acc_init=True, bias_en=True)
    trace.store(16, qshift=6, act_mode=tc.ACT_NONE)
    trace.gemm(15, acc_init=True)
    trace.store(17, qshift=6, act_mode=tc.ACT_NONE)
    return finish_custom(spec, trace)


def build_activation_mask(spec):
    trace = TraceBuilder()
    activation = [[0 for _ in range(tg.ARRAY_S)] for _ in range(tg.ARRAY_S)]
    weight = [[0 for _ in range(tg.ARRAY_S)] for _ in range(tg.ARRAY_S)]
    activation_mask = []
    for m_idx in range(tg.ARRAY_S):
        count = 1 + (m_idx % 7)
        activation_mask.append([
            int(((k_idx * 5 + m_idx) % 17) < count)
            if k_idx < 17 else int(k_idx % 2 == 0)
            for k_idx in range(tg.ARRAY_S)
        ])

    trace.load_w(16, weight)
    trace.load_a(16, activation, activation_mask=activation_mask)
    trace.preload_w(16)
    trace.gemm(16, acc_init=True, k_valid=17)
    trace.store(18, qshift=3)
    return finish_custom(spec, trace)


REGRESSION_TABLE = [
    custom_spec("ub_load_read", build_ub_load_read),
    gemm_spec("gemm_basic", 32, 32, 32, seed=20),
    gemm_spec("gemm_signed", 32, 32, 32, seed=21,
              activation_mode="ramp", weight_mode="mixed"),
    gemm_spec("gemm_multi_k", 32, 64, 32, seed=22),
    gemm_spec("weight_reuse", 64, 32, 32, seed=23),
    gemm_spec("multi_tile_slot", 64, 64, 64, seed=24,
              activation_mode="ramp", weight_mode="mixed"),
    gemm_spec("requant_relu", 32, 32, 32, seed=25,
              activation_mode="ramp", weight_mode="mixed", qmult=3),
    gemm_spec("store_read_ub", 32, 32, 32, seed=26, readback=True),
    gemm_spec("ready_stall", 32, 32, 32, seed=27, r_stall=1),
    custom_spec("nop_command", build_nop),
    custom_spec("ub_bank_boundary", build_ub_bank_boundary),
    custom_spec("ub_overwrite", build_ub_overwrite),
    custom_spec("wmem_bank_boundary", build_wmem_bank_boundary),
    custom_spec("preload_switch_reuse", build_preload_switch),
    custom_spec("activation_reuse", build_activation_reuse),
    custom_spec("dst_slot_isolation", build_dst_slot_isolation),
    custom_spec("activation_code_sweep", build_activation_sweep),
    custom_spec("weight_code_sweep", build_weight_sweep),
    custom_spec("relu_mixed_sign", build_relu_mixed_sign),
    custom_spec("requant_boundary", build_requant_boundary),
    custom_spec("acc_init_restart", build_acc_init_restart),
    custom_spec("read_ub_stall", build_read_ub_stall, r_stall=2),
    custom_spec("k_valid_tail", build_k_valid_tail, k_dim=5),
    custom_spec("act_mode_none", build_act_mode_none),
    custom_spec("bias_init", build_bias_init),
    custom_spec("activation_mask", build_activation_mask, k_dim=17),
    gemm_spec("input_valid_gap", 32, 32, 32, seed=28,
              activation_mode="ramp", weight_mode="mixed", w_stall=1, a_stall=2),
    gemm_spec("rectangular_tiles", 96, 64, 96, seed=29,
              activation_mode="ramp", weight_mode="mixed",
              r_stall=1, w_stall=1, a_stall=1),
]


STRESS_TABLE = [
    gemm_spec("m_limit", 1024, 32, 32, seed=101),
    gemm_spec("n_limit", 32, 32, 4096, seed=102),
    gemm_spec("k_limit", 32, 8192, 32, seed=103),
    gemm_spec("large_mixed", 128, 128, 128, seed=104,
              activation_mode="ramp", weight_mode="mixed", r_stall=1),
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

    commands = tc.compile_gemm_trace(
        spec["M"], spec["K"], spec["N"],
        spec["qmult"], qshift,
        readback=spec["readback"],
    )
    weight_rows = []
    activation_rows = []
    for command in commands:
        if command.opcode == tc.CMD_OP_LOAD_W:
            weight_rows.extend(tg.pack_weight_tile(weight, command.kt, command.nt))
        elif command.opcode == tc.CMD_OP_LOAD_A:
            activation_rows.extend(tg.pack_activation_tile(
                activation, command.mt, command.kt
            ))

    golden_rows = tg.replay_commands(commands, weight_rows, activation_rows)
    output = [
        [tg.op_model(value, spec["qmult"], qshift) for value in row]
        for row in accumulator
    ]
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


def build_suite(profile="regression"):
    assert profile in ("regression", "stress", "all")
    specs = []
    if profile in ("regression", "all"):
        specs.extend(REGRESSION_TABLE)
    if profile in ("stress", "all"):
        specs.extend(STRESS_TABLE)

    cases = []
    for spec in specs:
        case = spec["builder"](spec) if spec["kind"] == "custom" else _gemm_case(spec)
        validate_case(case)
        cases.append(case)
    return cases


def validate_case(case):
    weight_beats = sum(
        tg.ARRAY_S if command.opcode == tc.CMD_OP_LOAD_W else
        tg.BIAS_BEATS if command.opcode == tc.CMD_OP_LOAD_BIAS else 0
        for command in case["commands"]
    )
    activation_beats = sum(
        tg.ARRAY_S + (tg.AMASK_BEATS if command.a_mask_en else 0)
        if command.opcode == tc.CMD_OP_LOAD_A else 0
        for command in case["commands"]
    )
    result_cmds = sum(
        command.opcode in (tc.CMD_OP_STORE_C, tc.CMD_OP_READ_UB)
        for command in case["commands"]
    )
    assert len(case["weight_rows"]) == weight_beats
    assert len(case["activation_rows"]) == activation_beats
    assert len(case["golden_rows"]) == result_cmds * tg.ARRAY_S
    assert case["r_stall"] in (0, 1, 2)
    assert case["w_stall"] in (0, 1, 2)
    assert case["a_stall"] in (0, 1, 2)

    for command in case["commands"]:
        decoded = tc.decode_descriptor(command.word)
        assert decoded["opcode"] == command.opcode
        assert decoded["mt"] == command.mt
        assert decoded["kt"] == command.kt
        assert decoded["nt"] == command.nt
        assert decoded["w_slot"] == command.w_slot
        assert decoded["src_slot"] == command.src_slot
        assert decoded["dst_slot"] == command.dst_slot
        assert decoded["k_valid"] == command.k_valid
        assert decoded["bias_en"] == command.bias_en
        assert decoded["act_mode"] == command.act_mode
        assert decoded["a_mask_en"] == command.a_mask_en
        assert decoded["qmult"] >= 0
        assert command.word >> tc.CMD_DEFINED_W == 0
        assert tc.field(command.word, 5, 3) == 0


def collect_coverage(cases):
    coverage = {
        "opcodes": set(),
        "activation_codes": set(),
        "weight_codes": set(),
        "w_slots": set(),
        "ub_slots": set(),
        "qmult": set(),
        "qshift": set(),
        "acc_init": set(),
        "acc_final": set(),
        "k_valid": set(),
        "bias_en": set(),
        "act_mode": set(),
        "a_mask_en": set(),
    }
    for case in cases:
        for command in case["commands"]:
            fields = tc.decode_descriptor(command.word)
            coverage["opcodes"].add(fields["opcode"])
            coverage["qmult"].add(fields["qmult"])
            coverage["qshift"].add(fields["qshift"])
            if fields["opcode"] == tc.CMD_OP_GEMM:
                coverage["acc_init"].add(fields["acc_init"])
                coverage["acc_final"].add(fields["acc_final"])
                coverage["k_valid"].add(fields["k_valid"])
                coverage["bias_en"].add(fields["bias_en"])
            if fields["opcode"] == tc.CMD_OP_STORE_C:
                coverage["act_mode"].add(fields["act_mode"])
            if fields["opcode"] == tc.CMD_OP_LOAD_A:
                coverage["a_mask_en"].add(fields["a_mask_en"])
            if fields["opcode"] in (tc.CMD_OP_LOAD_W, tc.CMD_OP_PRELOAD_W):
                coverage["w_slots"].add(fields["w_slot"])
            if fields["opcode"] in (tc.CMD_OP_LOAD_A, tc.CMD_OP_GEMM,
                                    tc.CMD_OP_READ_UB):
                coverage["ub_slots"].add(fields["src_slot"])
            if fields["opcode"] == tc.CMD_OP_STORE_C:
                coverage["ub_slots"].add(fields["dst_slot"])
        for row in case["activation_rows"]:
            coverage["activation_codes"].update(tg.unpack_row(row, tg.A_W))
        for row in case["weight_rows"]:
            coverage["weight_codes"].update(tg.unpack_row(row, tg.W_W))
    return coverage


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
        manifest.write("// Auto-generated by directed/gen_suite.py.\n\n")
        manifest.write("    localparam int NUM_TESTS     = %d;\n" % len(cases))
        manifest.write("    localparam int MAX_CMD_COUNT = %d;\n" % max_cmd)
        manifest.write("    localparam int MAX_W_ROWS    = %d;\n" % max_w)
        manifest.write("    localparam int MAX_A_ROWS    = %d;\n" % max_a)
        manifest.write("    localparam int MAX_R_ROWS    = %d;\n\n" % max_r)
        manifest.write(_sv_int_array("TEST_M", [case["M"] for case in cases]))
        manifest.write(_sv_int_array("TEST_K", [case["K"] for case in cases]))
        manifest.write(_sv_int_array("TEST_N", [case["N"] for case in cases]))
        manifest.write(_sv_int_array("TEST_CMD_COUNT", [len(case["commands"]) for case in cases]))
        manifest.write(_sv_int_array("TEST_W_ROWS", [len(case["weight_rows"]) for case in cases]))
        manifest.write(_sv_int_array("TEST_A_ROWS", [len(case["activation_rows"]) for case in cases]))
        manifest.write(_sv_int_array("TEST_R_ROWS", [len(case["golden_rows"]) for case in cases]))
        manifest.write(_sv_int_array("TEST_RSTALL", [case["r_stall"] for case in cases]))
        manifest.write(_sv_int_array("TEST_WSTALL", [case["w_stall"] for case in cases]))
        manifest.write(_sv_int_array("TEST_ASTALL", [case["a_stall"] for case in cases]))
        names = ", ".join('"%s"' % case["name"] for case in cases)
        manifest.write("    localparam string TEST_NAME [NUM_TESTS] = '{%s};\n" % names)


def write_summary(path, cases):
    coverage = collect_coverage(cases)
    with open(path, "w", newline="\n") as summary:
        summary.write("TPU verification suite\n")
        summary.write("=" * 104 + "\n")
        summary.write(
            "%-2s %-22s %5s %5s %5s %6s %6s %6s %6s %6s %6s\n" %
            ("id", "name", "M", "K", "N", "qmult", "qshift",
             "cmds", "wrows", "arows", "rrows")
        )
        for index, case in enumerate(cases):
            summary.write(
                "%-2d %-22s %5d %5d %5d %6d %6d %6d %6d %6d %6d\n" %
                (index, case["name"], case["M"], case["K"], case["N"],
                 case["qmult"], case["qshift"], len(case["commands"]),
                 len(case["weight_rows"]), len(case["activation_rows"]),
                 len(case["golden_rows"]))
            )
        summary.write("=" * 104 + "\n")
        summary.write("opcodes          : %s\n" % sorted(coverage["opcodes"]))
        summary.write("activation codes : %d / 16\n" % len(coverage["activation_codes"]))
        summary.write("weight codes     : %d / 256\n" % len(coverage["weight_codes"]))
        summary.write("WMEM slots       : %s\n" % sorted(coverage["w_slots"]))
        summary.write("UB slots         : %s\n" % sorted(coverage["ub_slots"]))
        summary.write("QMULT values     : %s\n" % sorted(coverage["qmult"]))
        summary.write("QSHIFT values    : %s\n" % sorted(coverage["qshift"]))
        summary.write("K_VALID values   : %s\n" % sorted(coverage["k_valid"]))
        summary.write("BIAS_EN values   : %s\n" % sorted(coverage["bias_en"]))
        summary.write("ACT_MODE values  : %s\n" % sorted(coverage["act_mode"]))
        summary.write("A_MASK_EN values : %s\n" % sorted(coverage["a_mask_en"]))


def write_suite(root, cases):
    os.makedirs(root, exist_ok=True)
    for stale_path in glob.glob(os.path.join(root, "t[0-9][0-9]_*.dat")):
        os.remove(stale_path)
    model_manifest = os.path.join(root, "manifest.json")
    if os.path.isfile(model_manifest):
        os.remove(model_manifest)

    for index, case in enumerate(cases):
        prefix = os.path.join(root, "t%02d_" % index)
        tg.write_dat(prefix + "command.dat",
                     [command.word for command in case["commands"]], tc.CMD_DESC_W)
        tg.write_dat(prefix + "weight.dat", case["weight_rows"], tg.W_W * tg.ARRAY_S)
        tg.write_dat(prefix + "activation.dat", case["activation_rows"],
                     tg.A_W * tg.ARRAY_S)
        tg.write_dat(prefix + "golden.dat", case["golden_rows"], tg.R_W * tg.ARRAY_S)

    write_manifest(os.path.join(root, "test_suite.svh"), cases)
    write_summary(os.path.join(root, "suite_summary.txt"), cases)


def main():
    parser = argparse.ArgumentParser(description="Generate TPU verification cases")
    parser.add_argument("--profile", choices=("regression", "stress", "all"),
                        default="regression")
    parser.add_argument("--outdir", default=None, help="active pattern directory")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = args.outdir or os.path.normpath(os.path.join(here, "..", "pattern"))
    cases = build_suite(args.profile)
    write_suite(root, cases)

    for index, case in enumerate(cases):
        print(
            "[%02d] %-22s M=%-4d K=%-5d N=%-4d cmds=%-4d rows=%d/%d/%d" %
            (index, case["name"], case["M"], case["K"], case["N"],
             len(case["commands"]), len(case["weight_rows"]),
             len(case["activation_rows"]), len(case["golden_rows"]))
        )
    print("generated %d %s cases -> %s" % (len(cases), args.profile, root))


if __name__ == "__main__":
    main()
