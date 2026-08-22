#!/usr/bin/env python3
"""Compile serial TPU commands for the 32x32 architecture."""

from collections import namedtuple


ARRAY_S = 32
M_MAX = 1024
N_MAX = 4096
K_MAX = 8192

WMEM_SLOTS = 256
UB_SLOTS = 512

CMD_DESC_W = 512
CMD_DEFINED_W = 88

CMD_OP_NOP = 0
CMD_OP_LOAD_W = 1
CMD_OP_PRELOAD_W = 2
CMD_OP_LOAD_A = 3
CMD_OP_GEMM = 4
CMD_OP_STORE_C = 5
CMD_OP_READ_UB = 6
CMD_OP_LOAD_BIAS = 7

CMD_OP_W = 3
QMULT_W = 18
QSHIFT_W = 6
K_VALID_W = 6
ACT_MODE_W = 2
CMD_MT_W = 5
CMD_KT_W = 8
CMD_NT_W = 7
WMEM_SLOT_W = 8
UB_SLOT_W = 9

CMD_OP_LSB = 0
CMD_ACC_INIT_BIT = 3
CMD_ACC_FINAL_BIT = 4
CMD_QMULT_LSB = 8
CMD_QSHIFT_LSB = 26
CMD_MT_LSB = 32
CMD_KT_LSB = 37
CMD_NT_LSB = 45
CMD_W_SLOT_LSB = 52
CMD_SRC_SLOT_LSB = 60
CMD_DST_SLOT_LSB = 69
CMD_K_VALID_LSB = 78
CMD_BIAS_EN_BIT = 84
CMD_ACT_MODE_LSB = 85
CMD_A_MASK_EN_BIT = 87

ACT_NONE = 0
ACT_RELU = 1

OPCODE_NAME = {
    CMD_OP_NOP: "NOP",
    CMD_OP_LOAD_W: "LOAD_W",
    CMD_OP_PRELOAD_W: "PRELOAD_W",
    CMD_OP_LOAD_A: "LOAD_A",
    CMD_OP_GEMM: "GEMM",
    CMD_OP_STORE_C: "STORE_C",
    CMD_OP_READ_UB: "READ_UB",
    CMD_OP_LOAD_BIAS: "LOAD_BIAS",
}

Command = namedtuple(
    "Command",
    "word opcode mt kt nt w_slot src_slot dst_slot acc_init acc_final "
    "k_valid bias_en act_mode a_mask_en",
)


def mask(width):
    return (1 << width) - 1


def set_field(word, value, lsb, width):
    assert 0 <= value <= mask(width), "field value does not fit"
    return word | (value << lsb)


def field(word, lsb, width=1):
    return (word >> lsb) & mask(width)


def signed_field(word, lsb, width):
    value = field(word, lsb, width)
    return value - (1 << width) if value & (1 << (width - 1)) else value


def opcode(word):
    return field(word, CMD_OP_LSB, CMD_OP_W)


def opcode_name(word):
    return OPCODE_NAME.get(opcode(word), "INVALID")


def pack_descriptor(op, *, acc_init=False, acc_final=False, qmult=1,
                    qshift=0, mt=0, kt=0, nt=0, w_slot=0,
                    src_slot=0, dst_slot=0, k_valid=None,
                    bias_en=False, act_mode=ACT_NONE, a_mask_en=False):
    """Pack one 512-bit descriptor. All reserved bits remain zero."""
    assert op in OPCODE_NAME, "unsupported opcode"
    assert 0 <= qmult < (1 << (QMULT_W - 1)), \
        "qmult must be a non-negative signed QMULT_W value"
    assert 0 <= qshift <= mask(QSHIFT_W), "qshift must fit QSHIFT_W"
    if k_valid is None:
        k_valid = ARRAY_S if op == CMD_OP_GEMM else 0

    assert not (acc_init or acc_final) or op == CMD_OP_GEMM, \
        "ACC flags are legal only for GEMM"
    assert (1 <= k_valid <= ARRAY_S) if op == CMD_OP_GEMM else k_valid == 0, \
        "K_VALID must be 1..32 for GEMM and zero otherwise"
    assert not bias_en or (op == CMD_OP_GEMM and acc_init), \
        "BIAS_EN requires GEMM with ACC_INIT"
    assert act_mode in (ACT_NONE, ACT_RELU), "unsupported activation mode"
    assert op == CMD_OP_STORE_C or act_mode == ACT_NONE, \
        "ACT_MODE is legal only for STORE_C"
    assert op == CMD_OP_LOAD_A or not a_mask_en, \
        "A_MASK_EN is legal only for LOAD_A"

    word = 0
    word = set_field(word, op, CMD_OP_LSB, CMD_OP_W)
    word = set_field(word, int(acc_init), CMD_ACC_INIT_BIT, 1)
    word = set_field(word, int(acc_final), CMD_ACC_FINAL_BIT, 1)
    word = set_field(word, qmult & mask(QMULT_W), CMD_QMULT_LSB, QMULT_W)
    word = set_field(word, qshift, CMD_QSHIFT_LSB, QSHIFT_W)
    word = set_field(word, mt, CMD_MT_LSB, CMD_MT_W)
    word = set_field(word, kt, CMD_KT_LSB, CMD_KT_W)
    word = set_field(word, nt, CMD_NT_LSB, CMD_NT_W)
    word = set_field(word, w_slot, CMD_W_SLOT_LSB, WMEM_SLOT_W)
    word = set_field(word, src_slot, CMD_SRC_SLOT_LSB, UB_SLOT_W)
    word = set_field(word, dst_slot, CMD_DST_SLOT_LSB, UB_SLOT_W)
    word = set_field(word, k_valid, CMD_K_VALID_LSB, K_VALID_W)
    word = set_field(word, int(bias_en), CMD_BIAS_EN_BIT, 1)
    word = set_field(word, act_mode, CMD_ACT_MODE_LSB, ACT_MODE_W)
    word = set_field(word, int(a_mask_en), CMD_A_MASK_EN_BIT, 1)
    return word


def make_command(op, *, acc_init=False, acc_final=False, qmult=1,
                 qshift=0, mt=0, kt=0, nt=0, w_slot=0,
                 src_slot=0, dst_slot=0, k_valid=None,
                 bias_en=False, act_mode=ACT_NONE, a_mask_en=False):
    resolved_k_valid = ARRAY_S if k_valid is None and op == CMD_OP_GEMM \
        else (0 if k_valid is None else k_valid)
    word = pack_descriptor(
        op,
        acc_init=acc_init,
        acc_final=acc_final,
        qmult=qmult,
        qshift=qshift,
        mt=mt,
        kt=kt,
        nt=nt,
        w_slot=w_slot,
        src_slot=src_slot,
        dst_slot=dst_slot,
        k_valid=resolved_k_valid,
        bias_en=bias_en,
        act_mode=act_mode,
        a_mask_en=a_mask_en,
    )
    return Command(
        word, op, mt, kt, nt, w_slot, src_slot, dst_slot,
        bool(acc_init), bool(acc_final), resolved_k_valid,
        bool(bias_en), act_mode, bool(a_mask_en),
    )


def decode_descriptor(word):
    """Return all defined descriptor fields for consistency checks."""
    return {
        "opcode": opcode(word),
        "acc_init": field(word, CMD_ACC_INIT_BIT),
        "acc_final": field(word, CMD_ACC_FINAL_BIT),
        "qmult": signed_field(word, CMD_QMULT_LSB, QMULT_W),
        "qshift": field(word, CMD_QSHIFT_LSB, QSHIFT_W),
        "mt": field(word, CMD_MT_LSB, CMD_MT_W),
        "kt": field(word, CMD_KT_LSB, CMD_KT_W),
        "nt": field(word, CMD_NT_LSB, CMD_NT_W),
        "w_slot": field(word, CMD_W_SLOT_LSB, WMEM_SLOT_W),
        "src_slot": field(word, CMD_SRC_SLOT_LSB, UB_SLOT_W),
        "dst_slot": field(word, CMD_DST_SLOT_LSB, UB_SLOT_W),
        "k_valid": field(word, CMD_K_VALID_LSB, K_VALID_W),
        "bias_en": field(word, CMD_BIAS_EN_BIT),
        "act_mode": field(word, CMD_ACT_MODE_LSB, ACT_MODE_W),
        "a_mask_en": field(word, CMD_A_MASK_EN_BIT),
    }


def validate_shape(m_dim, k_dim, n_dim):
    for name, value, limit in (
        ("M", m_dim, M_MAX),
        ("K", k_dim, K_MAX),
        ("N", n_dim, N_MAX),
    ):
        assert value > 0 and value <= limit, "%s exceeds TPU limit" % name
        assert value % ARRAY_S == 0, "%s must be a multiple of 32" % name


def compile_gemm_trace(m_dim, k_dim, n_dim, qmult, qshift, *, readback=False,
                       act_mode=ACT_RELU):
    """Compile one GEMM using persistent WMEM and UB tile slots.

    The first verification compiler keeps every unique input tile resident.
    This is sufficient for the directed suite and makes weight reuse explicit.
    """
    validate_shape(m_dim, k_dim, n_dim)
    mt_total = m_dim // ARRAY_S
    kt_total = k_dim // ARRAY_S
    nt_total = n_dim // ARRAY_S

    weight_tiles = kt_total * nt_total
    activation_tiles = mt_total * kt_total
    output_tiles = mt_total * nt_total
    assert weight_tiles <= WMEM_SLOTS, "test needs more WMEM slots than available"
    assert activation_tiles + output_tiles <= UB_SLOTS, \
        "test needs more UB slots than available"

    commands = []

    for nt in range(nt_total):
        for kt in range(kt_total):
            w_slot = kt * nt_total + nt
            commands.append(make_command(
                CMD_OP_LOAD_W,
                qmult=qmult,
                qshift=qshift,
                kt=kt,
                nt=nt,
                w_slot=w_slot,
            ))

    for mt in range(mt_total):
        for kt in range(kt_total):
            src_slot = mt * kt_total + kt
            commands.append(make_command(
                CMD_OP_LOAD_A,
                qmult=qmult,
                qshift=qshift,
                mt=mt,
                kt=kt,
                src_slot=src_slot,
            ))

    output_base = activation_tiles
    active_w_slot = None
    for nt in range(nt_total):
        for mt in range(mt_total):
            for kt in range(kt_total):
                w_slot = kt * nt_total + nt
                src_slot = mt * kt_total + kt
                if active_w_slot != w_slot:
                    commands.append(make_command(
                        CMD_OP_PRELOAD_W,
                        qmult=qmult,
                        qshift=qshift,
                        mt=mt,
                        kt=kt,
                        nt=nt,
                        w_slot=w_slot,
                    ))
                    active_w_slot = w_slot
                commands.append(make_command(
                    CMD_OP_GEMM,
                    acc_init=(kt == 0),
                    acc_final=(kt == kt_total - 1),
                    qmult=qmult,
                    qshift=qshift,
                    mt=mt,
                    kt=kt,
                    nt=nt,
                    w_slot=w_slot,
                    src_slot=src_slot,
                    k_valid=ARRAY_S,
                ))

            dst_slot = output_base + mt * nt_total + nt
            commands.append(make_command(
                CMD_OP_STORE_C,
                qmult=qmult,
                qshift=qshift,
                mt=mt,
                kt=kt_total - 1,
                nt=nt,
                dst_slot=dst_slot,
                act_mode=act_mode,
            ))

    if readback:
        for nt in range(nt_total):
            for mt in range(mt_total):
                dst_slot = output_base + mt * nt_total + nt
                commands.append(make_command(
                    CMD_OP_READ_UB,
                    qmult=qmult,
                    qshift=qshift,
                    mt=mt,
                    nt=nt,
                    src_slot=dst_slot,
                ))

    return commands


def compile_ub_read_trace(slot, *, qmult=1, qshift=0):
    """Compile LOAD_A followed by READ_UB for one 32-row tile."""
    assert 0 <= slot < UB_SLOTS, "UB slot out of range"
    return [
        make_command(
            CMD_OP_LOAD_A,
            qmult=qmult,
            qshift=qshift,
            src_slot=slot,
        ),
        make_command(
            CMD_OP_READ_UB,
            qmult=qmult,
            qshift=qshift,
            src_slot=slot,
        ),
    ]
