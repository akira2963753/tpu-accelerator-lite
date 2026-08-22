#!/usr/bin/env python3
"""Compile a quantized multi-layer MLP into one TPU replay bundle."""

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
TESTBED_ROOT = os.path.normpath(os.path.join(HERE, ".."))
PYTHON_ROOT = os.path.join(TESTBED_ROOT, "python")
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

import tpu_command as tc  # noqa: E402
import tpu_gen as tg  # noqa: E402


MODEL_FORMAT = "tpu-mlp-v1"
BUNDLE_FORMAT = "tpu-workload-v1"


@dataclass(frozen=True)
class TargetSpec:
    array_size: int = tc.ARRAY_S
    activation_width: int = tg.A_W
    weight_width: int = tg.W_W
    accumulator_width: int = tg.ACC_W
    wmem_slots: int = tc.WMEM_SLOTS
    ub_slots: int = tc.UB_SLOTS


@dataclass(frozen=True)
class TensorIR:
    name: str
    shape: tuple
    scale: float
    layout: str = "MK"


@dataclass(frozen=True)
class LayerIR:
    name: str
    in_features: int
    out_features: int
    weight_key: str
    bias_key: str
    activation: str


@dataclass(frozen=True)
class GemmIR:
    name: str
    m_dim: int
    k_dim: int
    n_dim: int
    mt: int
    kt: int
    nt: int


@dataclass(frozen=True)
class TileIR:
    layer: str
    mt: int
    kt: int
    nt: int
    src_slot: int
    w_slot: int
    dst_slot: int


@dataclass(frozen=True)
class CommandIR:
    layer: str
    opcode: str
    word: int


@dataclass
class QuantizedLayer:
    ir: LayerIR
    raw_weight: np.ndarray
    bias_acc: np.ndarray
    weight_scale: float
    input_scale: float
    output_scale: float
    qmult: int
    qshift: int
    act_mode: int
    stats: dict


@dataclass
class WorkloadBundle:
    commands: list
    weight_rows: list
    activation_rows: list
    golden_rows: list
    manifest: dict


def ceil_div(value, divisor):
    return (int(value) + int(divisor) - 1) // int(divisor)


def round_half_up(values):
    return np.floor(np.asarray(values, dtype=np.float64) + 0.5)


def wrap_signed_np(values, width):
    modulus = 1 << width
    sign = 1 << (width - 1)
    wrapped = np.asarray(values, dtype=np.int64) & (modulus - 1)
    return np.where(wrapped >= sign, wrapped - modulus, wrapped).astype(np.int64)


def signed_payload_np(payload):
    payload = np.asarray(payload, dtype=np.int64) & 0xF
    return np.where(payload >= 8, payload - 16, payload)


def decode_activation_np(payload):
    return signed_payload_np(payload) * 2 + 1


def effective_weight_np(raw_weight):
    raw = np.asarray(raw_weight, dtype=np.int64) & 0xFF
    top = (raw >> 4) & 0xF
    non_msr = (top != 0) & (top != 0xF)
    reduced = np.where(non_msr, top, (raw >> 1) & 0xF)
    value = (reduced << 1) | 1
    value = np.where(value >= 16, value - 32, value)
    return value * np.where(non_msr, 64, 8)


def derive_activation_scale(values):
    peak = float(np.max(np.abs(values))) if np.size(values) else 0.0
    return peak / 15.0 if peak > 0.0 else 1.0


def quantize_activation(values, scale):
    if scale <= 0.0:
        raise ValueError("activation scale must be positive")
    target = np.asarray(values, dtype=np.float64) / float(scale)
    signed_code = round_half_up((target - 1.0) / 2.0)
    signed_code = np.clip(signed_code, -8, 7).astype(np.int64)
    return (signed_code & 0xF).astype(np.uint8)


def activation_stats(values, scale):
    target = np.asarray(values, dtype=np.float64) / float(scale)
    unbounded = round_half_up((target - 1.0) / 2.0)
    return {
        "saturated": int(np.count_nonzero((unbounded < -8) | (unbounded > 7))),
        "total": int(unbounded.size),
    }


def quantize_weight(weight):
    weight = np.asarray(weight, dtype=np.float64)
    peak = float(np.max(np.abs(weight))) if weight.size else 0.0
    raw_scale = peak / 127.0 if peak > 0.0 else 1.0
    unbounded = round_half_up(weight / raw_scale)
    clipped = np.clip(unbounded, -127, 127).astype(np.int64)
    raw = (clipped & 0xFF).astype(np.uint8)
    effective = effective_weight_np(raw).astype(np.float64)
    denominator = float(np.sum(effective * effective))
    scale = float(np.sum(weight * effective) / denominator) \
        if denominator > 0.0 else raw_scale / 8.0
    if scale <= 0.0:
        scale = raw_scale / 8.0
    stats = {
        "raw_scale": raw_scale,
        "effective_scale": scale,
        "saturated": int(np.count_nonzero(np.abs(unbounded) > 127)),
        "total": int(weight.size),
    }
    return raw, scale, stats


def quantize_bias(bias, accumulator_scale):
    bias = np.asarray(bias, dtype=np.float64)
    if accumulator_scale <= 0.0:
        raise ValueError("accumulator scale must be positive")
    quantized = round_half_up(bias / accumulator_scale).astype(np.int64)
    minimum = -(1 << (tg.BIAS_W - 1))
    maximum = (1 << (tg.BIAS_W - 1)) - 1
    overflow = (quantized < minimum) | (quantized > maximum)
    if np.any(overflow):
        raise ValueError("bias exceeds signed %d-bit accumulator range" % tg.BIAS_W)
    return quantized, {
        "saturated": 0,
        "total": int(bias.size),
        "minimum": int(np.min(quantized)) if quantized.size else 0,
        "maximum": int(np.max(quantized)) if quantized.size else 0,
    }


def accumulator_bound(raw_weight, bias_acc):
    effective = np.abs(effective_weight_np(raw_weight).astype(np.int64))
    bound = np.abs(np.asarray(bias_acc, dtype=np.int64)) + \
        15 * np.sum(effective, axis=0)
    maximum = (1 << (tg.ACC_W - 1)) - 1
    if np.any(bound > maximum):
        lane = int(np.argmax(bound))
        raise ValueError(
            "ACC worst-case bound exceeds signed %d-bit range at lane %d" %
            (tg.ACC_W, lane)
        )
    return int(np.max(bound)) if bound.size else 0


def exact_accumulate(payload, raw_weight, bias_acc, k_dim):
    payload = np.asarray(payload, dtype=np.uint8)
    raw_weight = np.asarray(raw_weight, dtype=np.uint8)
    bias_acc = np.asarray(bias_acc, dtype=np.int64)
    if payload.shape[1] < k_dim or raw_weight.shape[0] < k_dim:
        raise ValueError("padded GEMM arrays are smaller than K")
    if raw_weight.shape[1] != bias_acc.shape[0]:
        raise ValueError("bias and weight output dimensions differ")

    activation = decode_activation_np(payload)
    weight = effective_weight_np(raw_weight)
    accumulator = np.broadcast_to(
        bias_acc.reshape(1, -1),
        (payload.shape[0], raw_weight.shape[1]),
    ).copy()
    accumulator = wrap_signed_np(accumulator, tg.ACC_W)

    for k_base in range(0, k_dim, tc.ARRAY_S):
        k_valid = min(tc.ARRAY_S, k_dim - k_base)
        psum = np.matmul(
            activation[:, k_base:k_base + k_valid],
            weight[k_base:k_base + k_valid, :],
        )
        psum = wrap_signed_np(psum, tg.PSUM_W)
        if k_base == 0:
            accumulator = wrap_signed_np(psum + bias_acc, tg.ACC_W)
        else:
            accumulator = wrap_signed_np(accumulator + psum, tg.ACC_W)
    return accumulator


def choose_requant(accumulator, act_mode):
    values = np.asarray(accumulator, dtype=np.int64)
    if act_mode == tc.ACT_RELU:
        peak = int(max(0, int(np.max(values)))) if values.size else 0
    else:
        peak = int(np.max(np.abs(values))) if values.size else 0
    if peak == 0:
        return 1, 0

    factor = 7.0 / float(peak)
    maximum = (1 << (tc.QMULT_W - 1)) - 1
    ratio = maximum / factor
    qshift = math.floor(math.log2(ratio)) if ratio >= 1.0 else 0
    qshift = min(max(qshift, 0), (1 << tc.QSHIFT_W) - 1)
    qmult = int(round_half_up(factor * (1 << qshift)))
    while qmult > maximum and qshift > 0:
        qshift -= 1
        qmult = int(round_half_up(factor * (1 << qshift)))
    return min(max(qmult, 1), maximum), qshift


def output_payload_np(accumulator, qmult, qshift, act_mode):
    values = wrap_signed_np(accumulator, tg.ACC_W)
    activated = np.maximum(values, 0) if act_mode == tc.ACT_RELU else values
    product = wrap_signed_np(activated * int(qmult), tg.ACC_W + tg.QMULT_W)
    quantized = product >> int(qshift)
    if act_mode == tc.ACT_RELU:
        quantized = np.maximum(quantized, 0)
    clipped = np.clip(quantized, tg.SAT_MIN, tg.SAT_MAX).astype(np.int64)
    return (clipped & 0xF).astype(np.uint8), {
        "saturated": int(np.count_nonzero(quantized != clipped)),
        "total": int(clipped.size),
        "minimum": int(np.min(quantized)) if quantized.size else 0,
        "maximum": int(np.max(quantized)) if quantized.size else 0,
    }


def fit_output_scale(reference, payload):
    reference = np.asarray(reference, dtype=np.float64)
    decoded = decode_activation_np(payload).astype(np.float64)
    denominator = float(np.sum(decoded * decoded))
    if denominator == 0.0:
        return 1.0
    scale = float(np.sum(reference * decoded) / denominator)
    return scale if scale > 0.0 else derive_activation_scale(reference)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(model_path):
    model_path = os.path.abspath(model_path)
    with open(model_path, "r", encoding="utf-8") as source:
        model = json.load(source)
    if model.get("format") != MODEL_FORMAT:
        raise ValueError("unsupported model format")

    model_dir = os.path.dirname(model_path)
    tensor_path = os.path.join(model_dir, model["tensor_file"])
    with np.load(tensor_path, allow_pickle=False) as archive:
        tensors = {name: archive[name] for name in archive.files}
    layers = []
    previous_features = int(model["input_features"])
    for entry in model["layers"]:
        activation = entry.get("activation", "none").lower()
        if activation not in ("none", "relu"):
            raise ValueError("unsupported activation %s" % activation)
        layer = LayerIR(
            name=entry["name"],
            in_features=int(entry["in_features"]),
            out_features=int(entry["out_features"]),
            weight_key=entry["weight"],
            bias_key=entry.get("bias", ""),
            activation=activation,
        )
        if layer.in_features != previous_features:
            raise ValueError("layer %s has a broken feature chain" % layer.name)
        if layer.in_features > tc.K_MAX or layer.out_features > tc.N_MAX:
            raise ValueError("layer %s exceeds TPU dimensions" % layer.name)
        weight = tensors[layer.weight_key]
        if tuple(weight.shape) != (layer.out_features, layer.in_features):
            raise ValueError("layer %s weight shape must be [N,K]" % layer.name)
        if (layer.bias_key and
                tuple(tensors[layer.bias_key].shape) != (layer.out_features,)):
            raise ValueError("layer %s bias shape mismatch" % layer.name)
        layers.append(layer)
        previous_features = layer.out_features
    if not layers:
        raise ValueError("model has no Linear layers")
    return model_path, model, tensors, layers


def compile_quantized_layers(model_path):
    model_path, model, tensors, layers = load_model(model_path)
    if "calibration_inputs" not in tensors:
        raise ValueError("tensor package has no calibration_inputs")
    calibration_float = np.asarray(tensors["calibration_inputs"], dtype=np.float64)
    if (calibration_float.ndim != 2 or
            calibration_float.shape[1] != model["input_features"]):
        raise ValueError("calibration input shape mismatch")

    input_scale = derive_activation_scale(calibration_float)
    calibration_payload = quantize_activation(calibration_float, input_scale)
    quantized_layers = []

    for layer in layers:
        weight = np.asarray(tensors[layer.weight_key], dtype=np.float64)
        bias = np.asarray(tensors[layer.bias_key], dtype=np.float64) \
            if layer.bias_key else np.zeros(layer.out_features, dtype=np.float64)
        raw_weight_nk, weight_scale, weight_stats = quantize_weight(weight)
        raw_weight = raw_weight_nk.T.copy()
        accumulator_scale = input_scale * weight_scale
        bias_acc, bias_stats = quantize_bias(bias, accumulator_scale)
        worst_accumulator = accumulator_bound(raw_weight, bias_acc)
        accumulator = exact_accumulate(
            calibration_payload,
            raw_weight,
            bias_acc,
            layer.in_features,
        )
        act_mode = tc.ACT_RELU if layer.activation == "relu" else tc.ACT_NONE
        qmult, qshift = choose_requant(accumulator, act_mode)
        output_payload, output_stats = output_payload_np(
            accumulator, qmult, qshift, act_mode
        )

        reference = np.matmul(calibration_float, weight.T) + bias
        if act_mode == tc.ACT_RELU:
            reference = np.maximum(reference, 0.0)
        output_scale = fit_output_scale(reference, output_payload)
        stats = {
            "activation": activation_stats(calibration_float, input_scale),
            "weight": weight_stats,
            "bias": bias_stats,
            "output": output_stats,
            "accumulator_min": int(np.min(accumulator)),
            "accumulator_max": int(np.max(accumulator)),
            "accumulator_worst_case_bound": worst_accumulator,
        }
        quantized_layers.append(QuantizedLayer(
            ir=layer,
            raw_weight=raw_weight,
            bias_acc=bias_acc,
            weight_scale=weight_scale,
            input_scale=input_scale,
            output_scale=output_scale,
            qmult=qmult,
            qshift=qshift,
            act_mode=act_mode,
            stats=stats,
        ))
        calibration_float = reference
        calibration_payload = output_payload
        input_scale = output_scale
    return model_path, model, tensors, quantized_layers


def allocate_slots(count, excluded, capacity):
    slots = [slot for slot in range(capacity) if slot not in excluded][:count]
    if len(slots) != count:
        raise ValueError("UB capacity exceeded by live tensors")
    return slots


def pad_payload(payload, rows, columns):
    padded = np.zeros((rows, columns), dtype=np.uint8)
    padded[:payload.shape[0], :payload.shape[1]] = payload
    return padded


def pad_weight(raw_weight, rows, columns):
    padded = np.zeros((rows, columns), dtype=np.uint8)
    padded[:raw_weight.shape[0], :raw_weight.shape[1]] = raw_weight
    return padded


def pack_weight_tile_np(weight, kt, nt):
    rows = []
    base_k = kt * tc.ARRAY_S
    base_n = nt * tc.ARRAY_S
    for lane in range(tc.ARRAY_S):
        rows.append(tg.pack_row(
            weight[base_k + lane, base_n:base_n + tc.ARRAY_S],
            tg.W_W,
        ))
    return rows


def pack_activation_tile_np(activation, mt, kt):
    rows = []
    base_m = mt * tc.ARRAY_S
    base_k = kt * tc.ARRAY_S
    for lane in range(tc.ARRAY_S):
        rows.append(tg.pack_row(
            activation[base_m + lane, base_k:base_k + tc.ARRAY_S],
            tg.A_W,
        ))
    return rows


def pack_output_tile_np(output, mt, nt):
    rows = []
    base_m = mt * tc.ARRAY_S
    base_n = nt * tc.ARRAY_S
    for lane in range(tc.ARRAY_S):
        expanded = [
            int(value) << (tg.R_W - tg.A_W)
            for value in output[base_m + lane, base_n:base_n + tc.ARRAY_S]
        ]
        rows.append(tg.pack_row(expanded, tg.R_W))
    return rows


def build_workload(model_path, rtl_inputs, rtl_labels=None):
    model_path, model, _tensors, layers = compile_quantized_layers(model_path)
    rtl_inputs = np.asarray(rtl_inputs, dtype=np.float64)
    if rtl_inputs.ndim != 2 or rtl_inputs.shape[1] != model["input_features"]:
        raise ValueError("RTL input shape mismatch")
    if rtl_inputs.shape[0] < 1 or rtl_inputs.shape[0] > tc.M_MAX:
        raise ValueError("RTL batch must be 1..%d" % tc.M_MAX)

    target = TargetSpec()
    m_dim = int(rtl_inputs.shape[0])
    m_pad = ceil_div(m_dim, target.array_size) * target.array_size
    input_scale = layers[0].input_scale
    payload = quantize_activation(rtl_inputs, input_scale)
    payload = pad_payload(
        payload,
        m_pad,
        ceil_div(model["input_features"], target.array_size) * target.array_size,
    )

    commands = []
    command_ir = []
    weight_rows = []
    activation_rows = []
    golden_rows = []
    layer_manifest = []
    independent_outputs = []

    initial_kt = ceil_div(model["input_features"], target.array_size)
    mt_total = m_pad // target.array_size
    input_slots = list(range(mt_total * initial_kt))
    if len(input_slots) > target.ub_slots:
        raise ValueError("input tensor exceeds UB capacity")

    for layer_index, layer in enumerate(layers):
        ir = layer.ir
        kt_total = ceil_div(ir.in_features, target.array_size)
        nt_total = ceil_div(ir.out_features, target.array_size)
        nt_block_capacity = target.wmem_slots // kt_total
        if nt_block_capacity == 0:
            raise ValueError("layer %s K tiles exceed WMEM capacity" % ir.name)
        nt_block_capacity = min(nt_total, nt_block_capacity)
        weight_blocks = []
        for nt_begin in range(0, nt_total, nt_block_capacity):
            weight_blocks.append({
                "nt_begin": nt_begin,
                "nt_count": min(nt_block_capacity, nt_total - nt_begin),
            })
        if len(input_slots) != mt_total * kt_total:
            raise ValueError("UB input tile layout does not match layer %s" % ir.name)

        output_slots = allocate_slots(
            mt_total * nt_total,
            set(input_slots),
            target.ub_slots,
        )
        k_pad = kt_total * target.array_size
        n_pad = nt_total * target.array_size
        padded_weight = pad_weight(layer.raw_weight, k_pad, n_pad)
        padded_bias = np.zeros(n_pad, dtype=np.int64)
        padded_bias[:ir.out_features] = layer.bias_acc

        layer_start = len(commands)
        if layer_index == 0:
            for mt in range(mt_total):
                for kt in range(kt_total):
                    src_slot = input_slots[mt * kt_total + kt]
                    command = tc.make_command(
                        tc.CMD_OP_LOAD_A,
                        mt=mt,
                        kt=kt,
                        src_slot=src_slot,
                    )
                    commands.append(command)
                    command_ir.append(CommandIR(ir.name, "LOAD_A", command.word))
                    activation_rows.extend(pack_activation_tile_np(payload, mt, kt))

        accumulator = exact_accumulate(
            payload,
            padded_weight,
            padded_bias,
            ir.in_features,
        )
        output, rtl_output_stats = output_payload_np(
            accumulator,
            layer.qmult,
            layer.qshift,
            layer.act_mode,
        )

        for block in weight_blocks:
            nt_begin = block["nt_begin"]
            nt_count = block["nt_count"]
            for local_nt in range(nt_count):
                nt = nt_begin + local_nt
                for kt in range(kt_total):
                    w_slot = kt * nt_count + local_nt
                    command = tc.make_command(
                        tc.CMD_OP_LOAD_W,
                        kt=kt,
                        nt=nt,
                        w_slot=w_slot,
                    )
                    commands.append(command)
                    command_ir.append(CommandIR(ir.name, "LOAD_W", command.word))
                    weight_rows.extend(pack_weight_tile_np(padded_weight, kt, nt))

            for local_nt in range(nt_count):
                nt = nt_begin + local_nt
                if ir.bias_key:
                    bias = padded_bias[
                        nt * target.array_size:(nt + 1) * target.array_size
                    ]
                    command = tc.make_command(tc.CMD_OP_LOAD_BIAS, nt=nt)
                    commands.append(command)
                    command_ir.append(CommandIR(ir.name, "LOAD_BIAS", command.word))
                    weight_rows.extend(tg.pack_bias([int(value) for value in bias]))

                for mt in range(mt_total):
                    for kt in range(kt_total):
                        w_slot = kt * nt_count + local_nt
                        src_slot = input_slots[mt * kt_total + kt]
                        command = tc.make_command(
                            tc.CMD_OP_PRELOAD_W,
                            mt=mt,
                            kt=kt,
                            nt=nt,
                            w_slot=w_slot,
                        )
                        commands.append(command)
                        command_ir.append(CommandIR(
                            ir.name, "PRELOAD_W", command.word
                        ))
                        command = tc.make_command(
                            tc.CMD_OP_GEMM,
                            mt=mt,
                            kt=kt,
                            nt=nt,
                            w_slot=w_slot,
                            src_slot=src_slot,
                            acc_init=(kt == 0),
                            acc_final=(kt == kt_total - 1),
                            k_valid=min(
                                target.array_size,
                                ir.in_features - kt * target.array_size,
                            ),
                            bias_en=(kt == 0 and bool(ir.bias_key)),
                        )
                        commands.append(command)
                        command_ir.append(CommandIR(ir.name, "GEMM", command.word))

                    dst_slot = output_slots[mt * nt_total + nt]
                    command = tc.make_command(
                        tc.CMD_OP_STORE_C,
                        qmult=layer.qmult,
                        qshift=layer.qshift,
                        mt=mt,
                        kt=kt_total - 1,
                        nt=nt,
                        dst_slot=dst_slot,
                        act_mode=layer.act_mode,
                    )
                    commands.append(command)
                    command_ir.append(CommandIR(ir.name, "STORE_C", command.word))
                    golden_rows.extend(pack_output_tile_np(output, mt, nt))

        valid_output = output[:m_dim, :ir.out_features].copy()
        independent_outputs.append(valid_output)
        layer_manifest.append({
            "name": ir.name,
            "shape": {"M": m_dim, "K": ir.in_features, "N": ir.out_features},
            "tiles": {"MT": mt_total, "KT": kt_total, "NT": nt_total},
            "activation": ir.activation,
            "act_mode": layer.act_mode,
            "qmult": layer.qmult,
            "qshift": layer.qshift,
            "input_scale": layer.input_scale,
            "weight_scale": layer.weight_scale,
            "output_scale": layer.output_scale,
            "input_slots": input_slots,
            "output_slots": output_slots,
            "weight_tiles": kt_total * nt_total,
            "weight_slots": kt_total * nt_block_capacity,
            "weight_blocks": weight_blocks,
            "command_begin": layer_start,
            "command_count": len(commands) - layer_start,
            "stats": dict(layer.stats, rtl_output=rtl_output_stats),
        })
        payload = output
        input_slots = output_slots

    replay_rows = tg.replay_commands(commands, weight_rows, activation_rows)
    if replay_rows != golden_rows:
        mismatch = next(
            index
            for index, pair in enumerate(zip(replay_rows, golden_rows))
            if pair[0] != pair[1]
        )
        raise AssertionError(
            "independent golden differs from command replay at row %d" % mismatch
        )

    final_payload = independent_outputs[-1]
    final_signed = signed_payload_np(final_payload)
    predictions = np.argmax(final_signed, axis=1).astype(np.int64)
    labels = None if rtl_labels is None else np.asarray(
        rtl_labels, dtype=np.int64
    )[:m_dim]
    if labels is not None and labels.shape != (m_dim,):
        raise ValueError("RTL label shape mismatch")

    manifest = {
        "format": BUNDLE_FORMAT,
        "name": model.get("name", "mlp"),
        "source_model": os.path.basename(model_path),
        "target": asdict(target),
        "test": {
            "name": "mnist_mlp",
            "M": m_dim,
            "K": int(model["input_features"]),
            "N": int(layers[-1].ir.out_features),
            "command_count": len(commands),
            "weight_rows": len(weight_rows),
            "activation_rows": len(activation_rows),
            "golden_rows": len(golden_rows),
        },
        "layers": layer_manifest,
        "predictions": predictions.tolist(),
        "labels": None if labels is None else labels.tolist(),
        "rtl_batch_accuracy": None if labels is None else
            float(np.mean(predictions == labels)),
        "files": {},
    }
    return WorkloadBundle(
        commands=commands,
        weight_rows=weight_rows,
        activation_rows=activation_rows,
        golden_rows=golden_rows,
        manifest=manifest,
    )


def write_test_suite(path, bundle):
    test = bundle.manifest["test"]
    with open(path, "w", newline="\n") as output:
        output.write("// Auto-generated by model/tpu_compiler.py.\n\n")
        output.write("    localparam int NUM_TESTS     = 1;\n")
        output.write(
            "    localparam int MAX_CMD_COUNT = %d;\n" % test["command_count"]
        )
        output.write(
            "    localparam int MAX_W_ROWS    = %d;\n" % test["weight_rows"]
        )
        output.write(
            "    localparam int MAX_A_ROWS    = %d;\n" % test["activation_rows"]
        )
        output.write(
            "    localparam int MAX_R_ROWS    = %d;\n\n" % test["golden_rows"]
        )
        output.write(
            "    localparam int TEST_M          [NUM_TESTS] = '{%d};\n" %
            test["M"]
        )
        output.write(
            "    localparam int TEST_K          [NUM_TESTS] = '{%d};\n" %
            test["K"]
        )
        output.write(
            "    localparam int TEST_N          [NUM_TESTS] = '{%d};\n" %
            test["N"]
        )
        output.write(
            "    localparam int TEST_CMD_COUNT  [NUM_TESTS] = '{%d};\n" %
            test["command_count"]
        )
        output.write(
            "    localparam int TEST_W_ROWS     [NUM_TESTS] = '{%d};\n" %
            test["weight_rows"]
        )
        output.write(
            "    localparam int TEST_A_ROWS     [NUM_TESTS] = '{%d};\n" %
            test["activation_rows"]
        )
        output.write(
            "    localparam int TEST_R_ROWS     [NUM_TESTS] = '{%d};\n" %
            test["golden_rows"]
        )
        output.write("    localparam int TEST_RSTALL     [NUM_TESTS] = '{0};\n")
        output.write("    localparam int TEST_WSTALL     [NUM_TESTS] = '{0};\n")
        output.write("    localparam int TEST_ASTALL     [NUM_TESTS] = '{0};\n")
        output.write(
            "    localparam string TEST_NAME [NUM_TESTS] = "
            "'{\"mnist_mlp\"};\n"
        )


def write_summary(path, bundle):
    manifest = bundle.manifest
    test = manifest["test"]
    with open(path, "w", newline="\n") as output:
        output.write("TPU MNIST MLP workload\n")
        output.write("=" * 72 + "\n")
        output.write("shape          : %d x %d -> %d\n" %
                     (test["M"], test["K"], test["N"]))
        output.write("commands       : %d\n" % test["command_count"])
        output.write("stream rows    : W=%d A=%d R=%d\n" %
                     (test["weight_rows"], test["activation_rows"],
                      test["golden_rows"]))
        for layer in manifest["layers"]:
            shape = layer["shape"]
            tiles = layer["tiles"]
            output.write(
                "%-10s M=%-4d K=%-4d N=%-4d tiles=%dx%dx%d "
                "q=%d/2^%d act=%s cmds=%d\n" %
                (layer["name"], shape["M"], shape["K"], shape["N"],
                 tiles["MT"], tiles["KT"], tiles["NT"], layer["qmult"],
                 layer["qshift"], layer["activation"], layer["command_count"])
            )
        output.write("predictions    : %s\n" % manifest["predictions"])
        if manifest["rtl_batch_accuracy"] is not None:
            output.write("batch accuracy : %.4f\n" % manifest["rtl_batch_accuracy"])


def write_bundle(outdir, bundle):
    os.makedirs(outdir, exist_ok=True)
    for name in os.listdir(outdir):
        if name.startswith("t") and name.endswith(".dat"):
            os.remove(os.path.join(outdir, name))

    paths = {
        "command": os.path.join(outdir, "t00_command.dat"),
        "weight": os.path.join(outdir, "t00_weight.dat"),
        "activation": os.path.join(outdir, "t00_activation.dat"),
        "golden": os.path.join(outdir, "t00_golden.dat"),
        "test_suite": os.path.join(outdir, "test_suite.svh"),
        "summary": os.path.join(outdir, "suite_summary.txt"),
    }
    tg.write_dat(paths["command"], [command.word for command in bundle.commands],
                 tc.CMD_DESC_W)
    tg.write_dat(paths["weight"], bundle.weight_rows, tg.W_W * tc.ARRAY_S)
    tg.write_dat(paths["activation"], bundle.activation_rows,
                 tg.A_W * tc.ARRAY_S)
    tg.write_dat(paths["golden"], bundle.golden_rows, tg.R_W * tc.ARRAY_S)
    write_test_suite(paths["test_suite"], bundle)
    write_summary(paths["summary"], bundle)

    bundle.manifest["files"] = {
        key: {
            "name": os.path.basename(path),
            "sha256": sha256_file(path),
        }
        for key, path in paths.items()
    }
    manifest_path = os.path.join(outdir, "manifest.json")
    with open(manifest_path, "w", newline="\n", encoding="utf-8") as output:
        json.dump(bundle.manifest, output, indent=2, sort_keys=True)
        output.write("\n")
    return manifest_path


def read_dat(path):
    with open(path, "r", encoding="ascii") as source:
        return [int(line.strip(), 16) for line in source if line.strip()]


def command_from_word(word):
    fields = tc.decode_descriptor(word)
    return tc.Command(
        word,
        fields["opcode"],
        fields["mt"],
        fields["kt"],
        fields["nt"],
        fields["w_slot"],
        fields["src_slot"],
        fields["dst_slot"],
        bool(fields["acc_init"]),
        bool(fields["acc_final"]),
        fields["k_valid"],
        bool(fields["bias_en"]),
        fields["act_mode"],
        bool(fields["a_mask_en"]),
    )


def replay_bundle(bundle_path):
    manifest_path = bundle_path
    if os.path.isdir(manifest_path):
        manifest_path = os.path.join(manifest_path, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as source:
        manifest = json.load(source)
    if manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError("unsupported workload bundle")
    root = os.path.dirname(os.path.abspath(manifest_path))

    for entry in manifest["files"].values():
        path = os.path.join(root, entry["name"])
        if sha256_file(path) != entry["sha256"]:
            raise AssertionError("checksum mismatch: %s" % entry["name"])

    commands = [command_from_word(word) for word in
                read_dat(os.path.join(root, manifest["files"]["command"]["name"]))]
    weight_rows = read_dat(os.path.join(root, manifest["files"]["weight"]["name"]))
    activation_rows = read_dat(os.path.join(
        root, manifest["files"]["activation"]["name"]
    ))
    golden_rows = read_dat(os.path.join(root, manifest["files"]["golden"]["name"]))
    test = manifest["test"]
    counts = (len(commands), len(weight_rows), len(activation_rows), len(golden_rows))
    expected = (test["command_count"], test["weight_rows"],
                test["activation_rows"], test["golden_rows"])
    if counts != expected:
        raise AssertionError("bundle row counts do not match manifest")
    replay_rows = tg.replay_commands(commands, weight_rows, activation_rows)
    if replay_rows != golden_rows:
        raise AssertionError("bundle replay differs from golden output")
    return manifest


def hardware_inference(model_path, inputs, batch_size=256):
    _, _, _, layers = compile_quantized_layers(model_path)
    inputs = np.asarray(inputs, dtype=np.float64)
    predictions = []
    for base in range(0, inputs.shape[0], batch_size):
        values = inputs[base:base + batch_size]
        payload = quantize_activation(values, layers[0].input_scale)
        for layer in layers:
            accumulator = exact_accumulate(
                payload, layer.raw_weight, layer.bias_acc, layer.ir.in_features
            )
            payload, _ = output_payload_np(
                accumulator, layer.qmult, layer.qshift, layer.act_mode
            )
        predictions.extend(np.argmax(signed_payload_np(payload), axis=1).tolist())
    return np.asarray(predictions, dtype=np.int64)


def compile_command(args):
    model_path, model, tensors, _ = load_model(args.model)
    if "rtl_inputs" not in tensors:
        raise ValueError("tensor package has no rtl_inputs")
    rtl_inputs = tensors["rtl_inputs"][args.sample_offset:
                                        args.sample_offset + args.batch_size]
    labels = tensors["rtl_labels"][args.sample_offset:
                                    args.sample_offset + args.batch_size] \
        if "rtl_labels" in tensors else None
    if rtl_inputs.shape[0] != args.batch_size:
        raise ValueError("requested RTL batch is not available")
    bundle = build_workload(model_path, rtl_inputs, labels)
    outdir = os.path.abspath(args.outdir or os.path.join(TESTBED_ROOT, "pattern"))
    manifest_path = write_bundle(outdir, bundle)
    replay_bundle(manifest_path)
    print("[PASS] independent MLP model matches command replay")
    print("[PASS] bundle replay matches generated golden")
    print("generated %d-layer MLP -> %s" % (len(model["layers"]), outdir))
    print("commands=%d rows=%d/%d/%d" % (
        len(bundle.commands), len(bundle.weight_rows),
        len(bundle.activation_rows), len(bundle.golden_rows),
    ))


def replay_command(args):
    manifest = replay_bundle(args.bundle)
    print("[PASS] %s bundle replay passed (%d commands)" %
          (manifest["name"], manifest["test"]["command_count"]))


def evaluate_command(args):
    _, _, tensors, _ = load_model(args.model)
    if "test_inputs" not in tensors or "test_labels" not in tensors:
        raise ValueError("tensor package has no full MNIST test set")
    inputs = tensors["test_inputs"]
    labels = tensors["test_labels"].astype(np.int64)
    if args.limit:
        inputs = inputs[:args.limit]
        labels = labels[:args.limit]
    predictions = hardware_inference(args.model, inputs, args.batch_size)
    accuracy = float(np.mean(predictions == labels))
    print("hardware-aware accuracy: %.2f%% (%d/%d)" %
          (accuracy * 100.0, int(np.sum(predictions == labels)), len(labels)))


def build_parser():
    parser = argparse.ArgumentParser(description="TPU MLP model compiler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile", help="compile one MLP bundle")
    compile_parser.add_argument("model", help="model JSON file")
    compile_parser.add_argument("--outdir", default=None)
    compile_parser.add_argument("--batch-size", type=int, default=32)
    compile_parser.add_argument("--sample-offset", type=int, default=0)
    compile_parser.set_defaults(function=compile_command)

    replay_parser = subparsers.add_parser("replay", help="replay one bundle")
    replay_parser.add_argument("bundle", help="bundle directory or manifest.json")
    replay_parser.set_defaults(function=replay_command)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate the hardware model on exported MNIST data"
    )
    evaluate_parser.add_argument("model", help="model JSON file")
    evaluate_parser.add_argument("--batch-size", type=int, default=256)
    evaluate_parser.add_argument("--limit", type=int, default=0)
    evaluate_parser.set_defaults(function=evaluate_command)
    return parser


def main():
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
