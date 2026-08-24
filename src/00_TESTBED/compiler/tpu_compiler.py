#!/usr/bin/env python3
"""Compile portable models for the 32x32 Type-2 TPU target."""

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


if __name__ == "__main__":
    sys.modules.setdefault("tpu_compiler", sys.modules[__name__])


MODEL_FORMAT = "tpu-model-v1"
LEGACY_MODEL_FORMAT = "tpu-mlp-v1"
COMPILED_FORMAT = "tpu-compiled-v1"
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
    op_type: str = "linear"
    input_tensor: str = ""
    output_tensor: str = ""
    input_shape: tuple = ()
    output_shape: tuple = ()
    kernel_size: tuple = (1, 1)
    stride: tuple = (1, 1)
    padding: tuple = (0, 0)
    dilation: tuple = (1, 1)
    groups: int = 1
    bn_weight_key: str = ""
    bn_bias_key: str = ""
    bn_mean_key: str = ""
    bn_var_key: str = ""
    bn_eps: float = 1.0e-5


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


def choose_a5_scale(values):
    values = np.asarray(values, dtype=np.float64)
    magnitude = np.abs(values[np.isfinite(values)])
    if magnitude.size == 0 or float(np.max(magnitude)) == 0.0:
        return 1.0e-12

    percentiles = (90.0, 95.0, 97.0, 98.0, 99.0, 99.5,
                   99.9, 99.95, 99.99, 100.0)
    best_scale = None
    best_error = None
    for percentile in percentiles:
        clip = float(np.percentile(magnitude, percentile))
        if clip <= 0.0:
            continue
        scale = clip / 15.0
        payload = quantize_activation(values, scale)
        reconstructed = decode_activation_np(payload) * scale
        error = float(np.mean((reconstructed - values) ** 2))
        if best_error is None or error < best_error:
            best_scale = scale
            best_error = error
    return best_scale


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


def exact_accumulate(payload, raw_weight, bias_acc, k_dim, valid_mask=None):
    payload = np.asarray(payload, dtype=np.uint8)
    raw_weight = np.asarray(raw_weight, dtype=np.uint8)
    bias_acc = np.asarray(bias_acc, dtype=np.int64)
    if payload.shape[1] < k_dim or raw_weight.shape[0] < k_dim:
        raise ValueError("padded GEMM arrays are smaller than K")
    if raw_weight.shape[1] != bias_acc.shape[0]:
        raise ValueError("bias and weight output dimensions differ")

    activation = decode_activation_np(payload)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != payload.shape:
            raise ValueError("activation payload and validity mask differ")
        activation = np.where(valid_mask, activation, 0)
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


def encode_requant_factor(factor):
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("requant factor must be finite and positive")
    maximum = (1 << (tc.QMULT_W - 1)) - 1
    if factor > maximum:
        raise ValueError("requant factor exceeds QMULT range")

    ratio = maximum / factor
    qshift = math.floor(math.log2(ratio)) if ratio >= 1.0 else 0
    qshift = min(max(qshift, 0), (1 << tc.QSHIFT_W) - 1)
    qmult = int(round_half_up(factor * (1 << qshift)))
    while qmult > maximum and qshift > 0:
        qshift -= 1
        qmult = int(round_half_up(factor * (1 << qshift)))
    if qmult < 1:
        raise ValueError("requant factor is smaller than descriptor precision")
    return qmult, qshift


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


def choose_rank_requant(accumulator, reference, base_factor, act_mode):
    accumulator = np.asarray(accumulator, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.float64)
    ideal_rank = np.argmax(accumulator, axis=1)
    candidates = set()
    for ratio in np.geomspace(0.125, 8.0, 257):
        factor = base_factor * float(ratio)
        try:
            candidates.add(encode_requant_factor(factor))
        except ValueError:
            continue
    candidates.add(encode_requant_factor(base_factor))

    best = None
    for qmult, qshift in candidates:
        payload, output_stats = output_payload_np(
            accumulator, qmult, qshift, act_mode
        )
        signed = signed_payload_np(payload)
        quantized_rank = np.argmax(signed, axis=1)
        agreement = int(np.count_nonzero(quantized_rank == ideal_rank))
        maximum = np.max(signed, axis=1, keepdims=True)
        tie_rows = int(np.count_nonzero(
            np.count_nonzero(signed == maximum, axis=1) > 1
        ))
        output_scale = fit_output_scale(reference, payload)
        reconstructed = decode_activation_np(
            payload
        ).astype(np.float64) * output_scale
        mse = float(np.mean((reconstructed - reference) ** 2))
        encoded_factor = qmult / float(1 << qshift)
        distance = abs(math.log2(encoded_factor / base_factor))
        score = agreement, -tie_rows, -mse, -distance
        if best is None or score > best[0]:
            best = (
                score,
                qmult,
                qshift,
                payload,
                output_stats,
                output_scale,
                {
                    "agreement": agreement,
                    "total": int(accumulator.shape[0]),
                    "tie_rows": tie_rows,
                    "candidate_count": len(candidates),
                    "base_factor": base_factor,
                    "selected_factor": encoded_factor,
                },
            )
    if best is None:
        raise ValueError("rank-aware calibration has no legal requant candidate")
    return best[1:]


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


def pair(value, name):
    if isinstance(value, int):
        result = (int(value), int(value))
    else:
        result = tuple(int(item) for item in value)
    if len(result) != 2 or any(item < 0 for item in result):
        raise ValueError("%s must contain two non-negative integers" % name)
    return result


def conv_output_shape(input_shape, out_channels, kernel_size, stride,
                      padding, dilation):
    _channels, height, width = input_shape
    output_height = (
        height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1
    output_width = (
        width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1
    if output_height <= 0 or output_width <= 0:
        raise ValueError("Conv2D output has a non-positive spatial dimension")
    return int(out_channels), output_height, output_width


def fold_batch_norm(weight, bias, tensors, layer):
    weight = np.asarray(weight, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if not layer.bn_weight_key:
        return weight, bias
    gamma = np.asarray(tensors[layer.bn_weight_key], dtype=np.float64)
    beta = np.asarray(tensors[layer.bn_bias_key], dtype=np.float64)
    mean = np.asarray(tensors[layer.bn_mean_key], dtype=np.float64)
    variance = np.asarray(tensors[layer.bn_var_key], dtype=np.float64)
    factor = gamma / np.sqrt(variance + layer.bn_eps)
    reshape = (factor.shape[0],) + (1,) * (weight.ndim - 1)
    folded_weight = weight * factor.reshape(reshape)
    folded_bias = beta + (bias - mean) * factor
    return folded_weight, folded_bias


def im2col_payload(payload, layer):
    payload = np.asarray(payload, dtype=np.uint8)
    if payload.ndim != 4:
        raise ValueError("Conv2D input payload must use NCHW layout")
    batch, channels, height, width = payload.shape
    if (channels, height, width) != layer.input_shape:
        raise ValueError("Conv2D input tensor shape mismatch for %s" % layer.name)
    output_channels, output_height, output_width = layer.output_shape
    kernel_height, kernel_width = layer.kernel_size
    matrix = np.zeros(
        (batch * output_height * output_width, layer.in_features),
        dtype=np.uint8,
    )
    valid = np.zeros(matrix.shape, dtype=bool)
    row = 0
    for batch_index in range(batch):
        for output_row in range(output_height):
            for output_column in range(output_width):
                column = 0
                for channel in range(channels):
                    for kernel_row in range(kernel_height):
                        input_row = (
                            output_row * layer.stride[0] - layer.padding[0] +
                            kernel_row * layer.dilation[0]
                        )
                        for kernel_column in range(kernel_width):
                            input_column = (
                                output_column * layer.stride[1] -
                                layer.padding[1] +
                                kernel_column * layer.dilation[1]
                            )
                            if (0 <= input_row < height and
                                    0 <= input_column < width):
                                matrix[row, column] = payload[
                                    batch_index,
                                    channel,
                                    input_row,
                                    input_column,
                                ]
                                valid[row, column] = True
                            column += 1
                row += 1
    return matrix, valid


def matrix_to_nchw(matrix, batch, output_shape):
    channels, height, width = output_shape
    matrix = np.asarray(matrix)
    expected = (batch * height * width, channels)
    if matrix.shape != expected:
        raise ValueError("matrix output shape does not match NCHW tensor")
    return matrix.reshape(batch, height, width, channels).transpose(0, 3, 1, 2)


def load_model(model_path):
    model_path = os.path.abspath(model_path)
    with open(model_path, "r", encoding="utf-8") as source:
        model = json.load(source)
    if model.get("format") not in (MODEL_FORMAT, LEGACY_MODEL_FORMAT):
        raise ValueError("unsupported model format")

    model_dir = os.path.dirname(model_path)
    tensor_path = os.path.join(model_dir, model["tensor_file"])
    with np.load(tensor_path, allow_pickle=False) as archive:
        tensors = {name: archive[name] for name in archive.files}
    layers = []
    previous_features = int(model.get("input_features", 0))
    root_input = model.get("input", {}).get("name", "input")
    previous_tensor = root_input
    available_tensors = {root_input}
    for entry in model["layers"]:
        op_type = entry.get("type", "linear").lower()
        if op_type not in ("linear", "conv2d"):
            raise ValueError("unsupported TPU operator %s" % op_type)
        activation = entry.get("activation", "none").lower()
        if activation not in ("none", "relu"):
            raise ValueError("unsupported activation %s" % activation)
        input_tensor = entry.get("input", previous_tensor)
        output_tensor = entry.get("output", entry["name"] + "_output")
        if input_tensor not in available_tensors:
            raise ValueError("layer %s uses unavailable tensor %s" %
                             (entry["name"], input_tensor))
        if op_type == "linear":
            layer = LayerIR(
                name=entry["name"],
                in_features=int(entry["in_features"]),
                out_features=int(entry["out_features"]),
                weight_key=entry["weight"],
                bias_key=entry.get("bias", ""),
                activation=activation,
                op_type=op_type,
                input_tensor=input_tensor,
                output_tensor=output_tensor,
            )
            if layer.in_features != previous_features:
                raise ValueError("layer %s has a broken feature chain" % layer.name)
            if tuple(tensors[layer.weight_key].shape) != (
                    layer.out_features, layer.in_features):
                raise ValueError("layer %s weight shape must be [N,K]" % layer.name)
        else:
            input_shape = tuple(int(value) for value in entry["input_shape"])
            if len(input_shape) != 3:
                raise ValueError("Conv2D input_shape must be [C,H,W]")
            in_channels = int(entry["in_channels"])
            out_channels = int(entry["out_channels"])
            kernel_size = pair(entry["kernel_size"], "kernel_size")
            stride = pair(entry.get("stride", 1), "stride")
            padding = pair(entry.get("padding", 0), "padding")
            dilation = pair(entry.get("dilation", 1), "dilation")
            if any(value == 0 for value in kernel_size + stride + dilation):
                raise ValueError(
                    "kernel_size, stride and dilation must be positive"
                )
            groups = int(entry.get("groups", 1))
            if groups != 1:
                raise ValueError("Conv2D groups other than 1 are not implemented")
            if input_shape[0] != in_channels:
                raise ValueError("Conv2D input channel metadata mismatch")
            output_shape = conv_output_shape(
                input_shape, out_channels, kernel_size, stride, padding, dilation
            )
            declared_output = tuple(
                int(value) for value in entry.get("output_shape", output_shape)
            )
            if declared_output != output_shape:
                raise ValueError("Conv2D output_shape metadata mismatch")
            batch_norm = entry.get("batch_norm", {})
            layer = LayerIR(
                name=entry["name"],
                in_features=in_channels * kernel_size[0] * kernel_size[1],
                out_features=out_channels,
                weight_key=entry["weight"],
                bias_key=entry.get("bias", ""),
                activation=activation,
                op_type=op_type,
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                input_shape=input_shape,
                output_shape=output_shape,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bn_weight_key=batch_norm.get("weight", ""),
                bn_bias_key=batch_norm.get("bias", ""),
                bn_mean_key=batch_norm.get("running_mean", ""),
                bn_var_key=batch_norm.get("running_var", ""),
                bn_eps=float(batch_norm.get("eps", 1.0e-5)),
            )
            expected_weight = (
                out_channels, in_channels, kernel_size[0], kernel_size[1]
            )
            if tuple(tensors[layer.weight_key].shape) != expected_weight:
                raise ValueError("layer %s weight shape must be OIHW" % layer.name)
            if layer.bn_weight_key:
                for key in (
                    layer.bn_weight_key,
                    layer.bn_bias_key,
                    layer.bn_mean_key,
                    layer.bn_var_key,
                ):
                    if key not in tensors or tuple(tensors[key].shape) != (
                            out_channels,):
                        raise ValueError("layer %s BatchNorm tensor mismatch" %
                                         layer.name)
        if layer.in_features > tc.K_MAX or layer.out_features > tc.N_MAX:
            raise ValueError("layer %s exceeds TPU dimensions" % layer.name)
        if (layer.bias_key and
                tuple(tensors[layer.bias_key].shape) != (layer.out_features,)):
            raise ValueError("layer %s bias shape mismatch" % layer.name)
        layers.append(layer)
        previous_features = layer.out_features
        previous_tensor = output_tensor
        available_tensors.add(output_tensor)
    if not layers:
        raise ValueError("model has no TPU layers")
    for operation in model.get("host_operations", []):
        if operation.get("type") != "add":
            raise ValueError("unsupported Host operator %s" % operation.get("type"))
        if any(name not in available_tensors for name in operation["inputs"]):
            raise ValueError("Host add uses an unavailable tensor")
        available_tensors.add(operation["output"])
    return model_path, model, tensors, layers


def compile_linear_quantized_layers(model_path):
    model_path, model, tensors, layers = load_model(model_path)
    if "calibration_inputs" not in tensors:
        raise ValueError("tensor package has no calibration_inputs")
    calibration_float = np.asarray(tensors["calibration_inputs"], dtype=np.float64)
    if (calibration_float.ndim != 2 or
            calibration_float.shape[1] != model["input_features"]):
        raise ValueError("calibration input shape mismatch")

    input_scale = choose_a5_scale(calibration_float)
    calibration_payload = quantize_activation(calibration_float, input_scale)
    quantized_layers = []

    output_semantics = model.get("output_semantics")
    if output_semantics is None and model.get("format") == LEGACY_MODEL_FORMAT:
        output_semantics = "classification_logits"

    for layer_index, layer in enumerate(layers):
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

        activation_float = decode_activation_np(
            calibration_payload
        ).astype(np.float64) * input_scale
        effective_weight = effective_weight_np(
            raw_weight_nk
        ).astype(np.float64) * weight_scale
        reference = np.matmul(activation_float, effective_weight.T) + bias
        if act_mode == tc.ACT_RELU:
            reference = np.maximum(reference, 0.0)

        output_scale = choose_a5_scale(reference)
        requant_factor = accumulator_scale / (2.0 * output_scale)
        rank_stats = None
        if (layer_index == len(layers) - 1 and
                output_semantics == "classification_logits"):
            (qmult, qshift, output_payload, output_stats,
             output_scale, rank_stats) = choose_rank_requant(
                accumulator, reference, requant_factor, act_mode
            )
        else:
            qmult, qshift = encode_requant_factor(requant_factor)
            output_payload, output_stats = output_payload_np(
                accumulator, qmult, qshift, act_mode
            )
        ideal_payload = quantize_activation(reference, output_scale)
        reconstructed = decode_activation_np(
            output_payload
        ).astype(np.float64) * output_scale
        encoded_factor = qmult / float(1 << qshift)
        stats = {
            "activation": activation_stats(calibration_float, input_scale),
            "weight": weight_stats,
            "bias": bias_stats,
            "output": output_stats,
            "requant": {
                "target_factor": requant_factor,
                "encoded_factor": encoded_factor,
                "relative_factor_error": abs(
                    encoded_factor - requant_factor
                ) / requant_factor,
                "payload_mismatch": int(np.count_nonzero(
                    output_payload != ideal_payload
                )),
                "total": int(output_payload.size),
                "reconstruction_mse": float(np.mean(
                    (reconstructed - reference) ** 2
                )),
                "mode": "rank" if rank_stats else "scale",
                "rank": rank_stats,
            },
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


def compile_conv_quantized_layers(model_path):
    model_path, source_model, tensors, layers = load_model(model_path)
    if "calibration_inputs" not in tensors:
        raise ValueError("tensor package has no calibration_inputs")
    root_name = source_model.get("input", {}).get("name", "input")
    root_values = np.asarray(tensors["calibration_inputs"], dtype=np.float64)
    declared_shape = tuple(
        int(value) for value in source_model["input"]["shape"][1:]
    )
    if root_values.ndim != 4 or tuple(root_values.shape[1:]) != declared_shape:
        raise ValueError("calibration input shape mismatch")

    tensor_states = {
        root_name: {
            "float": root_values,
            "payload": None,
            "scale": None,
        }
    }
    quantized_layers = []
    for layer in layers:
        if layer.op_type != "conv2d":
            raise ValueError("mixed Linear and Conv2D graphs are not implemented")
        state = tensor_states[layer.input_tensor]
        input_float = np.asarray(state["float"], dtype=np.float64)
        input_scale = state["scale"]
        input_payload = state["payload"]
        if input_scale is None:
            input_scale = choose_a5_scale(input_float)
        if input_payload is None:
            input_payload = quantize_activation(input_float, input_scale)

        weight = np.asarray(tensors[layer.weight_key], dtype=np.float64)
        bias = np.asarray(tensors[layer.bias_key], dtype=np.float64) \
            if layer.bias_key else np.zeros(layer.out_features, dtype=np.float64)
        weight, bias = fold_batch_norm(weight, bias, tensors, layer)
        weight_nk = weight.reshape(layer.out_features, layer.in_features)
        raw_weight_nk, weight_scale, weight_stats = quantize_weight(weight_nk)
        raw_weight = raw_weight_nk.T.copy()
        accumulator_scale = input_scale * weight_scale
        bias_acc, bias_stats = quantize_bias(bias, accumulator_scale)
        worst_accumulator = accumulator_bound(raw_weight, bias_acc)

        activation_matrix, valid_mask = im2col_payload(input_payload, layer)
        accumulator = exact_accumulate(
            activation_matrix,
            raw_weight,
            bias_acc,
            layer.in_features,
            valid_mask,
        )
        act_mode = tc.ACT_RELU if layer.activation == "relu" else tc.ACT_NONE
        activation_float = decode_activation_np(
            activation_matrix
        ).astype(np.float64) * input_scale
        activation_float = np.where(valid_mask, activation_float, 0.0)
        effective_weight = effective_weight_np(
            raw_weight_nk
        ).astype(np.float64) * weight_scale
        reference_matrix = np.matmul(
            activation_float, effective_weight.T
        ) + bias
        if act_mode == tc.ACT_RELU:
            reference_matrix = np.maximum(reference_matrix, 0.0)

        output_scale = choose_a5_scale(reference_matrix)
        requant_factor = accumulator_scale / (2.0 * output_scale)
        qmult, qshift = encode_requant_factor(requant_factor)
        output_matrix, output_stats = output_payload_np(
            accumulator, qmult, qshift, act_mode
        )
        ideal_payload = quantize_activation(reference_matrix, output_scale)
        reconstructed = decode_activation_np(
            output_matrix
        ).astype(np.float64) * output_scale
        encoded_factor = qmult / float(1 << qshift)
        stats = {
            "activation": activation_stats(input_float, input_scale),
            "weight": weight_stats,
            "bias": bias_stats,
            "output": output_stats,
            "requant": {
                "target_factor": requant_factor,
                "encoded_factor": encoded_factor,
                "relative_factor_error": abs(
                    encoded_factor - requant_factor
                ) / requant_factor,
                "payload_mismatch": int(np.count_nonzero(
                    output_matrix != ideal_payload
                )),
                "total": int(output_matrix.size),
                "reconstruction_mse": float(np.mean(
                    (reconstructed - reference_matrix) ** 2
                )),
                "mode": "scale",
                "rank": None,
            },
            "accumulator_min": int(np.min(accumulator)),
            "accumulator_max": int(np.max(accumulator)),
            "accumulator_worst_case_bound": worst_accumulator,
            "masked_elements": int(np.count_nonzero(~valid_mask)),
            "matrix_elements": int(valid_mask.size),
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
        batch = input_payload.shape[0]
        tensor_states[layer.output_tensor] = {
            "float": matrix_to_nchw(
                reference_matrix, batch, layer.output_shape
            ),
            "payload": matrix_to_nchw(
                output_matrix, batch, layer.output_shape
            ),
            "scale": output_scale,
        }

    model = dict(source_model)
    host_operations = []
    for operation in source_model.get("host_operations", []):
        if operation["type"] != "add":
            raise ValueError("unsupported Host operator")
        left = tensor_states[operation["inputs"][0]]
        right = tensor_states[operation["inputs"][1]]
        left_values = decode_activation_np(left["payload"]) * left["scale"]
        right_values = decode_activation_np(right["payload"]) * right["scale"]
        output_values = left_values + right_values
        if operation.get("activation", "none") == "relu":
            output_values = np.maximum(output_values, 0.0)
        output_scale = choose_a5_scale(output_values)
        output_payload = quantize_activation(output_values, output_scale)
        tensor_states[operation["output"]] = {
            "float": decode_activation_np(output_payload) * output_scale,
            "payload": output_payload,
            "scale": output_scale,
        }
        compiled_operation = dict(operation)
        compiled_operation["input_scales"] = [left["scale"], right["scale"]]
        compiled_operation["output_scale"] = output_scale
        compiled_operation["stats"] = {
            "activation": activation_stats(output_values, output_scale),
            "minimum": float(np.min(output_values)),
            "maximum": float(np.max(output_values)),
        }
        host_operations.append(compiled_operation)
    model["host_operations"] = host_operations
    return model_path, model, tensors, quantized_layers


def compile_quantized_layers(model_path):
    _path, _model, _tensors, layers = load_model(model_path)
    if all(layer.op_type == "linear" for layer in layers):
        return compile_linear_quantized_layers(model_path)
    if all(layer.op_type == "conv2d" for layer in layers):
        return compile_conv_quantized_layers(model_path)
    raise ValueError("mixed Linear and Conv2D graphs are not implemented")


def layer_has_bias(layer):
    return bool(layer.ir.bias_key or layer.ir.bn_weight_key)


def write_compiled_model(outdir, model_path, model, layers):
    os.makedirs(outdir, exist_ok=True)
    tensor_name = "compiled_model.npz"
    tensor_path = os.path.join(outdir, tensor_name)
    arrays = {}
    entries = []
    for index, layer in enumerate(layers):
        weight_key = "layer_%d.raw_weight" % index
        bias_key = "layer_%d.bias_acc" % index
        arrays[weight_key] = layer.raw_weight.astype(np.uint8)
        arrays[bias_key] = layer.bias_acc.astype(np.int64)
        entries.append({
            "name": layer.ir.name,
            "type": layer.ir.op_type,
            "in_features": layer.ir.in_features,
            "out_features": layer.ir.out_features,
            "input": layer.ir.input_tensor,
            "output": layer.ir.output_tensor,
            "input_shape": list(layer.ir.input_shape),
            "output_shape": list(layer.ir.output_shape),
            "kernel_size": list(layer.ir.kernel_size),
            "stride": list(layer.ir.stride),
            "padding": list(layer.ir.padding),
            "dilation": list(layer.ir.dilation),
            "groups": layer.ir.groups,
            "activation": layer.ir.activation,
            "has_bias": layer_has_bias(layer),
            "raw_weight": weight_key,
            "bias_acc": bias_key,
            "weight_scale": layer.weight_scale,
            "input_scale": layer.input_scale,
            "output_scale": layer.output_scale,
            "qmult": layer.qmult,
            "qshift": layer.qshift,
            "act_mode": layer.act_mode,
            "stats": layer.stats,
        })
    np.savez_compressed(tensor_path, **arrays)

    output_semantics = model.get("output_semantics")
    if output_semantics is None and model.get("format") == LEGACY_MODEL_FORMAT:
        output_semantics = "classification_logits"
    document = {
        "format": COMPILED_FORMAT,
        "name": model.get("name", "model"),
        "source_model": os.path.basename(model_path),
        "input_features": int(model.get("input_features", 0)),
        "input": model.get("input"),
        "output": model.get("output"),
        "output_semantics": output_semantics or "tensor",
        "host_operations": model.get("host_operations", []),
        "target": asdict(TargetSpec()),
        "tensor_file": tensor_name,
        "tensor_sha256": sha256_file(tensor_path),
        "layers": entries,
    }
    compiled_path = os.path.join(outdir, "compiled_model.json")
    with open(compiled_path, "w", newline="\n", encoding="utf-8") as output:
        json.dump(document, output, indent=2)
        output.write("\n")
    return compiled_path


def load_compiled_model(compiled_path):
    compiled_path = os.path.abspath(compiled_path)
    with open(compiled_path, "r", encoding="utf-8") as source:
        document = json.load(source)
    if document.get("format") != COMPILED_FORMAT:
        raise ValueError("unsupported compiled model format")
    if document.get("target") != asdict(TargetSpec()):
        raise ValueError("compiled model target does not match this TPU")

    tensor_path = os.path.join(
        os.path.dirname(compiled_path), document["tensor_file"]
    )
    if sha256_file(tensor_path) != document["tensor_sha256"]:
        raise AssertionError("compiled tensor checksum mismatch")
    with np.load(tensor_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    layers = []
    for entry in document["layers"]:
        ir = LayerIR(
            name=entry["name"],
            in_features=int(entry["in_features"]),
            out_features=int(entry["out_features"]),
            weight_key=entry["raw_weight"],
            bias_key="compiled_bias" if entry["has_bias"] else "",
            activation=entry["activation"],
            op_type=entry.get("type", "linear"),
            input_tensor=entry.get("input", ""),
            output_tensor=entry.get("output", ""),
            input_shape=tuple(entry.get("input_shape", [])),
            output_shape=tuple(entry.get("output_shape", [])),
            kernel_size=tuple(entry.get("kernel_size", [1, 1])),
            stride=tuple(entry.get("stride", [1, 1])),
            padding=tuple(entry.get("padding", [0, 0])),
            dilation=tuple(entry.get("dilation", [1, 1])),
            groups=int(entry.get("groups", 1)),
        )
        layers.append(QuantizedLayer(
            ir=ir,
            raw_weight=np.asarray(arrays[entry["raw_weight"]], dtype=np.uint8),
            bias_acc=np.asarray(arrays[entry["bias_acc"]], dtype=np.int64),
            weight_scale=float(entry["weight_scale"]),
            input_scale=float(entry["input_scale"]),
            output_scale=float(entry["output_scale"]),
            qmult=int(entry["qmult"]),
            qshift=int(entry["qshift"]),
            act_mode=int(entry["act_mode"]),
            stats=entry["stats"],
        ))
    model = {
        "name": document["name"],
        "input_features": int(document["input_features"]),
        "input": document.get("input"),
        "output": document.get("output"),
        "output_semantics": document.get("output_semantics", "tensor"),
        "host_operations": document.get("host_operations", []),
        "layers": document["layers"],
    }
    return compiled_path, model, layers


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


def build_linear_workload(model_path, rtl_inputs, rtl_labels=None, compiled=None):
    if compiled is None:
        model_path, model, _tensors, layers = compile_quantized_layers(model_path)
    else:
        model_path, model, layers = compiled
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
            "name": model.get("name", "model") + "_rtl",
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


def emit_conv_layer(layer, input_payload, commands, weight_rows,
                    activation_rows, golden_rows):
    ir = layer.ir
    target = TargetSpec()
    activation, valid_mask = im2col_payload(input_payload, ir)
    m_dim = int(activation.shape[0])
    if m_dim > tc.M_MAX:
        raise ValueError(
            "Conv2D layer %s has M=%d; split batches before RTL emit" %
            (ir.name, m_dim)
        )
    mt_total = ceil_div(m_dim, target.array_size)
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

    m_pad = mt_total * target.array_size
    k_pad = kt_total * target.array_size
    n_pad = nt_total * target.array_size
    padded_activation = pad_payload(activation, m_pad, k_pad)
    padded_mask = np.zeros((m_pad, k_pad), dtype=bool)
    padded_mask[:m_dim, :ir.in_features] = valid_mask
    padded_weight = pad_weight(layer.raw_weight, k_pad, n_pad)
    padded_bias = np.zeros(n_pad, dtype=np.int64)
    padded_bias[:ir.out_features] = layer.bias_acc

    accumulator = exact_accumulate(
        padded_activation,
        padded_weight,
        padded_bias,
        ir.in_features,
        padded_mask,
    )
    padded_output, rtl_output_stats = output_payload_np(
        accumulator, layer.qmult, layer.qshift, layer.act_mode
    )
    output = padded_output[:m_dim, :ir.out_features].copy()

    layer_start = len(commands)
    masked_loads = 0
    activation_loads = 0
    for block in weight_blocks:
        nt_begin = block["nt_begin"]
        nt_count = block["nt_count"]
        for local_nt in range(nt_count):
            nt = nt_begin + local_nt
            for kt in range(kt_total):
                w_slot = kt * nt_count + local_nt
                command = tc.make_command(
                    tc.CMD_OP_LOAD_W, kt=kt, nt=nt, w_slot=w_slot
                )
                commands.append(command)
                weight_rows.extend(pack_weight_tile_np(padded_weight, kt, nt))

        for local_nt in range(nt_count):
            nt = nt_begin + local_nt
            if layer_has_bias(layer):
                bias = padded_bias[
                    nt * target.array_size:(nt + 1) * target.array_size
                ]
                commands.append(tc.make_command(tc.CMD_OP_LOAD_BIAS, nt=nt))
                weight_rows.extend(tg.pack_bias([int(value) for value in bias]))

            for mt in range(mt_total):
                for kt in range(kt_total):
                    k_valid = min(
                        target.array_size,
                        ir.in_features - kt * target.array_size,
                    )
                    row_begin = mt * target.array_size
                    column_begin = kt * target.array_size
                    tile_mask = padded_mask[
                        row_begin:row_begin + target.array_size,
                        column_begin:column_begin + target.array_size,
                    ]
                    a_mask_en = not np.all(tile_mask[:, :k_valid])
                    commands.append(tc.make_command(
                        tc.CMD_OP_LOAD_A,
                        mt=mt,
                        kt=kt,
                        src_slot=0,
                        a_mask_en=a_mask_en,
                    ))
                    activation_rows.extend(pack_activation_tile_np(
                        padded_activation, mt, kt
                    ))
                    if a_mask_en:
                        activation_rows.extend(tg.pack_activation_mask(
                            tile_mask.tolist()
                        ))
                        masked_loads += 1
                    activation_loads += 1

                    w_slot = kt * nt_count + local_nt
                    commands.append(tc.make_command(
                        tc.CMD_OP_PRELOAD_W,
                        mt=mt,
                        kt=kt,
                        nt=nt,
                        w_slot=w_slot,
                    ))
                    commands.append(tc.make_command(
                        tc.CMD_OP_GEMM,
                        mt=mt,
                        kt=kt,
                        nt=nt,
                        w_slot=w_slot,
                        src_slot=0,
                        acc_init=(kt == 0),
                        acc_final=(kt == kt_total - 1),
                        k_valid=k_valid,
                        bias_en=(kt == 0 and layer_has_bias(layer)),
                    ))

                commands.append(tc.make_command(
                    tc.CMD_OP_STORE_C,
                    qmult=layer.qmult,
                    qshift=layer.qshift,
                    mt=mt,
                    kt=kt_total - 1,
                    nt=nt,
                    dst_slot=1,
                    act_mode=layer.act_mode,
                ))
                golden_rows.extend(pack_output_tile_np(padded_output, mt, nt))

    batch = input_payload.shape[0]
    output_tensor = matrix_to_nchw(output, batch, ir.output_shape)
    manifest = {
        "name": ir.name,
        "type": ir.op_type,
        "input_tensor": ir.input_tensor,
        "output_tensor": ir.output_tensor,
        "input_shape": [batch] + list(ir.input_shape),
        "output_shape": [batch] + list(ir.output_shape),
        "shape": {"M": m_dim, "K": ir.in_features, "N": ir.out_features},
        "tiles": {"MT": mt_total, "KT": kt_total, "NT": nt_total},
        "kernel_size": list(ir.kernel_size),
        "stride": list(ir.stride),
        "padding": list(ir.padding),
        "activation": ir.activation,
        "act_mode": layer.act_mode,
        "qmult": layer.qmult,
        "qshift": layer.qshift,
        "input_scale": layer.input_scale,
        "weight_scale": layer.weight_scale,
        "output_scale": layer.output_scale,
        "weight_tiles": kt_total * nt_total,
        "weight_slots": kt_total * nt_block_capacity,
        "weight_blocks": weight_blocks,
        "activation_loads": activation_loads,
        "masked_loads": masked_loads,
        "command_begin": layer_start,
        "command_count": len(commands) - layer_start,
        "stats": dict(layer.stats, rtl_output=rtl_output_stats),
    }
    return output_tensor, manifest


def execute_host_operations(model, tensor_states):
    manifests = []
    for operation in model.get("host_operations", []):
        left = tensor_states[operation["inputs"][0]]
        right = tensor_states[operation["inputs"][1]]
        left_values = decode_activation_np(left["payload"]) * left["scale"]
        right_values = decode_activation_np(right["payload"]) * right["scale"]
        output_values = left_values + right_values
        if operation.get("activation", "none") == "relu":
            output_values = np.maximum(output_values, 0.0)
        output_scale = float(
            operation.get("output_scale", choose_a5_scale(output_values))
        )
        output_payload = quantize_activation(output_values, output_scale)
        tensor_states[operation["output"]] = {
            "payload": output_payload,
            "scale": output_scale,
            "float": decode_activation_np(output_payload) * output_scale,
        }
        manifests.append({
            "name": operation["name"],
            "type": operation["type"],
            "inputs": list(operation["inputs"]),
            "output": operation["output"],
            "activation": operation.get("activation", "none"),
            "input_scales": [left["scale"], right["scale"]],
            "output_scale": output_scale,
            "output_shape": list(output_payload.shape),
            "payload_sha256": hashlib.sha256(
                output_payload.tobytes()
            ).hexdigest(),
        })
    return manifests


def build_conv_workload(model_path, rtl_inputs, rtl_labels=None, compiled=None,
                        rtl_reference_outputs=None):
    if compiled is None:
        model_path, model, _tensors, layers = compile_quantized_layers(model_path)
    else:
        model_path, model, layers = compiled
    rtl_inputs = np.asarray(rtl_inputs, dtype=np.float64)
    root_shape = tuple(int(value) for value in model["input"]["shape"][1:])
    if rtl_inputs.ndim != 4 or tuple(rtl_inputs.shape[1:]) != root_shape:
        raise ValueError("RTL Conv2D input shape mismatch")

    root_name = model["input"]["name"]
    tensor_states = {
        root_name: {
            "float": rtl_inputs,
            "payload": None,
            "scale": None,
        }
    }
    commands = []
    weight_rows = []
    activation_rows = []
    golden_rows = []
    layer_manifest = []

    for layer in layers:
        state = tensor_states[layer.ir.input_tensor]
        if state["payload"] is None or not math.isclose(
                state["scale"] or layer.input_scale,
                layer.input_scale,
                rel_tol=1.0e-12,
                abs_tol=0.0):
            input_payload = quantize_activation(
                state["float"], layer.input_scale
            )
        else:
            input_payload = state["payload"]
        output_payload, manifest = emit_conv_layer(
            layer,
            input_payload,
            commands,
            weight_rows,
            activation_rows,
            golden_rows,
        )
        tensor_states[layer.ir.output_tensor] = {
            "payload": output_payload,
            "scale": layer.output_scale,
            "float": decode_activation_np(output_payload) * layer.output_scale,
        }
        layer_manifest.append(manifest)

    host_manifest = execute_host_operations(model, tensor_states)
    output_name = model.get("output", {}).get("name")
    final_state = tensor_states.get(output_name) if output_name else None
    reference_metrics = None
    if rtl_reference_outputs is not None and final_state is not None:
        reference = np.asarray(rtl_reference_outputs, dtype=np.float64)
        if reference.shape != final_state["float"].shape:
            raise ValueError("RTL reference output shape mismatch")
        difference = final_state["float"] - reference
        reference_metrics = {
            "mse": float(np.mean(difference * difference)),
            "maximum_absolute_error": float(np.max(np.abs(difference))),
        }

    replay_rows = tg.replay_commands(commands, weight_rows, activation_rows)
    if replay_rows != golden_rows:
        mismatch = next(
            index
            for index, pair in enumerate(zip(replay_rows, golden_rows))
            if pair[0] != pair[1]
        )
        raise AssertionError(
            "Conv2D golden differs from command replay at row %d" % mismatch
        )

    first_shape = layer_manifest[0]["shape"]
    manifest = {
        "format": BUNDLE_FORMAT,
        "name": model.get("name", "conv_model"),
        "source_model": os.path.basename(model_path),
        "target": asdict(TargetSpec()),
        "test": {
            "name": model.get("name", "conv_model") + "_rtl",
            "M": first_shape["M"],
            "K": first_shape["K"],
            "N": first_shape["N"],
            "command_count": len(commands),
            "weight_rows": len(weight_rows),
            "activation_rows": len(activation_rows),
            "golden_rows": len(golden_rows),
        },
        "layers": layer_manifest,
        "host_operations": host_manifest,
        "reference_metrics": reference_metrics,
        "labels": None if rtl_labels is None else
            np.asarray(rtl_labels, dtype=np.int64).tolist(),
        "predictions": [],
        "rtl_batch_accuracy": None,
        "files": {},
    }
    return WorkloadBundle(
        commands=commands,
        weight_rows=weight_rows,
        activation_rows=activation_rows,
        golden_rows=golden_rows,
        manifest=manifest,
    )


def build_workload(model_path, rtl_inputs, rtl_labels=None, compiled=None,
                   rtl_reference_outputs=None):
    if compiled is None:
        probe = compile_quantized_layers(model_path)
        compiled = (probe[0], probe[1], probe[3])
    if all(layer.ir.op_type == "linear" for layer in compiled[2]):
        return build_linear_workload(
            model_path, rtl_inputs, rtl_labels, compiled=compiled
        )
    return build_conv_workload(
        model_path,
        rtl_inputs,
        rtl_labels,
        compiled=compiled,
        rtl_reference_outputs=rtl_reference_outputs,
    )


def write_test_suite(path, bundle):
    test = bundle.manifest["test"]
    with open(path, "w", newline="\n") as output:
        output.write("// Auto-generated by compiler/tpu_compiler.py.\n\n")
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
            "'{\"%s\"};\n" % test["name"]
        )


def write_summary(path, bundle):
    manifest = bundle.manifest
    test = manifest["test"]
    with open(path, "w", newline="\n") as output:
        output.write("TPU compiled model workload\n")
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
        if manifest.get("host_operations"):
            output.write("host operators : %d\n" %
                         len(manifest["host_operations"]))
        if manifest.get("predictions"):
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


def compile_command(args):
    model_path, model, _tensors, layers = compile_quantized_layers(args.model)
    outdir = os.path.abspath(
        args.outdir or os.path.join(TESTBED_ROOT, "compiler", "build")
    )
    compiled_path = write_compiled_model(outdir, model_path, model, layers)
    load_compiled_model(compiled_path)
    print("[PASS] compiled model checksum and target contract verified")
    print("compiled %d-layer model -> %s" % (len(layers), compiled_path))


def emit_test_command(args):
    compiled = load_compiled_model(args.compiled)
    _input_path, _input_model, tensors, _input_layers = load_model(args.inputs)
    if args.input_key not in tensors:
        raise ValueError("tensor package has no %s" % args.input_key)
    rtl_inputs = tensors[args.input_key][args.sample_offset:
                                         args.sample_offset + args.batch_size]
    labels = None
    if args.label_key and args.label_key in tensors:
        labels = tensors[args.label_key][args.sample_offset:
                                          args.sample_offset + args.batch_size]
    if rtl_inputs.shape[0] != args.batch_size:
        raise ValueError("requested RTL batch is not available")
    rtl_reference_outputs = None
    if "rtl_reference_outputs" in tensors:
        rtl_reference_outputs = tensors["rtl_reference_outputs"][
            args.sample_offset:args.sample_offset + args.batch_size
        ]
    bundle = build_workload(
        args.compiled,
        rtl_inputs,
        labels,
        compiled=compiled,
        rtl_reference_outputs=rtl_reference_outputs,
    )
    outdir = os.path.abspath(
        args.outdir or os.path.join(TESTBED_ROOT, "pattern")
    )
    manifest_path = write_bundle(outdir, bundle)
    replay_bundle(manifest_path)
    print("[PASS] compiled model matches command replay")
    print("[PASS] bundle replay matches generated golden")
    print("emitted %d-layer RTL test -> %s" %
          (len(compiled[2]), outdir))
    print("commands=%d rows=%d/%d/%d" % (
        len(bundle.commands), len(bundle.weight_rows),
        len(bundle.activation_rows), len(bundle.golden_rows),
    ))


def replay_command(args):
    manifest = replay_bundle(args.bundle)
    print("[PASS] %s bundle replay passed (%d commands)" %
          (manifest["name"], manifest["test"]["command_count"]))


def evaluate_command(args):
    import tpu_reference

    tpu_reference.evaluate_command(args)


def ablate_command(args):
    import tpu_reference

    tpu_reference.ablate_command(args)


def build_parser():
    parser = argparse.ArgumentParser(description="Type-2 TPU model compiler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser(
        "compile", help="compile and quantize one portable model"
    )
    compile_parser.add_argument("model", help="model JSON file")
    compile_parser.add_argument("--outdir", default=None)
    compile_parser.set_defaults(function=compile_command)

    emit_parser = subparsers.add_parser(
        "emit-test", help="emit one RTL test from a compiled model"
    )
    emit_parser.add_argument("compiled", help="compiled_model.json file")
    emit_parser.add_argument(
        "--inputs", required=True, help="model package containing test inputs"
    )
    emit_parser.add_argument("--input-key", default="rtl_inputs")
    emit_parser.add_argument("--label-key", default="rtl_labels")
    emit_parser.add_argument("--outdir", default=None)
    emit_parser.add_argument("--batch-size", type=int, default=32)
    emit_parser.add_argument("--sample-offset", type=int, default=0)
    emit_parser.set_defaults(function=emit_test_command)

    replay_parser = subparsers.add_parser("replay", help="replay one bundle")
    replay_parser.add_argument("bundle", help="bundle directory or manifest.json")
    replay_parser.set_defaults(function=replay_command)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate a model with the bit-true reference"
    )
    evaluate_parser.add_argument("model", help="model JSON file")
    evaluate_parser.add_argument("--batch-size", type=int, default=256)
    evaluate_parser.add_argument("--limit", type=int, default=0)
    evaluate_parser.set_defaults(function=evaluate_command)

    ablate_parser = subparsers.add_parser(
        "ablate", help="compare quantization and bit-true stages"
    )
    ablate_parser.add_argument("model", help="model JSON file")
    ablate_parser.add_argument("--batch-size", type=int, default=512)
    ablate_parser.add_argument("--limit", type=int, default=0)
    ablate_parser.add_argument("--include-bittrue", action="store_true")
    ablate_parser.set_defaults(function=ablate_command)
    return parser


def main():
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
