#!/usr/bin/env python3
"""Reference inference and quantization analysis for the TPU compiler."""

import numpy as np

import tpu_compiler as compiler


tc = compiler.tc
tg = compiler.tg


def hardware_inference(model_path, inputs, batch_size=256):
    _, _, _, layers = compiler.compile_quantized_layers(model_path)
    inputs = np.asarray(inputs, dtype=np.float64)
    predictions = []
    for base in range(0, inputs.shape[0], batch_size):
        payload = compiler.quantize_activation(
            inputs[base:base + batch_size], layers[0].input_scale
        )
        for layer in layers:
            accumulator = compiler.exact_accumulate(
                payload, layer.raw_weight, layer.bias_acc, layer.ir.in_features
            )
            payload, _ = compiler.output_payload_np(
                accumulator, layer.qmult, layer.qshift, layer.act_mode
            )
        predictions.extend(np.argmax(
            compiler.signed_payload_np(payload), axis=1
        ).tolist())
    return np.asarray(predictions, dtype=np.int64)


def fake_weight(weight, mode):
    weight = np.asarray(weight, dtype=np.float64)
    if mode == "fp32":
        return weight.copy()

    raw, effective_scale, stats = compiler.quantize_weight(weight)
    if mode == "int8":
        signed_raw = np.where(
            raw.astype(np.int64) >= 128,
            raw.astype(np.int64) - 256,
            raw.astype(np.int64),
        )
        return signed_raw.astype(np.float64) * stats["raw_scale"]
    if mode == "type2":
        return compiler.effective_weight_np(
            raw
        ).astype(np.float64) * effective_scale
    raise ValueError("unsupported fake weight mode %s" % mode)


def fake_quant_activation(values, scale, codebook, relu):
    values = np.asarray(values, dtype=np.float64)
    if scale <= 0.0:
        raise ValueError("fake quant scale must be positive")

    if codebook == "int4":
        quantized = compiler.round_half_up(values / scale)
        minimum = 0 if relu else -8
        quantized = np.clip(quantized, minimum, 7)
        return quantized * scale
    if codebook == "fixed_lsb":
        quantized = compiler.round_half_up((values / scale - 1.0) / 2.0)
        minimum = 0 if relu else -8
        quantized = np.clip(quantized, minimum, 7)
        return (quantized * 2.0 + 1.0) * scale
    raise ValueError("unsupported activation codebook %s" % codebook)


def choose_fake_scale(values, codebook, relu):
    values = np.asarray(values, dtype=np.float64)
    if codebook == "fixed_lsb":
        return compiler.choose_a5_scale(values)

    magnitude = values if relu else np.abs(values)
    magnitude = magnitude[np.isfinite(magnitude)]
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
        scale = clip / 7.0
        reconstructed = fake_quant_activation(values, scale, codebook, relu)
        error = float(np.mean((reconstructed - values) ** 2))
        if best_error is None or error < best_error:
            best_scale = scale
            best_error = error
    return best_scale


def prepare_fake_model(model_path, weight_mode, activation_codebook=None):
    _, model, tensors, layers = compiler.load_model(model_path)
    calibration = np.asarray(tensors["calibration_inputs"], dtype=np.float64)
    fake_layers = []
    input_relu = False
    input_scale = choose_fake_scale(
        calibration, activation_codebook, input_relu
    ) if activation_codebook else None

    for layer_index, ir in enumerate(layers):
        weight = np.asarray(tensors[ir.weight_key], dtype=np.float64)
        bias = np.asarray(tensors[ir.bias_key], dtype=np.float64) \
            if ir.bias_key else np.zeros(ir.out_features, dtype=np.float64)
        reduced_weight = fake_weight(weight, weight_mode)
        layer_input = fake_quant_activation(
            calibration, input_scale, activation_codebook, input_relu
        ) if activation_codebook else calibration
        output = np.matmul(layer_input, reduced_weight.T) + bias
        output_relu = ir.activation == "relu"
        if output_relu:
            output = np.maximum(output, 0.0)

        next_scale = None
        if activation_codebook and layer_index != len(layers) - 1:
            next_scale = choose_fake_scale(
                output, activation_codebook, output_relu
            )
        fake_layers.append({
            "ir": ir,
            "weight": reduced_weight,
            "bias": bias,
            "input_scale": input_scale,
            "input_relu": input_relu,
            "weight_mse": float(np.mean((reduced_weight - weight) ** 2)),
        })
        calibration = output
        input_scale = next_scale
        input_relu = output_relu
    return model, tensors, fake_layers


def fake_inference(model_path, inputs, weight_mode, activation_codebook=None,
                   batch_size=512):
    _, _, layers = prepare_fake_model(
        model_path, weight_mode, activation_codebook
    )
    inputs = np.asarray(inputs, dtype=np.float64)
    predictions = []
    for base in range(0, inputs.shape[0], batch_size):
        values = inputs[base:base + batch_size]
        for layer in layers:
            if activation_codebook:
                values = fake_quant_activation(
                    values,
                    layer["input_scale"],
                    activation_codebook,
                    layer["input_relu"],
                )
            values = np.matmul(values, layer["weight"].T) + layer["bias"]
            if layer["ir"].activation == "relu":
                values = np.maximum(values, 0.0)
        predictions.extend(np.argmax(values, axis=1).tolist())
    return np.asarray(predictions, dtype=np.int64), layers


def compiler_scale_float_inference(model_path, inputs, batch_size=512,
                                   compiled=None):
    if compiled is None:
        _, _, tensors, layers = compiler.compile_quantized_layers(model_path)
    else:
        tensors, layers = compiled
    inputs = np.asarray(inputs, dtype=np.float64)
    predictions = []
    for base in range(0, inputs.shape[0], batch_size):
        values = inputs[base:base + batch_size]
        for layer in layers:
            source = layer.ir
            payload = compiler.quantize_activation(values, layer.input_scale)
            activation = compiler.decode_activation_np(
                payload
            ) * layer.input_scale
            weight = compiler.effective_weight_np(
                layer.raw_weight.T
            ).astype(np.float64) * layer.weight_scale
            bias = np.asarray(tensors[source.bias_key], dtype=np.float64) \
                if source.bias_key else np.zeros(source.out_features)
            values = np.matmul(activation, weight.T) + bias
            if source.activation == "relu":
                values = np.maximum(values, 0.0)
        predictions.extend(np.argmax(values, axis=1).tolist())
    return np.asarray(predictions, dtype=np.int64)


def progressive_accumulate(payload, raw_weight, bias_acc, k_dim,
                           psum_wrap=False, acc_wrap=False):
    activation = compiler.decode_activation_np(payload)
    weight = compiler.effective_weight_np(raw_weight)
    bias_acc = np.asarray(bias_acc, dtype=np.int64)

    if not psum_wrap:
        accumulator = np.matmul(
            activation[:, :k_dim], weight[:k_dim, :]
        ).astype(np.int64)
        return accumulator + bias_acc

    accumulator = None
    for k_base in range(0, k_dim, tc.ARRAY_S):
        k_valid = min(tc.ARRAY_S, k_dim - k_base)
        psum = np.matmul(
            activation[:, k_base:k_base + k_valid],
            weight[k_base:k_base + k_valid, :],
        )
        psum = compiler.wrap_signed_np(psum, tg.PSUM_W)
        if accumulator is None:
            accumulator = psum + bias_acc
        else:
            accumulator = accumulator + psum
        if acc_wrap:
            accumulator = compiler.wrap_signed_np(accumulator, tg.ACC_W)
    return accumulator.astype(np.int64)


def requant_no_wrap(accumulator, qmult, qshift, act_mode):
    values = np.asarray(accumulator, dtype=np.int64)
    if act_mode == tc.ACT_RELU:
        values = np.maximum(values, 0)
    quantized = (values * int(qmult)) >> int(qshift)
    minimum = 0 if act_mode == tc.ACT_RELU else tg.SAT_MIN
    quantized = np.clip(quantized, minimum, tg.SAT_MAX).astype(np.int64)
    return (quantized & 0xF).astype(np.uint8)


def progressive_integer_inference(model_path, inputs, mode, batch_size=512,
                                  compiled=None):
    if compiled is None:
        _, _, tensors, layers = compiler.compile_quantized_layers(model_path)
    else:
        tensors, layers = compiled
    inputs = np.asarray(inputs, dtype=np.float64)
    predictions = []
    for base in range(0, inputs.shape[0], batch_size):
        values = inputs[base:base + batch_size]
        payload = compiler.quantize_activation(values, layers[0].input_scale)
        for layer_index, layer in enumerate(layers):
            source = layer.ir
            bias_float = np.asarray(tensors[source.bias_key], dtype=np.float64) \
                if source.bias_key else np.zeros(source.out_features)
            use_bias29 = mode != "int64_fp_bias"
            psum_wrap = mode in ("psum21", "bittrue")
            acc_wrap = mode == "bittrue"
            bias_acc = layer.bias_acc if use_bias29 else \
                np.zeros(source.out_features, dtype=np.int64)
            accumulator = progressive_accumulate(
                payload,
                layer.raw_weight,
                bias_acc,
                source.in_features,
                psum_wrap=psum_wrap,
                acc_wrap=acc_wrap,
            )

            use_integer_requant = mode in ("requant", "psum21", "bittrue")
            if use_integer_requant:
                payload = compiler.output_payload_np(
                    accumulator, layer.qmult, layer.qshift, layer.act_mode
                )[0] if acc_wrap else requant_no_wrap(
                    accumulator, layer.qmult, layer.qshift, layer.act_mode
                )
                if layer_index == len(layers) - 1:
                    values = compiler.signed_payload_np(payload)
                continue

            accumulator_scale = layer.input_scale * layer.weight_scale
            values = accumulator.astype(np.float64) * accumulator_scale
            if not use_bias29:
                values = values + bias_float
            if source.activation == "relu":
                values = np.maximum(values, 0.0)
            if layer_index != len(layers) - 1:
                next_scale = layers[layer_index + 1].input_scale
                payload = compiler.quantize_activation(values, next_scale)
        predictions.extend(np.argmax(values, axis=1).tolist())
    return np.asarray(predictions, dtype=np.int64)


def bittrue_breakdown(model_path, inputs, labels, batch_size=512,
                      compiled=None):
    labels = np.asarray(labels, dtype=np.int64)
    if compiled is None:
        _, _, tensors, layers = compiler.compile_quantized_layers(model_path)
        compiled = tensors, layers
    stages = []
    predictions = compiler_scale_float_inference(
        model_path, inputs, batch_size, compiled
    )
    stages.append({
        "name": "A5 compiler-scale FP MAC",
        "accuracy": float(np.mean(predictions == labels)),
        "weight_mse": None,
        "activation_scales": None,
    })
    for name, mode in (
        ("A5 + INT64 MAC + FP32 bias", "int64_fp_bias"),
        ("+ Bias29", "bias29"),
        ("+ Integer QMULT/QSHIFT", "requant"),
        ("+ PSUM21 wrap", "psum21"),
        ("+ ACC29 wrap (bit-true)", "bittrue"),
    ):
        predictions = progressive_integer_inference(
            model_path, inputs, mode, batch_size, compiled
        )
        stages.append({
            "name": name,
            "accuracy": float(np.mean(predictions == labels)),
            "weight_mse": None,
            "activation_scales": None,
        })
    return stages


def quantization_ablation(model_path, inputs, labels, batch_size=512,
                          include_bittrue=False, compiled=None):
    stages = [
        ("FP32", "fp32", None),
        ("INT8-W", "int8", None),
        ("Type2-W", "type2", None),
        ("Type2-W + INT4-A", "type2", "int4"),
        ("Type2-W + A5 FixedLSB", "type2", "fixed_lsb"),
    ]
    labels = np.asarray(labels, dtype=np.int64)
    results = []
    for name, weight_mode, codebook in stages:
        predictions, layers = fake_inference(
            model_path, inputs, weight_mode, codebook, batch_size
        )
        results.append({
            "name": name,
            "accuracy": float(np.mean(predictions == labels)),
            "weight_mse": [layer["weight_mse"] for layer in layers],
            "activation_scales": [layer["input_scale"] for layer in layers],
        })
    if include_bittrue:
        results.extend(bittrue_breakdown(
            model_path, inputs, labels, batch_size, compiled
        ))
    return results


def evaluate_command(args):
    _, _, tensors, _ = compiler.load_model(args.model)
    if "test_inputs" not in tensors or "test_labels" not in tensors:
        raise ValueError("tensor package has no test dataset")
    inputs = tensors["test_inputs"]
    labels = tensors["test_labels"].astype(np.int64)
    if args.limit:
        inputs = inputs[:args.limit]
        labels = labels[:args.limit]
    predictions = hardware_inference(args.model, inputs, args.batch_size)
    accuracy = float(np.mean(predictions == labels))
    print("hardware-aware accuracy: %.2f%% (%d/%d)" %
          (accuracy * 100.0, int(np.sum(predictions == labels)), len(labels)))


def ablate_command(args):
    _, _, tensors, _ = compiler.load_model(args.model)
    if "test_inputs" not in tensors or "test_labels" not in tensors:
        raise ValueError("tensor package has no test dataset")
    inputs = tensors["test_inputs"]
    labels = tensors["test_labels"].astype(np.int64)
    if args.limit:
        inputs = inputs[:args.limit]
        labels = labels[:args.limit]
    compiled = None
    calibrated_layers = []
    if args.include_bittrue:
        _, _, compiled_tensors, calibrated_layers = \
            compiler.compile_quantized_layers(args.model)
        compiled = compiled_tensors, calibrated_layers
    results = quantization_ablation(
        args.model,
        inputs,
        labels,
        batch_size=args.batch_size,
        include_bittrue=args.include_bittrue,
        compiled=compiled,
    )

    print("Quantization ablation (%d samples)" % len(labels))
    print("=" * 52)
    for result in results:
        print("%-28s %6.2f%%" %
              (result["name"], result["accuracy"] * 100.0))
    print("=" * 52)
    print("Non-bit-true stages use FP64 MAC and FP32 bias without width wrapping.")
    if calibrated_layers:
        print("\nCompiler calibration")
        print("=" * 78)
        for layer in calibrated_layers:
            requant = layer.stats["requant"]
            mismatch = 100.0 * requant["payload_mismatch"] / requant["total"]
            print(
                "%-8s in=%-10.4g out=%-10.4g q=%-6d/2^%-2d mismatch=%7.4f%%" %
                (layer.ir.name, layer.input_scale, layer.output_scale,
                 layer.qmult, layer.qshift, mismatch)
            )
            if requant["rank"]:
                rank = requant["rank"]
                agreement = 100.0 * rank["agreement"] / rank["total"]
                print(
                    "         rank=%7.4f%% ties=%-4d candidates=%d" %
                    (agreement, rank["tie_rows"], rank["candidate_count"])
                )
        print("=" * 78)
