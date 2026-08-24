#!/usr/bin/env python3
"""Self-check the portable TPU compiler flow."""

import json
import os
import sys
import tempfile

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tpu_compiler as compiler  # noqa: E402
import tpu_reference as reference  # noqa: E402
import pytorch_exporter as exporter  # noqa: E402
from models import mnist_mlp  # noqa: E402
from models import mnist_resnet18  # noqa: E402


def write_demo_model(root):
    rng = np.random.default_rng(310)
    layers = [("fc1", 784, 512, "relu"),
              ("fc2", 512, 256, "relu"),
              ("fc3", 256, 10, "none")]
    tensors = {
        "calibration_inputs": rng.normal(0.0, 0.8, (64, 784)).astype(np.float32),
        "rtl_inputs": rng.normal(0.0, 0.8, (32, 784)).astype(np.float32),
        "rtl_labels": np.arange(32, dtype=np.int64) % 10,
    }
    entries = []
    for name, k_dim, n_dim, activation in layers:
        tensors[name + ".weight"] = rng.normal(
            0.0, 0.04, (n_dim, k_dim)
        ).astype(np.float32)
        tensors[name + ".bias"] = rng.normal(
            0.0, 0.02, n_dim
        ).astype(np.float32)
        entries.append({
            "name": name,
            "type": "linear",
            "in_features": k_dim,
            "out_features": n_dim,
            "weight": name + ".weight",
            "bias": name + ".bias",
            "activation": activation,
        })

    tensor_path = os.path.join(root, "demo.npz")
    np.savez_compressed(tensor_path, **tensors)
    model_path = os.path.join(root, "demo.json")
    with open(model_path, "w", newline="\n", encoding="utf-8") as output:
        json.dump({
            "format": compiler.MODEL_FORMAT,
            "name": "compiler_self_check_mnist_mlp",
            "input_features": 784,
            "output_semantics": "classification_logits",
            "tensor_file": "demo.npz",
            "layers": entries,
        }, output, indent=2)
        output.write("\n")
    return model_path


def write_demo_conv_model(root):
    rng = np.random.default_rng(311)
    tensors = {
        "calibration_inputs": rng.normal(
            0.0, 0.8, (2, 3, 5, 5)
        ).astype(np.float32),
        "rtl_inputs": rng.normal(0.0, 0.8, (1, 3, 5, 5)).astype(np.float32),
    }
    layer_specs = [
        ("main_conv1", "input", "main_relu1", 3, 4, 3, 2, 1, "relu"),
        ("main_conv2", "main_relu1", "main_output", 4, 4, 3, 1, 1, "none"),
        ("shortcut_conv", "input", "shortcut_output", 3, 4, 1, 2, 0, "none"),
    ]
    entries = []
    for name, source, output, in_channels, out_channels, kernel, stride, \
            padding, activation in layer_specs:
        prefix = name
        tensors[prefix + ".weight"] = rng.normal(
            0.0,
            0.08,
            (out_channels, in_channels, kernel, kernel),
        ).astype(np.float32)
        tensors[prefix + ".bn.weight"] = rng.uniform(
            0.8, 1.2, out_channels
        ).astype(np.float32)
        tensors[prefix + ".bn.bias"] = rng.normal(
            0.0, 0.03, out_channels
        ).astype(np.float32)
        tensors[prefix + ".bn.running_mean"] = rng.normal(
            0.0, 0.04, out_channels
        ).astype(np.float32)
        tensors[prefix + ".bn.running_var"] = rng.uniform(
            0.7, 1.3, out_channels
        ).astype(np.float32)
        input_shape = [in_channels, 5, 5] if source == "input" else [4, 3, 3]
        output_shape = [out_channels, 3, 3]
        entries.append({
            "name": name,
            "type": "conv2d",
            "input": source,
            "output": output,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": [kernel, kernel],
            "stride": [stride, stride],
            "padding": [padding, padding],
            "dilation": [1, 1],
            "groups": 1,
            "weight": prefix + ".weight",
            "bias": "",
            "batch_norm": {
                "weight": prefix + ".bn.weight",
                "bias": prefix + ".bn.bias",
                "running_mean": prefix + ".bn.running_mean",
                "running_var": prefix + ".bn.running_var",
                "eps": 1.0e-5,
            },
            "activation": activation,
        })

    tensor_path = os.path.join(root, "demo_conv.npz")
    np.savez_compressed(tensor_path, **tensors)
    model_path = os.path.join(root, "demo_conv.json")
    with open(model_path, "w", newline="\n", encoding="utf-8") as output:
        json.dump({
            "format": compiler.MODEL_FORMAT,
            "name": "compiler_self_check_resnet_block",
            "input": {
                "name": "input",
                "shape": [None, 3, 5, 5],
                "layout": "NCHW",
            },
            "output": {
                "name": "block_output",
                "shape": [None, 4, 3, 3],
                "layout": "NCHW",
            },
            "output_semantics": "feature_map",
            "tensor_file": "demo_conv.npz",
            "layers": entries,
            "host_operations": [{
                "name": "residual_add_relu",
                "type": "add",
                "inputs": ["main_output", "shortcut_output"],
                "output": "block_output",
                "activation": "relu",
            }],
        }, output, indent=2)
        output.write("\n")
    return model_path


def naive_conv_payload(input_payload, layer):
    ir = layer.ir
    batch, _channels, height, width = input_payload.shape
    output = np.zeros((batch,) + ir.output_shape, dtype=np.uint8)
    for batch_index in range(batch):
        for output_channel in range(ir.out_features):
            for output_row in range(ir.output_shape[1]):
                for output_column in range(ir.output_shape[2]):
                    accumulator = int(layer.bias_acc[output_channel])
                    k_index = 0
                    partial_sum = 0
                    for input_channel in range(ir.input_shape[0]):
                        for kernel_row in range(ir.kernel_size[0]):
                            input_row = (
                                output_row * ir.stride[0] - ir.padding[0] +
                                kernel_row * ir.dilation[0]
                            )
                            for kernel_column in range(ir.kernel_size[1]):
                                input_column = (
                                    output_column * ir.stride[1] -
                                    ir.padding[1] +
                                    kernel_column * ir.dilation[1]
                                )
                                if (0 <= input_row < height and
                                        0 <= input_column < width):
                                    partial_sum = compiler.tg.wrap_signed(
                                        partial_sum + compiler.tg.rpe_contribution(
                                            int(input_payload[
                                                batch_index,
                                                input_channel,
                                                input_row,
                                                input_column,
                                            ]),
                                            int(layer.raw_weight[
                                                k_index, output_channel
                                            ]),
                                        ),
                                        compiler.tg.PSUM_W,
                                    )
                                k_index += 1
                                if (k_index % compiler.tc.ARRAY_S == 0 or
                                        k_index == ir.in_features):
                                    accumulator = compiler.tg.wrap_signed(
                                        accumulator + partial_sum,
                                        compiler.tg.ACC_W,
                                    )
                                    partial_sum = 0
                    output[
                        batch_index, output_channel, output_row, output_column
                    ] = compiler.tg.op_model(
                        accumulator,
                        layer.qmult,
                        layer.qshift,
                        layer.act_mode,
                    )
    return output


def check_conv_flow(root, checks):
    model_path = write_demo_conv_model(root)
    _, model, tensors, layers = compiler.compile_quantized_layers(model_path)
    assert [layer.ir.in_features for layer in layers] == [27, 36, 3]
    assert [layer.ir.out_features for layer in layers] == [4, 4, 4]
    checks.append(("Conv2D graph and BatchNorm folding", True))

    bundle = compiler.build_workload(
        model_path, tensors["rtl_inputs"], compiled=None
    )
    assert [layer["shape"] for layer in bundle.manifest["layers"]] == [
        {"M": 9, "K": 27, "N": 4},
        {"M": 9, "K": 36, "N": 4},
        {"M": 9, "K": 3, "N": 4},
    ]
    assert bundle.manifest["layers"][0]["masked_loads"] > 0
    assert bundle.manifest["layers"][1]["masked_loads"] > 0
    assert len(bundle.manifest["host_operations"]) == 1
    checks.append(("Conv2D im2col mask and residual graph", True))

    descriptors = [
        compiler.tc.decode_descriptor(command.word)
        for command in bundle.commands
    ]
    k_valid = {
        fields["k_valid"] for fields in descriptors
        if fields["opcode"] == compiler.tc.CMD_OP_GEMM
    }
    assert k_valid == {3, 4, 27, 32}
    assert any(
        fields["opcode"] == compiler.tc.CMD_OP_LOAD_A and
        fields["a_mask_en"]
        for fields in descriptors
    )
    checks.append(("Conv2D descriptor tails and spatial masks", True))

    compiled_path = compiler.write_compiled_model(
        os.path.join(root, "compiled_conv"), model_path, model, layers
    )
    compiled = compiler.load_compiled_model(compiled_path)
    compiled_bundle = compiler.build_workload(
        compiled_path, tensors["rtl_inputs"], compiled=compiled
    )
    assert [command.word for command in compiled_bundle.commands] == [
        command.word for command in bundle.commands
    ]
    assert compiled_bundle.activation_rows == bundle.activation_rows
    assert compiled_bundle.golden_rows == bundle.golden_rows
    checks.append(("Conv2D compile and emit-test separation", True))

    result, states = reference.graph_bittrue_inference(
        compiled_path, tensors["rtl_inputs"], compiled=compiled
    )
    first_input = compiler.quantize_activation(
        tensors["rtl_inputs"], layers[0].input_scale
    )
    naive = naive_conv_payload(first_input, layers[0])
    assert np.array_equal(naive, states["main_relu1"]["payload"])
    assert result["payload"].shape == (1, 4, 3, 3)
    checks.append(("independent naive Conv2D bit-true reference", True))

    outdir = os.path.join(root, "pattern_conv")
    manifest_path = compiler.write_bundle(outdir, bundle)
    compiler.replay_bundle(manifest_path)
    checks.append(("Conv2D workload bundle replay", True))


def check_manifest(manifest):
    assert len(manifest["layers"]) == 3
    assert [layer["tiles"]["KT"] for layer in manifest["layers"]] == [25, 16, 8]
    assert [layer["tiles"]["NT"] for layer in manifest["layers"]] == [16, 8, 1]
    assert [layer["weight_tiles"] for layer in manifest["layers"]] == [400, 128, 8]
    assert [layer["weight_slots"] for layer in manifest["layers"]] == [250, 128, 8]
    assert manifest["layers"][0]["weight_blocks"] == [
        {"nt_begin": 0, "nt_count": 10},
        {"nt_begin": 10, "nt_count": 6},
    ]
    assert [layer["act_mode"] for layer in manifest["layers"]] == [1, 1, 0]
    assert manifest["test"]["M"] == 32
    assert manifest["test"]["K"] == 784
    assert manifest["test"]["N"] == 10
    assert manifest["test"]["command_count"] == 1683
    assert manifest["test"]["weight_rows"] == 17252
    assert manifest["test"]["activation_rows"] == 800
    assert manifest["test"]["golden_rows"] == 800


def check_descriptors(bundle):
    gemm = [compiler.tc.decode_descriptor(command.word)
            for command in bundle.commands
            if command.opcode == compiler.tc.CMD_OP_GEMM]
    k_valid = sorted({fields["k_valid"] for fields in gemm})
    assert k_valid == [16, 32]
    assert all(fields["bias_en"] == fields["acc_init"] for fields in gemm)
    assert max(fields["w_slot"] for fields in gemm) < compiler.tc.WMEM_SLOTS
    assert max(max(layer["input_slots"] + layer["output_slots"])
               for layer in bundle.manifest["layers"]) < compiler.tc.UB_SLOTS


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix="tpu-compiler-") as root:
        model_path = write_demo_model(root)
        _, _, tensors, _ = compiler.load_model(model_path)
        checks.append((
            "generic frontend contract",
            exporter.MODEL_FORMAT == compiler.MODEL_FORMAT and
            mnist_mlp.model_ir()["output_semantics"] ==
            "classification_logits" and
            mnist_resnet18.model_ir()["layers"][1]["input_shape"] ==
            [128, 14, 14],
        ))
        bundle = compiler.build_workload(
            model_path, tensors["rtl_inputs"], tensors["rtl_labels"]
        )
        checks.append(("independent command replay", len(bundle.golden_rows) == 800))
        check_descriptors(bundle)
        checks.append(("descriptor and slot legality", True))

        outdir = os.path.join(root, "pattern")
        manifest_path = compiler.write_bundle(outdir, bundle)
        manifest = compiler.replay_bundle(manifest_path)
        check_manifest(manifest)
        checks.append(("bundle files and checksums", True))
        checks.append(("complete MNIST MLP topology", True))
        checks.append((
            "scale-aware requant calibration",
            all(
                layer["stats"]["requant"]["relative_factor_error"] < 1.0e-4
                for layer in manifest["layers"][:-1]
            ),
        ))
        final_requant = manifest["layers"][-1]["stats"]["requant"]
        checks.append((
            "label-free final rank calibration",
            final_requant["mode"] == "rank" and
            final_requant["rank"]["candidate_count"] > 1 and
            0 <= final_requant["rank"]["agreement"] <=
            final_requant["rank"]["total"],
        ))

        _, model, _source_tensors, quantized_layers = \
            compiler.compile_quantized_layers(model_path)
        compiled_path = compiler.write_compiled_model(
            os.path.join(root, "compiled"),
            model_path,
            model,
            quantized_layers,
        )
        compiled = compiler.load_compiled_model(compiled_path)
        compiled_bundle = compiler.build_workload(
            compiled_path,
            tensors["rtl_inputs"],
            tensors["rtl_labels"],
            compiled=compiled,
        )
        checks.append((
            "compile and emit-test separation",
            [command.word for command in compiled_bundle.commands] ==
            [command.word for command in bundle.commands] and
            compiled_bundle.weight_rows == bundle.weight_rows and
            compiled_bundle.activation_rows == bundle.activation_rows and
            compiled_bundle.golden_rows == bundle.golden_rows,
        ))

        predictions = reference.hardware_inference(
            model_path, tensors["rtl_inputs"], batch_size=11
        )
        checks.append(("batched hardware inference", predictions.shape == (32,)))

        ablation = reference.quantization_ablation(
            model_path,
            tensors["rtl_inputs"][:8],
            tensors["rtl_labels"][:8],
            batch_size=4,
        )
        checks.append((
            "non-bit-true quantization ablation",
            [stage["name"] for stage in ablation] == [
                "FP32",
                "INT8-W",
                "Type2-W",
                "Type2-W + INT4-A",
                "Type2-W + A5 FixedLSB",
            ],
        ))

        breakdown = reference.bittrue_breakdown(
            model_path,
            tensors["rtl_inputs"][:4],
            tensors["rtl_labels"][:4],
            batch_size=2,
        )
        checks.append((
            "progressive bit-true ablation",
            [stage["name"] for stage in breakdown] == [
                "A5 compiler-scale FP MAC",
                "A5 + INT64 MAC + FP32 bias",
                "+ Bias29",
                "+ Integer QMULT/QSHIFT",
                "+ PSUM21 wrap",
                "+ ACC29 wrap (bit-true)",
            ],
        ))
        progressive = reference.progressive_integer_inference(
            model_path,
            tensors["rtl_inputs"][:4],
            "bittrue",
            batch_size=2,
        )
        checks.append((
            "progressive final stage equivalence",
            np.array_equal(progressive, predictions[:4]),
        ))

        check_conv_flow(root, checks)

    print("TPU model compiler self-check")
    for name, passed in checks:
        print("[%s] %s" % ("PASS" if passed else "FAIL", name))
    if not all(passed for _, passed in checks):
        raise SystemExit(1)
    print("PASS: all compiler checks passed")


if __name__ == "__main__":
    main()
