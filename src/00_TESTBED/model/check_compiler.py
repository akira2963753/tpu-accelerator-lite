#!/usr/bin/env python3
"""Self-check the project 784-512-256-10 MLP compiler flow."""

import json
import os
import sys
import tempfile

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tpu_compiler as compiler  # noqa: E402


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
            "tensor_file": "demo.npz",
            "layers": entries,
        }, output, indent=2)
        output.write("\n")
    return model_path


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

        predictions = compiler.hardware_inference(
            model_path, tensors["rtl_inputs"], batch_size=11
        )
        checks.append(("batched hardware inference", predictions.shape == (32,)))

    print("TPU model compiler self-check")
    for name, passed in checks:
        print("[%s] %s" % ("PASS" if passed else "FAIL", name))
    if not all(passed for _, passed in checks):
        raise SystemExit(1)
    print("PASS: all compiler checks passed")


if __name__ == "__main__":
    main()
