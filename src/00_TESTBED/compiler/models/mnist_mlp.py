#!/usr/bin/env python3
"""MNIST MLP adapter for the generic PyTorch TPU exporter."""

import argparse
import os
import sys

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
COMPILER_ROOT = os.path.dirname(HERE)
if COMPILER_ROOT not in sys.path:
    sys.path.insert(0, COMPILER_ROOT)

import pytorch_exporter as exporter  # noqa: E402


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
CHECKPOINT_NAME = "mnist_mlp_model.pth"


def build_model(nn):
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 10)
            self.dropout = nn.Dropout(0.2)

        def forward(self, inputs):
            inputs = inputs.reshape(inputs.shape[0], 784)
            inputs = nn.functional.relu(self.fc1(inputs))
            inputs = self.dropout(inputs)
            inputs = nn.functional.relu(self.fc2(inputs))
            inputs = self.dropout(inputs)
            return self.fc3(inputs)

    return MLP()


def find_checkpoint(checkpoint):
    if checkpoint:
        path = os.path.abspath(checkpoint)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path

    project_root = os.path.normpath(
        os.path.join(COMPILER_ROOT, "..", "..", "..")
    )
    candidates = [
        os.path.join(os.getcwd(), CHECKPOINT_NAME),
        os.path.join(project_root, "model", "MLP", CHECKPOINT_NAME),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "missing %s; pass --checkpoint or copy it to model/MLP" %
        CHECKPOINT_NAME
    )


def flatten_inputs(inputs):
    return inputs.reshape(inputs.shape[0], 784)


def collect_balanced_rtl_batch(dataset, count=32):
    quotas = [count // 10 + (1 if digit < count % 10 else 0) for digit in range(10)]
    selected_inputs = []
    selected_labels = []
    used = [0 for _ in range(10)]
    for sample, label in dataset:
        digit = int(label)
        if used[digit] < quotas[digit]:
            selected_inputs.append(sample.reshape(784).numpy())
            selected_labels.append(digit)
            used[digit] += 1
        if len(selected_inputs) == count:
            break
    return (
        np.asarray(selected_inputs, dtype=np.float32),
        np.asarray(selected_labels, dtype=np.int64),
    )


def model_ir():
    return {
        "name": "mnist_mlp_784_512_256_10",
        "input": {
            "name": "input",
            "shape": [None, 784],
            "layout": "MK",
        },
        "input_features": 784,
        "output_semantics": "classification_logits",
        "source_checkpoint": CHECKPOINT_NAME,
        "preprocess": {
            "layout": "NCHW_flattened",
            "mean": MNIST_MEAN,
            "std": MNIST_STD,
        },
        "layers": [
            {
                "name": "fc1",
                "type": "linear",
                "in_features": 784,
                "out_features": 512,
                "weight": "fc1.weight",
                "bias": "fc1.bias",
                "activation": "relu",
            },
            {
                "name": "fc2",
                "type": "linear",
                "in_features": 512,
                "out_features": 256,
                "weight": "fc2.weight",
                "bias": "fc2.bias",
                "activation": "relu",
            },
            {
                "name": "fc3",
                "type": "linear",
                "in_features": 256,
                "out_features": 10,
                "weight": "fc3.weight",
                "bias": "fc3.bias",
                "activation": "none",
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export the trained MNIST MLP for the TPU compiler"
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--outdir", default=os.path.join(
        "compiler", "artifacts", "mnist"
    ))
    parser.add_argument("--data-dir", default=os.path.join("compiler", "data"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--calibration-count", type=int, default=512)
    parser.add_argument("--rtl-count", type=int, default=32)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    torch, nn, torchvision, transforms = exporter.require_torch()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    train_dataset = torchvision.datasets.MNIST(
        args.data_dir, train=True, download=args.download, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        args.data_dir, train=False, download=args.download, transform=transform
    )
    calibration_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    checkpoint = find_checkpoint(args.checkpoint)
    model = build_model(nn).to(device)
    exporter.load_checkpoint(model, checkpoint, device, torch)

    calibration_inputs, _ = exporter.collect_inputs(
        calibration_loader, args.calibration_count, flatten_inputs
    )
    test_inputs, test_labels = exporter.collect_inputs(
        test_loader, len(test_dataset), flatten_inputs
    )
    rtl_inputs, rtl_labels = collect_balanced_rtl_batch(
        test_dataset, args.rtl_count
    )
    accuracy = exporter.evaluate_classifier(model, test_loader, device, torch)
    json_path = exporter.export_package(
        model,
        model_ir(),
        {
            "calibration_inputs": calibration_inputs,
            "rtl_inputs": rtl_inputs,
            "rtl_labels": rtl_labels,
            "test_inputs": test_inputs,
            "test_labels": test_labels,
        },
        args.outdir,
        "mnist_mlp.npz",
    )
    print("checkpoint: %s" % checkpoint)
    print("FP32 test accuracy: %.2f%%" % (accuracy * 100.0))
    print("compiler model: %s" % json_path)


if __name__ == "__main__":
    main()
