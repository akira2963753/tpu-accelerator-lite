#!/usr/bin/env python3
"""Export the trained project MNIST MLP for TPU inference."""

import argparse
import json
import os

import numpy as np


MODEL_FORMAT = "tpu-mlp-v1"
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
CHECKPOINT_NAME = "mnist_mlp_model.pth"


def require_torch():
    try:
        import torch
        from torch import nn
        import torchvision
        from torchvision import transforms
    except ImportError as error:
        raise SystemExit(
            "PyTorch and torchvision are required only for checkpoint export"
        ) from error
    return torch, nn, torchvision, transforms


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

    project_root = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."
    ))
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


def load_checkpoint(model, checkpoint, device, torch):
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()


def collect_inputs(loader, count):
    inputs = []
    labels = []
    collected = 0
    for batch_inputs, batch_labels in loader:
        take = min(count - collected, batch_inputs.shape[0])
        inputs.append(batch_inputs[:take].reshape(take, 784).cpu().numpy())
        labels.append(batch_labels[:take].cpu().numpy())
        collected += take
        if collected == count:
            break
    if collected != count:
        raise RuntimeError("dataset does not contain enough samples")
    return (
        np.concatenate(inputs).astype(np.float32),
        np.concatenate(labels).astype(np.int64),
    )


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


def evaluate(model, loader, device, torch):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            predictions = model(inputs).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return correct / total


def export_package(model, calibration_loader, test_loader, test_dataset,
                   outdir, calibration_count, rtl_count):
    os.makedirs(outdir, exist_ok=True)
    calibration_inputs, _ = collect_inputs(calibration_loader, calibration_count)
    test_inputs, test_labels = collect_inputs(test_loader, len(test_dataset))
    rtl_inputs, rtl_labels = collect_balanced_rtl_batch(test_dataset, rtl_count)
    state = model.state_dict()

    tensor_name = "mnist_mlp.npz"
    np.savez_compressed(
        os.path.join(outdir, tensor_name),
        **{
            "fc1.weight": state["fc1.weight"].cpu().numpy().astype(np.float32),
            "fc1.bias": state["fc1.bias"].cpu().numpy().astype(np.float32),
            "fc2.weight": state["fc2.weight"].cpu().numpy().astype(np.float32),
            "fc2.bias": state["fc2.bias"].cpu().numpy().astype(np.float32),
            "fc3.weight": state["fc3.weight"].cpu().numpy().astype(np.float32),
            "fc3.bias": state["fc3.bias"].cpu().numpy().astype(np.float32),
            "calibration_inputs": calibration_inputs,
            "rtl_inputs": rtl_inputs,
            "rtl_labels": rtl_labels,
            "test_inputs": test_inputs,
            "test_labels": test_labels,
        }
    )

    model_json = {
        "format": MODEL_FORMAT,
        "name": "mnist_mlp_784_512_256_10",
        "input_features": 784,
        "tensor_file": tensor_name,
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
    json_path = os.path.join(outdir, "mnist_mlp.json")
    with open(json_path, "w", newline="\n", encoding="utf-8") as output:
        json.dump(model_json, output, indent=2)
        output.write("\n")
    return json_path


def main():
    parser = argparse.ArgumentParser(
        description="Export the trained project MNIST MLP for TPU inference"
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--outdir", default=os.path.join("model", "artifacts", "mnist"))
    parser.add_argument("--data-dir", default=os.path.join("model", "data"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--calibration-count", type=int, default=512)
    parser.add_argument("--rtl-count", type=int, default=32)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    torch, nn, torchvision, transforms = require_torch()
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
    load_checkpoint(model, checkpoint, device, torch)

    accuracy = evaluate(model, test_loader, device, torch)
    json_path = export_package(
        model, calibration_loader, test_loader, test_dataset, args.outdir,
        args.calibration_count, args.rtl_count,
    )
    print("checkpoint: %s" % checkpoint)
    print("FP32 test accuracy: %.2f%%" % (accuracy * 100.0))
    print("compiler model: %s" % json_path)


if __name__ == "__main__":
    main()
