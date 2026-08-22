#!/usr/bin/env python3
"""Train or load the reference MNIST MLP and export compiler inputs."""

import argparse
import json
import os
import random

import numpy as np


MODEL_FORMAT = "tpu-mlp-v1"
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torchvision
        import torchvision.transforms as transforms
    except ImportError as error:
        raise SystemExit(
            "PyTorch and torchvision are required only for MNIST export"
        ) from error
    return torch, nn, torchvision, transforms


def build_model(nn):
    class MNISTMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)

        def forward(self, inputs):
            inputs = inputs.reshape(inputs.shape[0], 784)
            inputs = nn.functional.relu(self.fc1(inputs))
            inputs = nn.functional.relu(self.fc2(inputs))
            return self.fc3(inputs)

    return MNISTMLP()


def collect_inputs(loader, count, torch):
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


def collect_balanced_rtl_batch(dataset, torch, count=32):
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
    return (np.asarray(selected_inputs, dtype=np.float32),
            np.asarray(selected_labels, dtype=np.int64))


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


def train(model, loader, device, epochs, learning_rate, torch, nn):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_samples = 0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.numel())
            total_samples += int(labels.numel())
        print("epoch %d/%d loss=%.6f" %
              (epoch + 1, epochs, total_loss / total_samples))


def export_package(model, train_loader, test_loader, test_dataset, outdir,
                   calibration_count, rtl_count, torch):
    os.makedirs(outdir, exist_ok=True)
    calibration_inputs, _ = collect_inputs(train_loader, calibration_count, torch)
    test_inputs, test_labels = collect_inputs(test_loader, len(test_dataset), torch)
    rtl_inputs, rtl_labels = collect_balanced_rtl_batch(test_dataset, torch, rtl_count)
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
        "name": "mnist_mlp_784_256_128_10",
        "input_features": 784,
        "tensor_file": tensor_name,
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
                "out_features": 256,
                "weight": "fc1.weight",
                "bias": "fc1.bias",
                "activation": "relu",
            },
            {
                "name": "fc2",
                "type": "linear",
                "in_features": 256,
                "out_features": 128,
                "weight": "fc2.weight",
                "bias": "fc2.bias",
                "activation": "relu",
            },
            {
                "name": "fc3",
                "type": "linear",
                "in_features": 128,
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
    parser = argparse.ArgumentParser(description="Train and export the MNIST MLP")
    parser.add_argument("--outdir", default=os.path.join("model", "artifacts", "mnist"))
    parser.add_argument("--data-dir", default=os.path.join("model", "data"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--calibration-count", type=int, default=512)
    parser.add_argument("--rtl-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    torch, nn, torchvision, transforms = require_torch()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
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
    model = build_model(nn).to(device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print("loaded checkpoint: %s" % args.checkpoint)
    else:
        train(model, train_loader, device, args.epochs,
              args.learning_rate, torch, nn)

    accuracy = evaluate(model, test_loader, device, torch)
    checkpoint_path = os.path.join(args.outdir, "mnist_mlp.pth")
    os.makedirs(args.outdir, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    json_path = export_package(
        model, calibration_loader, test_loader, test_dataset, args.outdir,
        args.calibration_count, args.rtl_count, torch,
    )
    print("FP32 test accuracy: %.2f%%" % (accuracy * 100.0))
    print("checkpoint: %s" % checkpoint_path)
    print("compiler model: %s" % json_path)


if __name__ == "__main__":
    main()
