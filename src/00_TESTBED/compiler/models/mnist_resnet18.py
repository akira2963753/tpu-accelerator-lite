#!/usr/bin/env python3
"""Export the trained MNIST ResNet-18 layer2[0] block."""

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
CHECKPOINT_NAME = "mnist_resnet18_model.pth"
BLOCK_NAME = "layer2.0"


def build_model(nn):
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.bn2 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes),
                )

        def forward(self, inputs):
            output = nn.functional.relu(self.bn1(self.conv1(inputs)))
            output = self.bn2(self.conv2(output))
            output = output + self.shortcut(inputs)
            return nn.functional.relu(output)

    class ResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_planes = 64
            self.conv1 = nn.Conv2d(
                1, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
            self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
            self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
            self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(512, 10)

        def _make_layer(self, block, planes, count, stride):
            strides = [stride] + [1] * (count - 1)
            layers = []
            for block_stride in strides:
                layers.append(block(self.in_planes, planes, block_stride))
                self.in_planes = planes
            return nn.Sequential(*layers)

        def forward(self, inputs):
            output = nn.functional.relu(self.bn1(self.conv1(inputs)))
            output = self.layer1(output)
            output = self.layer2(output)
            output = self.layer3(output)
            output = self.layer4(output)
            output = self.avgpool(output)
            output = self.dropout(output.flatten(1))
            return self.fc(output)

    return ResNet()


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
        os.path.join(project_root, "model", "ResNet", CHECKPOINT_NAME),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "missing %s; pass --checkpoint or copy it to model/ResNet" %
        CHECKPOINT_NAME
    )


def collect_block_inputs(model, loader, count, device, torch):
    captured_inputs = []
    captured_outputs = []
    captured_labels = []
    block = model.layer2[0]

    def capture_input(_module, arguments):
        captured_inputs.append(arguments[0].detach().cpu())

    def capture_output(_module, _arguments, output):
        captured_outputs.append(output.detach().cpu())

    input_hook = block.register_forward_pre_hook(capture_input)
    output_hook = block.register_forward_hook(capture_output)
    collected = 0
    model.eval()
    try:
        with torch.no_grad():
            for inputs, labels in loader:
                take = min(count - collected, inputs.shape[0])
                model(inputs[:take].to(device))
                captured_labels.append(labels[:take].cpu())
                collected += take
                if collected == count:
                    break
    finally:
        input_hook.remove()
        output_hook.remove()
    if collected != count:
        raise RuntimeError("dataset does not contain enough samples")
    return (
        torch.cat(captured_inputs).numpy().astype(np.float32),
        torch.cat(captured_outputs).numpy().astype(np.float32),
        torch.cat(captured_labels).numpy().astype(np.int64),
    )


def batch_norm_ir(prefix):
    return {
        "weight": prefix + ".weight",
        "bias": prefix + ".bias",
        "running_mean": prefix + ".running_mean",
        "running_var": prefix + ".running_var",
        "eps": 1.0e-5,
    }


def model_ir():
    return {
        "name": "mnist_resnet18_layer2_0",
        "input": {
            "name": "block_input",
            "shape": [None, 64, 28, 28],
            "layout": "NCHW",
        },
        "output": {
            "name": "block_output",
            "shape": [None, 128, 14, 14],
            "layout": "NCHW",
        },
        "output_semantics": "feature_map",
        "source_checkpoint": CHECKPOINT_NAME,
        "preprocess": {
            "dataset": "MNIST",
            "mean": MNIST_MEAN,
            "std": MNIST_STD,
            "capture": BLOCK_NAME,
        },
        "layers": [
            {
                "name": "main_conv1",
                "type": "conv2d",
                "input": "block_input",
                "output": "main_relu1",
                "input_shape": [64, 28, 28],
                "output_shape": [128, 14, 14],
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": [3, 3],
                "stride": [2, 2],
                "padding": [1, 1],
                "dilation": [1, 1],
                "groups": 1,
                "weight": "layer2.0.conv1.weight",
                "bias": "",
                "batch_norm": batch_norm_ir("layer2.0.bn1"),
                "activation": "relu",
            },
            {
                "name": "main_conv2",
                "type": "conv2d",
                "input": "main_relu1",
                "output": "main_output",
                "input_shape": [128, 14, 14],
                "output_shape": [128, 14, 14],
                "in_channels": 128,
                "out_channels": 128,
                "kernel_size": [3, 3],
                "stride": [1, 1],
                "padding": [1, 1],
                "dilation": [1, 1],
                "groups": 1,
                "weight": "layer2.0.conv2.weight",
                "bias": "",
                "batch_norm": batch_norm_ir("layer2.0.bn2"),
                "activation": "none",
            },
            {
                "name": "shortcut_conv",
                "type": "conv2d",
                "input": "block_input",
                "output": "shortcut_output",
                "input_shape": [64, 28, 28],
                "output_shape": [128, 14, 14],
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": [1, 1],
                "stride": [2, 2],
                "padding": [0, 0],
                "dilation": [1, 1],
                "groups": 1,
                "weight": "layer2.0.shortcut.0.weight",
                "bias": "",
                "batch_norm": batch_norm_ir("layer2.0.shortcut.1"),
                "activation": "none",
            },
        ],
        "host_operations": [
            {
                "name": "residual_add_relu",
                "type": "add",
                "inputs": ["main_output", "shortcut_output"],
                "output": "block_output",
                "activation": "relu",
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export MNIST ResNet-18 layer2[0] for the TPU compiler"
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--outdir", default=os.path.join(
        "compiler", "artifacts", "resnet18_layer2_0"
    ))
    parser.add_argument("--data-dir", default=os.path.join("compiler", "data"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--calibration-count", type=int, default=64)
    parser.add_argument("--rtl-count", type=int, default=1)
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
    rtl_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    checkpoint = find_checkpoint(args.checkpoint)
    model = build_model(nn).to(device)
    exporter.load_checkpoint(model, checkpoint, device, torch)

    calibration_inputs, _, _ = collect_block_inputs(
        model, calibration_loader, args.calibration_count, device, torch
    )
    rtl_inputs, rtl_reference_outputs, rtl_labels = collect_block_inputs(
        model, rtl_loader, args.rtl_count, device, torch
    )
    json_path = exporter.export_package(
        model,
        model_ir(),
        {
            "calibration_inputs": calibration_inputs,
            "rtl_inputs": rtl_inputs,
            "rtl_reference_outputs": rtl_reference_outputs,
            "rtl_labels": rtl_labels,
        },
        args.outdir,
        "mnist_resnet18_layer2_0.npz",
    )
    print("checkpoint: %s" % checkpoint)
    print("block: %s" % BLOCK_NAME)
    print("calibration inputs: %s" % (calibration_inputs.shape,))
    print("RTL inputs: %s" % (rtl_inputs.shape,))
    print("compiler model: %s" % json_path)


if __name__ == "__main__":
    main()
