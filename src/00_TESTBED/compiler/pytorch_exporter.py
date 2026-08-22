#!/usr/bin/env python3
"""Export PyTorch adapters into the portable TPU compiler format."""

import json
import os

import numpy as np


MODEL_FORMAT = "tpu-model-v1"


def require_torch():
    try:
        import torch
        import torchvision
        from torch import nn
        from torchvision import transforms
    except ImportError as error:
        raise SystemExit(
            "PyTorch and torchvision are required only for model export"
        ) from error
    return torch, nn, torchvision, transforms


def load_checkpoint(model, checkpoint, device, torch):
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()


def collect_inputs(loader, count, transform=None):
    inputs = []
    labels = []
    collected = 0
    for batch_inputs, batch_labels in loader:
        take = min(count - collected, batch_inputs.shape[0])
        selected = batch_inputs[:take]
        if transform:
            selected = transform(selected)
        inputs.append(selected.cpu().numpy())
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


def evaluate_classifier(model, loader, device, torch):
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


def export_package(model, model_ir, datasets, outdir, tensor_name):
    os.makedirs(outdir, exist_ok=True)
    state = model.state_dict()
    tensors = dict(datasets)
    for operation in model_ir["layers"]:
        for field in ("weight", "bias"):
            tensor_key = operation.get(field, "")
            if tensor_key:
                tensors[tensor_key] = state[tensor_key].detach().cpu().numpy() \
                    .astype(np.float32)

    np.savez_compressed(os.path.join(outdir, tensor_name), **tensors)
    document = dict(model_ir)
    document["format"] = MODEL_FORMAT
    document["tensor_file"] = tensor_name
    json_name = os.path.splitext(tensor_name)[0] + ".json"
    json_path = os.path.join(outdir, json_name)
    with open(json_path, "w", newline="\n", encoding="utf-8") as output:
        json.dump(document, output, indent=2)
        output.write("\n")
    return json_path
