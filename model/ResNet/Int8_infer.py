import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MODEL_DIR))

from train import ResNet18, get_data_loaders
from int8_utils import apply_weight_int8, count_clipped, evaluate, print_msr_report

CHECKPOINT = "mnist_resnet18_model.pth"
DEVICE = torch.device("cpu")


def find_checkpoint(name: str) -> Path:
    for base in (HERE, Path.cwd()):
        path = base / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {name}. Run `python train.py` in {HERE} first.")


def main() -> None:
    print(f"Device: {DEVICE}")
    ckpt = find_checkpoint(CHECKPOINT)
    print(f"Checkpoint: {ckpt}")

    _, test_loader = get_data_loaders(batch_size=64)
    model = ResNet18().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))

    fp32_acc = evaluate(model, test_loader, DEVICE)
    print(f"FP32  acc: {fp32_acc:.2f}%")

    clipped = sum(count_clipped(p) for n, p in model.named_parameters() if n.endswith(".weight"))
    int8_weights = apply_weight_int8(model)
    int8_acc = evaluate(model, test_loader, DEVICE)

    print(f"INT8  acc: {int8_acc:.2f}%")
    print(f"Clipped weights: {clipped}")
    print_msr_report(int8_weights)


if __name__ == "__main__":
    main()
