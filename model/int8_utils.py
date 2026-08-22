import torch
import torch.nn as nn

SCALE = 128  # Q1.7 fixed-point scale


def to_int8(w: torch.Tensor) -> torch.Tensor:
    return torch.round(w * SCALE).clamp(-128, 127).to(torch.int8)


def fake_quant(w: torch.Tensor) -> torch.Tensor:
    return to_int8(w).float() / SCALE


def count_clipped(w: torch.Tensor) -> int:
    return int((w.abs() > 127 / SCALE).sum().item())


def apply_weight_int8(model: nn.Module) -> dict[str, torch.Tensor]:
    int8_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w_int8 = to_int8(module.weight.data)
            module.weight.data = w_int8.float() / SCALE
            int8_weights[f"{name}.weight" if name else "weight"] = w_int8
    return int8_weights


def msr_ratio(w_int8: torch.Tensor, n: int) -> float:
    lo = -(1 << (8 - n))
    hi = (1 << (8 - n)) - 1
    mask = (w_int8 >= lo) & (w_int8 <= hi)
    return mask.float().mean().item() * 100.0


def print_msr_report(int8_weights: dict[str, torch.Tensor]) -> None:
    print("--- MSR ---")
    total = torch.cat([w.flatten() for w in int8_weights.values()])
    header = f"{'Layer':<28} {'Count':>10}  " + "  ".join(f"MSR-{n}" for n in range(3, 8))
    print(header)

    for name, w in int8_weights.items():
        ratios = "  ".join(f"{msr_ratio(w, n):6.2f}%" for n in range(3, 8))
        print(f"{name:<28} {w.numel():>10}  {ratios}")

    ratios = "  ".join(f"{msr_ratio(total, n):6.2f}%" for n in range(3, 8))
    msr4 = msr_ratio(total, 4)
    non_msr4_per_256 = (1.0 - msr4 / 100.0) * 256.0
    print(f"{'Total':<28} {total.numel():>10}  {ratios}")
    print(f"MSR-4={msr4:.2f}%  Non-MSR-4/256={non_msr4_per_256:.1f}")


def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total
