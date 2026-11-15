"""Verify the VQC's gradients: backpropagation (simulator-only shortcut) must
agree with the parameter-shift rule (the method real hardware uses).

Usage:  python scripts/05_gradient_check.py
Pass criterion: max absolute difference < 1e-5 for d = 4 and d = 8.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.heads import VQCHead  # noqa: E402


def grads(diff_method, d, x, y, state):
    torch.manual_seed(0)
    model = VQCHead(d, layers=2, diff_method=diff_method)
    model.load_state_dict(state)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
    loss.backward()
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)       # high precision for the comparison
    ok = True
    for d in (4, 8):
        torch.manual_seed(1)
        x = (torch.rand(16, d) - 0.5) * torch.pi   # angles in [-pi/2, pi/2]
        y = torch.randint(0, 2, (16,)).double()
        ref = VQCHead(d, layers=2, init_std=0.5)
        state = ref.state_dict()
        g_bp = grads("backprop", d, x, y, state)
        g_ps = grads("parameter-shift", d, x, y, state)
        diff = (g_bp - g_ps).abs().max().item()
        passed = diff < 1e-5
        ok &= passed
        print(f"d={d}: {g_bp.numel()} parameters, max |backprop - parameter-shift| = {diff:.2e} "
              f"-> {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if ok else 1)
