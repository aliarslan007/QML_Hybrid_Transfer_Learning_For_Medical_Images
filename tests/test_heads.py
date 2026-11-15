"""Tests for the heads (need torch and PennyLane; skipped otherwise)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")



def test_param_counts():
    from src.heads import build_head, count_params
    from src.utils import load_config
    cfg = load_config()
    expected = {("linear", 4): 5, ("mlp", 4): 25, ("vqc", 4): 26,
                ("linear", 8): 9, ("mlp", 8): 51, ("vqc", 8): 50}
    for (h, d), k in expected.items():
        assert count_params(build_head(h, d, cfg)) == k


def test_vqc_output_shape_and_range():
    from src.heads import VQCHead
    m = VQCHead(4)
    x = (torch.rand(7, 4) - 0.5) * torch.pi
    z = m.expvals(x)
    assert z.shape == (7, 4) and z.abs().max() <= 1 + 1e-6
    assert m(x).shape == (7,)
