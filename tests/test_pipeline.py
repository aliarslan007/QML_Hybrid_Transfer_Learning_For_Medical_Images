"""Sanity tests for the data pipeline. Run:  python -m pytest -q
Tests needing torch/PennyLane are skipped automatically if those are missing.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import ReducedData, ResultLog, run_reference_lr512  # noqa: E402
from src.folds import make_folds, stratified_subsample  # noqa: E402
from src.metrics import evaluate, youden_threshold  # noqa: E402
from src.reduce import AngleReducer  # noqa: E402


def fake_features(seed=0, n=(4708, 524, 624), dim=512, pos_rate=0.74):
    """Synthetic 'ResNet features' with a weak class signal."""
    rng = np.random.default_rng(seed)
    out = {}
    for split, m in zip(("train", "val", "test"), n):
        y = (rng.random(m) < pos_rate).astype(np.int64)
        X = rng.normal(size=(m, dim)).astype(np.float32)
        X[:, :5] += 0.8 * y[:, None]
        out[f"X_{split}"], out[f"y_{split}"] = X, y
    return out


def test_subsample_is_stratified_and_deterministic():
    y = np.array([1] * 3494 + [0] * 1214)        # PneumoniaMNIST train ratio
    a = stratified_subsample(y, 50, seed=7)
    b = stratified_subsample(y, 50, seed=7)
    assert np.array_equal(a, b)
    assert len(np.unique(a)) == 50
    assert abs(y[a].mean() - y.mean()) <= 1 / 50


def test_folds_distinct_and_complete():
    y = np.array([1] * 3494 + [0] * 1214)
    folds = make_folds(y, [50, 1000], n_folds=5, base_seed=2026)
    assert set(folds) == {"50", "1000"}
    assert all(len(ix) == int(n) for n, f in folds.items() for ix in f.values())
    assert folds["50"]["0"] != folds["50"]["1"]


def test_reducer_range_and_no_leakage():
    f = fake_features()
    r = AngleReducer(4).fit(f["X_train"])
    Z = r.transform(f["X_test"])
    assert Z.shape == (624, 4)
    assert Z.min() >= -np.pi / 2 - 1e-6 and Z.max() <= np.pi / 2 + 1e-6
    # Fitting must depend on the training split only
    r2 = AngleReducer(4).fit(f["X_train"])
    assert np.allclose(r.transform(f["X_val"]), r2.transform(f["X_val"]))


def test_metrics_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate(y, s, youden_threshold(y, s))
    assert m["auroc"] == 1.0 and m["balanced_accuracy"] == 1.0


def test_reference_and_resultlog(tmp_path):
    f = fake_features()
    data = ReducedData(f, [4], (1, 99))
    folds = make_folds(f["y_train"], [100], 1, 2026)
    row = run_reference_lr512(100, 0, data, folds)
    assert 0.5 < row["test_auroc"] <= 1.0
    log = ResultLog(tmp_path / "r.csv")
    log.write(row)
    assert ResultLog(tmp_path / "r.csv").is_done("lr512_reference", 512, 100, 0, 0)
