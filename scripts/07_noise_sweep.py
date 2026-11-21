"""Noise sweep: score trained VQC heads under simulated hardware noise.

The heads were trained noiselessly. Here the SAME trained weights are run on
a noisy simulator (density matrices, PennyLane default.mixed) and scored on
the test set. Nothing is retrained.

Noise model (per circuit):
  * gate noise:    depolarizing channel with prob. p after every 1-qubit gate,
                   and 10p on each of the two qubits after every CNOT
                   (two-qubit gates are ~10x noisier on IBM devices)
  * readout noise: bit flip with prob. r on every qubit before measurement
  * shot noise:    <Z> estimated from a finite number of shots

Usage:  python scripts/07_noise_sweep.py
Output: results/noise_sweep.csv, results/noise_sweep.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import ReducedData  # noqa: E402
from src.features import load_features  # noqa: E402
from src.heads import build_head  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.utils import load_config  # noqa: E402

P_GRID = [0.0, 0.001, 0.003, 0.01, 0.03]          # 1-qubit depolarizing probability
SETTINGS = [("gate noise only", 0.0, None),        # (name, readout r, shots)
            ("+ readout 2% + 1024 shots", 0.02, 1024)]


def noisy_expvals(model, x, p, r, shots, seed=0):
    """Same circuit as VQCHead, plus noise channels. Returns <Z_i>, shape (batch, d)."""
    import pennylane as qml
    d, layers, w = model.d, model.layers, model.weights.detach()
    dev = qml.device("default.mixed", wires=d, shots=shots, seed=seed)

    @qml.qnode(dev)
    def circuit(x):
        for i in range(d):
            qml.RY(x[..., i], wires=i)
            qml.DepolarizingChannel(p, wires=i)
        for l in range(layers):
            for i in range(d):
                qml.RY(w[l, i, 0], wires=i)
                qml.RZ(w[l, i, 1], wires=i)
                qml.RY(w[l, i, 2], wires=i)
                qml.DepolarizingChannel(p, wires=i)
            if model.entangle and d > 1:
                for i in range(d):
                    j = (i + 1) % d
                    qml.CNOT(wires=[i, j])
                    qml.DepolarizingChannel(min(10 * p, 0.75), wires=i)
                    qml.DepolarizingChannel(min(10 * p, 0.75), wires=j)
        for i in range(d):
            qml.BitFlip(r, wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(d)]

    out = circuit(torch.as_tensor(x))
    return np.stack([np.asarray(o).reshape(-1) for o in out], axis=-1)


def scores_from(model, z):
    """Readout identical to VQCHead.forward: scale * mean_i <Z_i> + bias."""
    return model.scale.item() * z.mean(axis=-1) + model.bias.item()


if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", nargs="+", default=["vqc", "vqc_noent"])
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--sizes", nargs="+", type=int, default=cfg["train_sizes"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    res = Path(cfg["paths"]["results"])
    runs = pd.read_csv(res / "vqc.csv")
    data = ReducedData(load_features(cfg["paths"]["features"]), [a.d], cfg["clip_percentiles"])
    X, y = data.X[a.d]["test"], data.y["test"]

    rows = []
    for head in a.heads:
        for n in a.sizes:
            for k in a.folds:
                wfile = res / "weights" / f"{head}_d{a.d}_n{n}_f{k}_s{a.seed}.pt"
                run = runs.query("head == @head and d == @a.d and n == @n and fold == @k and seed == @a.seed")
                if not wfile.exists() or run.empty:
                    print("missing, skipped:", wfile.name)
                    continue
                model = build_head(head, a.d, cfg)
                model.load_state_dict(torch.load(wfile))
                model.eval()
                thr = float(run["threshold"].iloc[0])

                # Safety check: zero noise must reproduce the trained model exactly
                with torch.no_grad():
                    ref = model(torch.as_tensor(X, dtype=torch.float32)).numpy()
                z0 = noisy_expvals(model, X, 0.0, 0.0, None)
                assert np.allclose(scores_from(model, z0), ref, atol=1e-5), "noisy circuit != trained circuit"

                for name, r, shots in SETTINGS:
                    for p in P_GRID:
                        m = evaluate(y, scores_from(model, noisy_expvals(model, X, p, r, shots)), thr)
                        rows.append({"head": head, "d": a.d, "n": n, "fold": k, "seed": a.seed,
                                     "setting": name, "p": p, "readout": r, "shots": shots or 0,
                                     "test_auroc": m["auroc"],
                                     "test_balanced_accuracy": m["balanced_accuracy"]})
                print(f"[{head}] n={n} fold={k}: p=0 AUROC {rows[-2*len(P_GRID)]['test_auroc']:.3f}"
                      f" -> p={P_GRID[-1]} AUROC {rows[-len(P_GRID)-1]['test_auroc']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(res / "noise_sweep.csv", index=False)

    # Plot: mean test AUROC vs p, one line per head and setting (n = largest size run)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, metric in zip(axes, ["test_auroc", "test_balanced_accuracy"]):
        for (head, name), g in df[df.n == max(a.sizes)].groupby(["head", "setting"]):
            s = g.groupby("p")[metric].agg(["mean", "std"])
            x = np.where(s.index == 0, 1e-4, s.index)
            ax.errorbar(x, s["mean"], yerr=s["std"], marker="o", capsize=3,
                        color="C0" if head == "vqc" else "C3",
                        linestyle="-" if name == "gate noise only" else "--",
                        label=f"{head}, {name}")
        ax.set_xscale("log")
        ax.set_xlabel("1-qubit depolarizing p (CNOT: 10p); leftmost point = noiseless")
        ax.set_title(metric.replace("test_", "").replace("_", " ") + f"  (n = {max(a.sizes)})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test score (mean ± s.d. over folds)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(res / "noise_sweep.png", dpi=200)
    print(df.groupby(["head", "setting", "n", "p"])["test_auroc"].mean().round(3).to_string())
