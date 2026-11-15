"""Aggregate results: mean +- s.d. test AUROC per (head, d, n) and learning curves.

Usage:  python scripts/06_summarize.py
Output: results/summary.csv, results/learning_curves.png
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_config  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    res = Path(cfg["paths"]["results"])
    frames = [pd.read_csv(f) for f in sorted(res.glob("*.csv")) if f.name != "summary.csv"]
    if not frames:
        sys.exit("No result files yet.")
    df = pd.concat(frames, ignore_index=True)

    metrics = ["test_auroc", "test_balanced_accuracy", "test_f1",
               "test_sensitivity", "test_specificity"]
    summary = (df.groupby(["head", "d", "n"])[metrics]
                 .agg(["mean", "std", "count"]).round(4))
    summary.to_csv(res / "summary.csv")
    print(summary["test_auroc"].to_string())

    dims = sorted(d for d in df["d"].unique() if d != 512)
    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4), sharey=True, squeeze=False)
    ref = df[df["head"] == "lr512_reference"].groupby("n")["test_auroc"].mean()
    for ax, d in zip(axes[0], dims):
        sub = df[df["d"] == d]
        for head, g in sub.groupby("head"):
            stats = g.groupby("n")["test_auroc"].agg(["mean", "std"])
            ax.errorbar(stats.index, stats["mean"], yerr=stats["std"],
                        marker="o", capsize=3, label=head)
        if len(ref):
            ax.plot(ref.index, ref.values, "k--", label="512-d LR (reference)")
        ax.set_xscale("log")
        ax.set_xlabel("training images n")
        ax.set_title(f"d = {d}")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("test AUROC (mean ± s.d.)")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(res / "learning_curves.png", dpi=200)
    print(f"Saved {res / 'learning_curves.png'}")
