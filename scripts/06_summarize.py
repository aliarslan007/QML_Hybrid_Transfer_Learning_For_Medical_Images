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
    frames = [pd.read_csv(res / f) for f in ("classical.csv", "vqc.csv")]
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
    colors = {"linear": "C0", "mlp": "C1", "vqc": "C2", "vqc_noent": "C3"}
    ref = df[df["head"] == "lr512_reference"].groupby("n")["test_auroc"].mean()
    for ax, d in zip(axes[0], dims):
        sub = df[(df["d"] == d) & ~df["head"].isin(["vqc_L1", "vqc_L3"])]
        for head, g in sub.groupby("head"):
            stats = g.groupby("n")["test_auroc"].agg(["mean", "std", "count"])
            runs = int(stats["count"].max())
            label = f"{head} ({runs} run{'s' if runs > 1 else ''} per point)"
            ax.errorbar(stats.index, stats["mean"], yerr=stats["std"].fillna(0),
                        marker="o", capsize=3, label=label,
                        color=colors.get(head), linestyle="-" if runs > 1 else ":")
        if len(ref):
            ax.plot(ref.index, ref.values, "k--", label="512-d LR (reference)")
        ax.set_xscale("log")
        ax.set_xlabel("training images n")
        ax.set_title(f"d = {d}")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("test AUROC (mean ± s.d.)")

    handles = {}
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center",
               ncol=min(len(handles), 4), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))

    fig.savefig(res / "learning_curves.png", dpi=200)
    print(f"Saved {res / 'learning_curves.png'}")
