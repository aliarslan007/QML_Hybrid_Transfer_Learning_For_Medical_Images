"""Paired statistics and the pre-registered falsification test.

For each (d, n) and each comparison (VQC vs MLP is the primary one):
  delta = AUROC_VQC - AUROC_other, on runs matched by (fold, seed),
  i.e. both heads trained on the same images from the same seed.

Two 95% confidence intervals for the mean delta:
  * run bootstrap (PRE-REGISTERED in the synopsis): resample the matched
    runs with replacement.
  * fold bootstrap (sensitivity check): resample whole folds, keeping all
    seeds of a chosen fold together. Runs on the same fold share the same
    training images, so they are not independent; this interval is wider
    and more honest about that.

Verdict for each interval (rules fixed in the synopsis):
  above zero        -> VQC better
  inside +-0.02     -> equivalent
  below zero        -> other head better
  otherwise         -> inconclusive
The small-data hypothesis is falsified unless the VQC-vs-MLP interval lies
entirely above zero at both n = 50 and n = 100.

Usage:  python scripts/08_stats.py
Output: results/stats.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_config  # noqa: E402

COMPARISONS = [("vqc", "mlp"), ("vqc", "linear"), ("vqc", "vqc_noent")]
B = 10_000          # bootstrap resamples
MARGIN = 0.02       # equivalence margin (synopsis)


def run_bootstrap(delta, rng):
    idx = rng.integers(0, len(delta), size=(B, len(delta)))
    return delta[idx].mean(axis=1)


def fold_bootstrap(df, rng):
    """df has columns fold, delta. Resample folds, keep each fold's seeds together."""
    groups = [g["delta"].to_numpy() for _, g in df.groupby("fold")]
    k = len(groups)
    means = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, k, size=k)
        means[b] = np.concatenate([groups[i] for i in pick]).mean()
    return means


def verdict(lo, hi):
    if lo > 0:
        return "VQC better"
    if -MARGIN < lo and hi < MARGIN:
        return "equivalent"
    if hi < 0:
        return "other better"
    return "inconclusive"


if __name__ == "__main__":
    cfg = load_config()
    res = Path(cfg["paths"]["results"])
    df = pd.concat([pd.read_csv(res / "classical.csv"), pd.read_csv(res / "vqc.csv")])
    df = df[df["head"] != "lr512_reference"]
    rng = np.random.default_rng(2026)
    keys = ["d", "n", "fold", "seed"]

    rows = []
    for a, b in COMPARISONS:
        A = df[df["head"] == a][keys + ["test_auroc"]]
        Bh = df[df["head"] == b][keys + ["test_auroc"]]
        m = A.merge(Bh, on=keys, suffixes=("_a", "_b"))          # matched runs only
        m["delta"] = m["test_auroc_a"] - m["test_auroc_b"]
        for (d, n), g in m.groupby(["d", "n"]):
            delta = g["delta"].to_numpy()
            r_lo, r_hi = np.percentile(run_bootstrap(delta, rng), [2.5, 97.5])
            f_lo, f_hi = np.percentile(fold_bootstrap(g, rng), [2.5, 97.5])
            rows.append({"comparison": f"{a} - {b}", "d": d, "n": n,
                         "runs": len(g), "folds": g["fold"].nunique(),
                         "mean_delta": delta.mean(),
                         "run_ci_lo": r_lo, "run_ci_hi": r_hi, "run_verdict": verdict(r_lo, r_hi),
                         "fold_ci_lo": f_lo, "fold_ci_hi": f_hi, "fold_verdict": verdict(f_lo, f_hi)})

    out = pd.DataFrame(rows)
    out.to_csv(res / "stats.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))

    print("\nPre-registered falsification test (VQC - MLP, run bootstrap):")
    prim = out[out["comparison"] == "vqc - mlp"]
    for d, g in prim.groupby("d"):
        small = g[g["n"].isin([50, 100])]
        if len(small) < 2:
            print(f"  d={d}: not enough runs yet")
            continue
        ok = (small["run_ci_lo"] > 0).all()
        print(f"  d={d}: small-data hypothesis {'SUPPORTED' if ok else 'FALSIFIED'} "
              f"(runs per point: {small['runs'].tolist()})")
