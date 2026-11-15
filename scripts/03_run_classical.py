"""Classical baselines on the full grid: linear and parameter-matched MLP heads,
plus the 512-feature logistic-regression reference.

Usage:  python scripts/03_run_classical.py
Output: results/classical.csv  (resumable: finished runs are skipped)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import (ReducedData, ResultLog, run_reference_lr512,  # noqa: E402
                            run_torch_head)
from src.features import load_features  # noqa: E402
from src.utils import load_config, load_json  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--heads", nargs="+", default=["linear", "mlp"])
    p.add_argument("--dims", nargs="+", type=int, default=cfg["dims"])
    p.add_argument("--sizes", nargs="+", type=int, default=cfg["train_sizes"])
    p.add_argument("--folds", nargs="+", type=int, default=list(range(cfg["n_folds"])))
    p.add_argument("--seeds", nargs="+", type=int, default=cfg["init_seeds"])
    p.add_argument("--no-reference", action="store_true")
    a = p.parse_args()

    data = ReducedData(load_features(cfg["paths"]["features"]), a.dims, cfg["clip_percentiles"])
    for d in a.dims:
        print(f"PCA d={d}: explained variance {data.explained[d]:.3f}")
    folds = load_json(cfg["paths"]["folds"])
    log = ResultLog(Path(cfg["paths"]["results"]) / "classical.csv")

    if not a.no_reference:
        for n in a.sizes:
            for k in a.folds:
                if not log.is_done("lr512_reference", 512, n, k, 0):
                    row = run_reference_lr512(n, k, data, folds)
                    log.write(row)
                    print(f"[ref512] n={n} fold={k} test AUROC={row['test_auroc']:.3f}")

    for head in a.heads:
        for d in a.dims:
            for n in a.sizes:
                for k in a.folds:
                    for s in a.seeds:
                        if log.is_done(head, d, n, k, s):
                            continue
                        row = run_torch_head(head, d, n, k, s, data, folds, cfg)
                        log.write(row)
                        print(f"[{head}] d={d} n={n} fold={k} seed={s} "
                              f"params={row['n_params']} val={row['val_auroc']:.3f} "
                              f"test AUROC={row['test_auroc']:.3f}")
