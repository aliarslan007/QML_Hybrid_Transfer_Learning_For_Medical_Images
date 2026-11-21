"""VQC head (first quantum model). By default runs a Week-1 pilot:
d=4, all training sizes, fold 0, seed 0. Pass flags for the full grid (Week 2).

Usage (pilot):     python scripts/04_run_vqc.py
Usage (full grid): python scripts/04_run_vqc.py --dims 4 8 --folds 0 1 2 3 4 --seeds 0 1 2
Ablation:          python scripts/04_run_vqc.py --heads vqc_noent ...
Output: results/vqc.csv  (resumable)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import ReducedData, ResultLog, run_torch_head  # noqa: E402
from src.features import load_features  # noqa: E402
from src.heads import build_head  # noqa: E402
from src.utils import load_config, load_json  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--heads", nargs="+", default=["vqc"])
    p.add_argument("--dims", nargs="+", type=int, default=[4])
    p.add_argument("--sizes", nargs="+", type=int, default=cfg["train_sizes"])
    p.add_argument("--folds", nargs="+", type=int, default=[0])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    a = p.parse_args()

    print(build_head(a.heads[0], a.dims[0], cfg).draw())   # show the circuit once   # show the circuit once

    data = ReducedData(load_features(cfg["paths"]["features"]), a.dims, cfg["clip_percentiles"])
    folds = load_json(cfg["paths"]["folds"])
    log = ResultLog(Path(cfg["paths"]["results"]) / "vqc.csv")

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
                              f"test AUROC={row['test_auroc']:.3f} ({row['train_seconds']}s)")
