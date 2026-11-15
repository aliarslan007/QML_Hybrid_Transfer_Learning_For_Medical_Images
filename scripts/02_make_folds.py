"""Create the fixed, stratified training subsamples and save their indices.

Usage:  python scripts/02_make_folds.py
Output: folds/pneumoniamnist_folds.json   (commit this file to GitHub)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import load_features  # noqa: E402
from src.folds import make_folds  # noqa: E402
from src.utils import load_config, save_json  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    y = load_features(cfg["paths"]["features"])["y_train"]
    folds = make_folds(y, cfg["train_sizes"], cfg["n_folds"], cfg["fold_base_seed"])
    save_json(folds, cfg["paths"]["folds"])
    full_ratio = y.mean()
    print(f"Training pool: {len(y)} images, pneumonia fraction {full_ratio:.3f}")
    for n, per_fold in folds.items():
        ratios = [y[np.asarray(ix)].mean() for ix in per_fold.values()]
        print(f"n={n:>5}: {len(per_fold)} folds, pneumonia fraction {min(ratios):.3f}-{max(ratios):.3f}")
    print(f"Saved to {cfg['paths']['folds']}")
