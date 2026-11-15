"""Fixed, stratified training subsamples ("folds").

For every training size n and fold index k we draw n images from the
4,708-image training split, keeping the pneumonia/normal ratio of the full
split. The indices are saved once and reused by every head, so all
comparisons are paired: quantum and classical heads see identical images.
"""
import numpy as np


def stratified_subsample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Return n sorted indices into y with the same class ratio as y."""
    y = np.asarray(y)
    if n > len(y):
        raise ValueError(f"n={n} exceeds pool size {len(y)}")
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    # Allocate per-class counts proportionally, then fix rounding
    quotas = np.floor(counts / counts.sum() * n).astype(int)
    remainder = n - quotas.sum()
    order = np.argsort(-(counts / counts.sum() * n - quotas))
    quotas[order[:remainder]] += 1
    if np.any(quotas < 1):
        raise ValueError("Subsample too small to contain every class")
    picked = [rng.choice(np.flatnonzero(y == c), size=q, replace=False)
              for c, q in zip(classes, quotas)]
    return np.sort(np.concatenate(picked))


def fold_seed(base_seed: int, n: int, fold: int) -> int:
    """Deterministic, distinct seed for each (n, fold) pair."""
    return base_seed * 100_000 + n * 100 + fold


def make_folds(y_train: np.ndarray, sizes, n_folds: int, base_seed: int) -> dict:
    folds = {}
    for n in sizes:
        folds[str(n)] = {
            str(k): stratified_subsample(y_train, n, fold_seed(base_seed, n, k)).tolist()
            for k in range(n_folds)
        }
    return folds
