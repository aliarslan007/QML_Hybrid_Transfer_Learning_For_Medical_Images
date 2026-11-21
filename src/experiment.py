"""One experiment = one (head, d, n, fold, seed) combination.

For each combination every config in the shared (lr, weight-decay) grid is
trained; the config with the best validation AUROC is kept. Its decision
threshold is chosen on validation, and only then is the test set scored.
"""
import csv
import itertools
from pathlib import Path
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .metrics import evaluate, youden_threshold
from .reduce import AngleReducer
from .utils import set_seed

RESULT_KEYS = ["head", "d", "n", "fold", "seed"]


class ReducedData:
    """PCA-reduced angle features for each d, fit once on the training split."""

    def __init__(self, feats: dict, dims, clip_percentiles):
        self.y = {s: feats[f"y_{s}"] for s in ("train", "val", "test")}
        self.raw = {s: feats[f"X_{s}"] for s in ("train", "val", "test")}
        self.X, self.explained = {}, {}
        for d in dims:
            r = AngleReducer(d, clip_percentiles).fit(self.raw["train"])
            self.X[d] = {s: r.transform(self.raw[s]) for s in ("train", "val", "test")}
            self.explained[d] = r.explained_variance


def run_torch_head(head, d, n, fold, seed, data: ReducedData, folds, cfg):
    from .heads import build_head, count_params
    from .train import predict_scores, train_head

    idx = np.asarray(folds[str(n)][str(fold)])
    X = data.X[d]
    X_tr, y_tr = X["train"][idx], data.y["train"][idx]
    t = cfg["training"]

    best = None
    for lr, wd in itertools.product(t["lr_grid"], t["wd_grid"]):
        set_seed(seed)
        model = build_head(head, d, cfg)
        info = train_head(model, X_tr, y_tr, X["val"], data.y["val"], lr, wd,
                          t["max_epochs"], t["patience"])
        val_scores = predict_scores(model, X["val"])
        val_auroc = roc_auc_score(data.y["val"], val_scores)
        if best is None or val_auroc > best["val_auroc"]:
            best = {"val_auroc": val_auroc, "lr": lr, "wd": wd, "model": model,
                    "val_scores": val_scores, **info}

    thr = youden_threshold(data.y["val"], best["val_scores"])
    test = evaluate(data.y["test"], predict_scores(best["model"], X["test"]), thr)
    wdir = Path(cfg["paths"]["results"]) / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    torch.save(best["model"].state_dict(), wdir / f"{head}_d{d}_n{n}_f{fold}_s{seed}.pt")
    return {"head": head, "d": d, "n": n, "fold": fold, "seed": seed,
            "n_params": count_params(best["model"]), "lr": best["lr"], "wd": best["wd"],
            "val_auroc": best["val_auroc"], "best_epoch": best["best_epoch"],
            "epochs_run": best["epochs_run"], "train_seconds": round(best["train_seconds"], 2),
            "threshold": thr, **{f"test_{k}": v for k, v in test.items()}}


def run_reference_lr512(n, fold, data: ReducedData, folds, C_grid=(0.01, 0.1, 1.0, 10.0)):
    """Reference only: logistic regression on all 512 ResNet features."""
    idx = np.asarray(folds[str(n)][str(fold)])
    scaler = StandardScaler().fit(data.raw["train"])           # unlabeled training split
    X = {s: scaler.transform(data.raw[s]) for s in ("train", "val", "test")}
    best = None
    for C in C_grid:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
        clf.fit(X["train"][idx], data.y["train"][idx])
        v = clf.decision_function(X["val"])
        auc = roc_auc_score(data.y["val"], v)
        if best is None or auc > best[0]:
            best = (auc, C, clf, v)
    auc, C, clf, v = best
    thr = youden_threshold(data.y["val"], v)
    test = evaluate(data.y["test"], clf.decision_function(X["test"]), thr)
    return {"head": "lr512_reference", "d": 512, "n": n, "fold": fold, "seed": 0,
            "n_params": 513, "lr": C, "wd": "", "val_auroc": auc, "best_epoch": "",
            "epochs_run": "", "train_seconds": "", "threshold": thr,
            **{f"test_{k}": v for k, v in test.items()}}


class ResultLog:
    """Append-only CSV; already-finished runs are skipped (safe to resume)."""

    def __init__(self, path):
        self.path = Path(path)
        self.done = set()
        if self.path.exists():
            with open(self.path) as f:
                for row in csv.DictReader(f):
                    self.done.add(tuple(str(row[k]) for k in RESULT_KEYS))

    def is_done(self, head, d, n, fold, seed) -> bool:
        return tuple(map(str, (head, d, n, fold, seed))) in self.done

    def write(self, row: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        self.done.add(tuple(str(row[k]) for k in RESULT_KEYS))
