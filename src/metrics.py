"""Evaluation metrics.

AUROC is the primary metric (threshold-free, robust to class imbalance).
Threshold-based metrics use a threshold chosen on the validation set
(Youden's J = sensitivity + specificity - 1), never on the test set.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)


def youden_threshold(y_true, scores) -> float:
    fpr, tpr, thr = roc_curve(y_true, scores)
    best = np.argmax(tpr - fpr)
    return float(thr[best]) if np.isfinite(thr[best]) else 0.0


def evaluate(y_true, scores, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auroc": roc_auc_score(y_true, scores),
        "auprc": average_precision_score(y_true, scores),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
    }
