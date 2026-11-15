"""Training loop shared by every head (identical treatment = fair comparison).

* Full-batch Adam (n <= 1000, so every step sees all training images).
* Class-weighted binary cross-entropy: positives weighted by n_neg / n_pos,
  so the majority class (pneumonia) cannot dominate the loss.
* Early stopping on validation loss; the best-validation weights are restored.
"""
import copy
import time

import numpy as np
import torch
from torch import nn


def to_tensor(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32)


def train_head(model, X_tr, y_tr, X_va, y_va, lr, weight_decay,
               max_epochs=300, patience=30):
    X_tr, y_tr, X_va, y_va = map(to_tensor, (X_tr, y_tr, X_va, y_va))
    n_pos = float(y_tr.sum())
    pos_weight = torch.tensor([(len(y_tr) - n_pos) / max(n_pos, 1.0)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va), y_va).item()
        history.append((loss.item(), val_loss))

        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return {"best_val_loss": best_loss, "best_epoch": best_epoch,
            "epochs_run": epoch, "train_seconds": time.time() - t0,
            "history": history}


@torch.no_grad()
def predict_scores(model, X) -> np.ndarray:
    model.eval()
    return model(to_tensor(X)).cpu().numpy().astype(float)
