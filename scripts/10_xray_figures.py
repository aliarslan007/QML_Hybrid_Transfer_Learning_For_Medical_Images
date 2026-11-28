"""X-ray figures for the report and presentation.

  results/xray_examples.png              4 normal + 4 pneumonia training images
  results/xray_hardware_predictions.png  6 of the 100 hardware test images with the
                                         trained VQC's pneumonia probability in exact
                                         simulation and on the IBM processor

Images are chosen by a fixed rule (the first images of each class in index
order), never by how well the model did on them.

Usage:  python scripts/10_xray_figures.py
Needs:  medmnist (downloads PneumoniaMNIST 224x224) and results/hardware/ from 09_hardware.py
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_config  # noqa: E402

LABEL = {0: "Normal", 1: "Pneumonia"}


def first_of_each_class(labels, k):
    """Indices of the first k images of each class, in index order."""
    labels = np.asarray(labels).reshape(-1)
    return [np.flatnonzero(labels == c)[:k] for c in (0, 1)]


def examples_figure(imgs, labels, out, k=4):
    normal, pneu = first_of_each_class(labels, k)
    fig, axes = plt.subplots(2, k, figsize=(2.1 * k, 4.8), constrained_layout=True)
    for row, (idx, name) in enumerate([(normal, "Normal"), (pneu, "Pneumonia")]):
        for col, i in enumerate(idx):
            ax = axes[row, col]
            ax.imshow(imgs[i], cmap="gray")
            ax.set_xticks([]), ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(name, fontsize=12, fontweight="bold")
    fig.suptitle("PneumoniaMNIST (224 x 224), training images", fontsize=12)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))


def hardware_figure(test_imgs, hw_dir, out, k=3):
    sel = json.loads((hw_dir / "selection.json").read_text())
    test_idx = np.asarray(sel["test_idx"])
    n_test = len(test_idx)
    y = np.load(hw_dir / "y.npy")[:n_test]                  # test images come first
    scale, bias = sel["scale"], sel["bias"]
    prob = {m: sigmoid(scale * np.load(hw_dir / f"z_{m}.npy")[:n_test].mean(axis=1) + bias)
            for m in ("exact", "hardware")}
    normal, pneu = first_of_each_class(y, k)
    fig, axes = plt.subplots(2, k, figsize=(2.5 * k, 6.4), constrained_layout=True)
    for row, idx in enumerate([normal, pneu]):
        for col, j in enumerate(idx):
            ax = axes[row, col]
            ax.imshow(test_imgs[test_idx[j]], cmap="gray")
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"True: {LABEL[int(y[j])]}\n"
                         f"P(pneumonia)  sim {prob['exact'][j]:.2f} | IBM {prob['hardware'][j]:.2f}",
                         fontsize=8.5)
    fig.suptitle("Test X-rays scored by the trained VQC head (d=4, n=1000):\n"
                 "exact simulation vs. IBM ibm_fez hardware", fontsize=10.5)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    from medmnist import PneumoniaMNIST
    cfg = load_config()
    root = cfg["paths"]["data_root"]
    Path(root).mkdir(parents=True, exist_ok=True)
    res = Path(cfg["paths"]["results"])
    train = PneumoniaMNIST(split="train", download=True, size=224, root=root)
    test = PneumoniaMNIST(split="test", download=True, size=224, root=root)
    examples_figure(train.imgs, train.labels, res / "xray_examples.png")
    hw_dir = res / "hardware"
    if (hw_dir / "selection.json").exists():
        hardware_figure(test.imgs, hw_dir, res / "xray_hardware_predictions.png")
    else:
        print("results/hardware/ not found: skipped the hardware figure")


if __name__ == "__main__":
    main()
