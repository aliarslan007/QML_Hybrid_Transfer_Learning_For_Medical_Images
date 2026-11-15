"""Download PneumoniaMNIST (224x224) and cache frozen ResNet-18 features.

Usage:  python scripts/01_extract_features.py
Output: features/pneumoniamnist_224_resnet18.npz
Time:   ~2 min on a Colab GPU, ~10-15 min on a laptop CPU.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import extract_all  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    set_seed(0)
    extract_all(cfg["paths"]["data_root"], cfg["paths"]["features"], size=cfg["image_size"])
