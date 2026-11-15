"""Small shared helpers: config loading and seeding."""
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike = "configs/default.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths against the repository root
    for key, value in cfg["paths"].items():
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else REPO_ROOT / p)
    return cfg


def set_seed(seed: int) -> None:
    """Seed every random number generator we use."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
