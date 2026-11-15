"""Step 1 of the pipeline: images -> frozen ResNet-18 -> 512-d feature vectors.

Features are computed once and cached to disk. Every later experiment
trains only on the cached features, so the CNN never runs again.
"""
from pathlib import Path

import numpy as np

# ImageNet statistics expected by the pretrained ResNet-18
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform():
    from torchvision import transforms
    # MedMNIST+ already provides 224x224 images, so no resizing is needed.
    # as_rgb=True in the dataset copies the grayscale channel to 3 channels.
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_backbone(device):
    """ImageNet-pretrained ResNet-18 with the classification layer removed."""
    import torch
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()          # output = 512-d pooled features
    for p in model.parameters():            # frozen: no weight is ever updated
        p.requires_grad_(False)
    return model.eval().to(device)


def extract_split(model, dataset, device, batch_size=128):
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            feats.append(model(x.to(device)).cpu().numpy())
            labels.append(np.asarray(y).reshape(-1))
    return np.concatenate(feats).astype(np.float32), np.concatenate(labels).astype(np.int64)


def extract_all(data_root: str, out_path: str, size: int = 224, device: str | None = None):
    import torch
    from medmnist import PneumoniaMNIST

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Path(data_root).mkdir(parents=True, exist_ok=True)
    transform = build_transform()
    model = build_backbone(device)

    arrays = {}
    for split in ("train", "val", "test"):
        ds = PneumoniaMNIST(split=split, download=True, size=size, as_rgb=True,
                            transform=transform, root=data_root)
        X, y = extract_split(model, ds, device)
        arrays[f"X_{split}"], arrays[f"y_{split}"] = X, y
        print(f"{split:5s}: X {X.shape}, positives {int(y.sum())}/{len(y)}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    print(f"Saved features to {out_path}")
    return arrays


def load_features(path: str) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
