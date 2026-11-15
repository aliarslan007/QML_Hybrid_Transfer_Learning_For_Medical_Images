"""Step 2 of the pipeline: 512-d features -> d angles in [-pi/2, pi/2].

* PCA is fit on the training split only (no labels are used).
  Validation and test images never influence the fit.
* Each PCA component is then mapped linearly to [-pi/2, pi/2] using the
  1st-99th percentile range of the training split, and clipped. Using
  percentiles keeps a few extreme images from squashing everyone else.

Every head (linear, MLP, VQC) receives exactly these d numbers.
"""
import numpy as np
from sklearn.decomposition import PCA


class AngleReducer:
    def __init__(self, d: int, clip_percentiles=(1, 99), seed: int = 0):
        self.d = d
        self.clip_percentiles = clip_percentiles
        self.pca = PCA(n_components=d, random_state=seed)

    def fit(self, X_train: np.ndarray):
        Z = self.pca.fit_transform(X_train)
        lo_p, hi_p = self.clip_percentiles
        self.lo_ = np.percentile(Z, lo_p, axis=0)
        self.hi_ = np.percentile(Z, hi_p, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.pca.transform(X)
        Z = np.clip(Z, self.lo_, self.hi_)
        unit = (Z - self.lo_) / (self.hi_ - self.lo_)        # in [0, 1]
        return ((unit * 2.0 - 1.0) * (np.pi / 2)).astype(np.float32)

    @property
    def explained_variance(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum())
