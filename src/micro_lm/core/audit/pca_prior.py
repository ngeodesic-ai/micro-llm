from __future__ import annotations
from pathlib import Path
from typing import Dict
import json
import numpy as np


def load_pca_prior(path: str) -> dict:
    """
    Load a PCA prior saved as a compressed npz with keys:
      - mean: (D,) float32
      - components: (k, D) float32
    Returns a dict {"mean": np.ndarray[D], "components": np.ndarray[k,D]}.
    Raises ValueError on malformed shapes or dtypes.
    """
    data = np.load(path)
    if "mean" not in data or "components" not in data:
        raise ValueError("npz must contain 'mean' and 'components'")

    mean = np.asarray(data["mean"], dtype=np.float32)
    comps = np.asarray(data["components"], dtype=np.float32)

    # Validate shapes
    if mean.ndim != 1:
        raise ValueError(f"'mean' must be 1D (D,), got shape {mean.shape}")
    if comps.ndim != 2:
        raise ValueError(f"'components' must be 2D (k, D), got shape {comps.shape}")
    if comps.shape[1] != mean.shape[0]:
        raise ValueError(
            f"components second dim (D={comps.shape[1]}) must match mean dim (D={mean.shape[0]})"
        )

    return {"mean": mean, "components": comps}


def apply_pca_prior(x: np.ndarray, prior: dict) -> np.ndarray:
    """
    Project x (D,) into prior's PCA subspace:
      y = components @ (x - mean)
    Returns (k,) float32.
    """
    mean = prior["mean"].astype(np.float32, copy=False)
    comps = prior["components"].astype(np.float32, copy=False)

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or x.shape[0] != mean.shape[0]:
        raise ValueError(f"x must be shape (D,), D={mean.shape[0]}")

    centered = x - mean
    # components: (k, D), centered: (D,) -> (k,)
    y = comps @ centered
    return y.astype(np.float32, copy=False)

