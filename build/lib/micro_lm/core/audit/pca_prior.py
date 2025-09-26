"""
# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Copyright 2025 Ian C. Moore (Provisional Patents #63/864,726, #63/865,437, #63/871,647 and #63/872,334)
# Email: ngeodesic@gmail.com
# Part of Noetic Geodesic Framework (NGF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

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

