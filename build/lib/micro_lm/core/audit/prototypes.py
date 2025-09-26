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
import numpy as np
from .utils import l2_normalize

def from_linear_coef(clf) -> np.ndarray:
    W = getattr(clf, "coef_", None)
    if W is None:
        raise ValueError("classifier has no coef_")
    W = np.asarray(W, dtype=np.float32)
    if W.ndim == 1:
        W = W[None, :]
    return l2_normalize(W, axis=1)

def from_centroids(X: np.ndarray, y: np.ndarray, K: int) -> np.ndarray:
    D = X.shape[1]
    P = np.zeros((K, D), dtype=np.float32)
    for k in range(K):
        idx = (y == k)
        if not np.any(idx):
            continue
        c = X[idx].mean(axis=0)
        P[k] = c
    return l2_normalize(P, axis=1)

def default_anchors(K: int, D: int, scale: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0, scale, size=(K, D)).astype(np.float32)