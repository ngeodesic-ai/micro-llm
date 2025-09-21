from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from .utils import zscore

def synth_traces(emb: np.ndarray, prototypes: np.ndarray, anchors: np.ndarray, T: int = 600,
                 alpha: float = 0.35, gamma: float = 0.04, dt: float = 0.02,
                 guided_env: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    y0 = emb.astype(np.float32)
    P = prototypes.astype(np.float32)
    C = anchors.astype(np.float32)
    K, D = P.shape
    Y = np.zeros((T, K, D), dtype=np.float32)
    Eperp = np.zeros((T, K), dtype=np.float32)
    I = np.eye(D, dtype=np.float32)
    for k in range(K):
        p = P[k]; c = C[k]
        Pi = I - np.outer(p, p)
        y = y0.copy(); v = np.zeros_like(y)
        for t in range(T):
            grad = Pi @ (y - c)
            v = (1 - gamma * dt / 2) * v - alpha * dt * grad
            y = y + dt * v
            Y[t, k] = y
            par = float(np.dot(p, y - c))
            perp = float(np.linalg.norm(y - c)**2 - par**2)
            Eperp[t, k] = perp
        kernel = np.ones(9, dtype=np.float32) / 9.0
        Eperp[:, k] = np.convolve(Eperp[:, k], kernel, mode="same")
    Z = zscore(Eperp, axis=0)
    if guided_env is not None:
        w = 0.2
        Z = (1 - w) * Z + w * zscore(guided_env, axis=0)
    Z = np.clip(Z, 0.0, None)
    return Y, Z