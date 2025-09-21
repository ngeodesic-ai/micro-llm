from __future__ import annotations
import numpy as np
from typing import Tuple

def make_rng(seed: int):
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

def zscore(x: np.ndarray, axis=None, eps: float = 1e-8) -> np.ndarray:
    m = x.mean(axis=axis, keepdims=True)
    s = x.std(axis=axis, keepdims=True) + eps
    return (x - m) / s

def circshift(x: np.ndarray, s: int) -> np.ndarray:
    s = int(s) % x.shape[0]
    if s == 0: return x
    return np.concatenate([x[-s:], x[:-s]], axis=0)

def nxcorr(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).ravel()
    q = np.asarray(q, dtype=np.float32).ravel()
    n = len(q)
    if len(x) < n:
        raise ValueError("signal shorter than template")
    # normalize template
    qz = (q - q.mean()) / (q.std() + 1e-8)
    out = np.empty(len(x) - n + 1, dtype=np.float32)
    for t in range(len(out)):
        seg = x[t:t+n]
        segz = (seg - seg.mean()) / (seg.std() + 1e-8)
        out[t] = float(np.dot(segz, qz) / len(qz))
    return out

def halfmax_window(env: np.ndarray, t_star: int) -> Tuple[int, int]:
    peak = float(env[t_star])
    if peak <= 0:
        return (t_star, t_star)
    thr = 0.5 * peak
    a = t_star
    while a > 0 and env[a] >= thr:
        a -= 1
    b = t_star
    T = len(env)
    while b < T - 1 and env[b] >= thr:
        b += 1
    return (max(0, a), min(T - 1, b))