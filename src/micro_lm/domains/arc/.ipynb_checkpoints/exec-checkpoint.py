from __future__ import annotations
from typing import Dict, List, Any, Tuple
import numpy as np

# Images are HxW integer arrays (0..C-1 for ARC-like palette) unless noted.

def apply_rotate_k(img: np.ndarray, k: int) -> np.ndarray:
    k = int(k) % 4
    return np.rot90(img, k=k)

def apply_recolor_map(img: np.ndarray, mapping: Dict[int,int]) -> np.ndarray:
    if not mapping:
        return img.copy()
    out = img.copy()
    # vectorized mapping via LUT
    maxc = int(max(mapping.keys())) if mapping else 0
    lut = np.arange(max(maxc+1, img.max()+1), dtype=img.dtype)
    for s,t in mapping.items():
        lut[int(s)] = int(t)
    return lut[out]

def apply_translate(img: np.ndarray, dx: int, dy: int, pad: int = 0) -> np.ndarray:
    H, W = img.shape[:2]
    out = np.full_like(img, pad)
    xs = np.arange(W); ys = np.arange(H)
    X, Y = np.meshgrid(xs, ys)
    Xs = X - int(dx); Ys = Y - int(dy)
    mask = (Xs >= 0) & (Xs < W) & (Ys >= 0) & (Ys < H)
    out[mask] = img[Ys[mask], Xs[mask]]
    return out

def apply_crop_bbox(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, pad: int = 0) -> np.ndarray:
    H, W = img.shape[:2]
    x0 = max(0, int(x0)); y0 = max(0, int(y0)); x1 = min(W-1, int(x1)); y1 = min(H-1, int(y1))
    if x0 > x1 or y0 > y1:
        return img.copy()
    crop = img[y0:y1+1, x0:x1+1]
    return crop

def replay(img0: np.ndarray, ops_seq: List[Dict[str,Any]], L_MAX: int = 3) -> np.ndarray:
    img = img0.copy()
    for i, op in enumerate(ops_seq[:int(L_MAX)]):
        name = op.get("op")
        p = op.get("params", {})
        if name == "rotate_k":
            img = apply_rotate_k(img, p.get("k", 0))
        elif name == "recolor_map":
            img = apply_recolor_map(img, p.get("mapping", {}))
        elif name == "translate":
            img = apply_translate(img, p.get("dx", 0), p.get("dy", 0))
        elif name == "crop_bbox":
            img = apply_crop_bbox(img, p.get("x0",0), p.get("y0",0), p.get("x1",0), p.get("y1",0))
        else:
            # unknown op → no-op to keep solver robust
            img = img
    return img