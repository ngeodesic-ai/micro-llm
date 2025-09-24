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
from typing import Dict, List, Tuple, Any
import numpy as np

# Helpers
_DEF_EPS = 1e-8

def _argmax_window(env: np.ndarray, window: Tuple[int,int]) -> int:
    a,b = window
    a = max(0, int(a)); b = min(int(b), env.shape[0]-1)
    seg = env[a:b+1]
    return int(a + np.argmax(seg))

def extract_rotate_k(env: np.ndarray, window: Tuple[int,int]) -> int:
    """Pick k∈{0,1,2,3} from 4 rotation channels by peak order inside window.
    Assumes channels [k0,k1,k2,k3] are contiguous in env (caller provides that slice).
    """
    a,b = window
    a = max(0, int(a)); b = min(int(b), env.shape[0]-1)
    # shape (T, 4)
    seg = env[a:b+1]
    # choose channel with max area in the window
    areas = seg.sum(axis=0)
    k = int(np.argmax(areas))
    return k

def extract_recolor_map(hist_src: np.ndarray, hist_tgt: np.ndarray) -> Dict[int,int]:
    """Greedy palette alignment from source to target histogram bins.
    hist_*: shape (C,) counts per color id.
    Returns a dict mapping src_color → tgt_color.
    """
    C = int(hist_src.shape[0])
    src_order = np.argsort(-hist_src)
    tgt_order = np.argsort(-hist_tgt)
    mapping = {int(src): int(tgt) for src, tgt in zip(src_order, tgt_order)}
    return mapping

def extract_translate(com: np.ndarray, window: Tuple[int,int]) -> Tuple[int,int]:
    """Estimate integer (dx,dy) from center-of-mass time trace within window.
    com: (T,2) array of (x,y) over time (precomputed by caller from debug traces).
    """
    a,b = window
    a = max(0, int(a)); b = min(int(b), com.shape[0]-1)
    seg = com[a:b+1]
    if len(seg) < 2:
        return (0,0)
    start = seg[0]; end = seg[-1]
    dx = int(np.round(end[0] - start[0]))
    dy = int(np.round(end[1] - start[1]))
    return dx, dy

def extract_crop_bbox(env2d: np.ndarray, window: Tuple[int,int], thr_frac: float = 0.5) -> Tuple[int,int,int,int]:
    """Compute bbox from 2D envelope inside window using half-max threshold.
    env2d: (T,H,W) or (H,W); if (T,H,W) we reduce over time window by max.
    Returns (x0,y0,x1,y1) inclusive coordinates.
    """
    if env2d.ndim == 3:
        a,b = window
        a = max(0, int(a)); b = min(int(b), env2d.shape[0]-1)
        E = env2d[a:b+1].max(axis=0)
    else:
        E = env2d
    E = np.asarray(E, dtype=np.float32)
    vmax = float(E.max())
    if vmax <= _DEF_EPS:
        return (0,0,E.shape[1]-1,E.shape[0]-1)
    mask = E >= (thr_frac * vmax)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return (0,0,E.shape[1]-1,E.shape[0]-1)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return (x0,y0,x1,y1)

def extract_args(keep: List[int], order: List[int], windows: Dict[int,Tuple[int,int]], env: np.ndarray,
                 labels: List[str]) -> List[Dict[str,Any]]:
    """Produce a time-ordered op list from kept channels.
    env: (T,K) envelope; labels[k] is channel label like 'rotate_0', 'translate', etc.
    Returns list of {op, params, k, t} sorted by t.
    """
    ops: List[Dict[str,Any]] = []
    for k in order:
        if k not in keep: 
            continue
        label = labels[k]
        win = windows.get(k, (0, env.shape[0]-1))
        t = _argmax_window(env[:,k], win)
        if label.startswith("rotate_"):
            # rotation family 4-way: assume k encodes which rot — parse suffix
            try:
                rot = int(label.split("_")[-1]) % 4
            except Exception:
                rot = 0
            ops.append({"op": "rotate_k", "params": {"k": rot}, "k": k, "t": t})
        elif label == "translate":
            # placeholder: dx,dy = 0 until caller provides COM trace
            ops.append({"op": "translate", "params": {"dx": 0, "dy": 0}, "k": k, "t": t})
        elif label == "crop":
            x0,y0,x1,y1 = (0,0,0,0)
            ops.append({"op": "crop_bbox", "params": {"x0":x0,"y0":y0,"x1":x1,"y1":y1}, "k": k, "t": t})
        elif label == "recolor":
            ops.append({"op": "recolor_map", "params": {}, "k": k, "t": t})
        else:
            ops.append({"op": label, "params": {}, "k": k, "t": t})
    ops.sort(key=lambda d: d["t"])  # time-ordered
    return ops