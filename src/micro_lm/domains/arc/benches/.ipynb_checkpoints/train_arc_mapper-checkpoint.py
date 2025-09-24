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

import argparse, sys, os, json, joblib, math
from collections import Counter
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

"""
PYTHONPATH=. python3 src/micro_lm/domains/arc/benches/train_arc_mapper.py \
  --labels_csv tests/fixtures/arc/arc_mapper_labeled.csv \
  --out_path .artifacts/arc_mapper.joblib \
  --C 8 --max_iter 2000 --calibrate
"""

# ---------------------------
# Utilities to load a grid
# ---------------------------
def _load_grid(cell):
    """
    Accepts:
      - JSON list-of-lists string (e.g., '[[0,1],[2,3]]')
      - path to .npy file (contains HxW ints 0..9)
    Returns np.ndarray[int] of shape [H, W].
    """
    if isinstance(cell, (list, tuple, np.ndarray)):
        return np.asarray(cell, dtype=int)
    if isinstance(cell, str):
        s = cell.strip()
        if s.startswith("["):
            return np.asarray(json.loads(s), dtype=int)
        if os.path.exists(s) and s.endswith(".npy"):
            return np.load(s).astype(int)
    raise ValueError(f"Unrecognized grid cell format: {type(cell)} {str(cell)[:64]}...")

# ---------------------------
# Feature helpers for ARC
# ---------------------------
def _palette_hist(g, n_colors=10):
    hist = np.bincount(g.ravel(), minlength=n_colors)[:n_colors].astype(float)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist

def _symmetry_scores(g):
    # Horizontal & vertical symmetry (1 = perfect)
    h, w = g.shape
    # horizontal (mirror across vertical axis)
    left  = g[:, :w//2]
    right = np.fliplr(g[:, w - w//2:])
    wmin  = min(left.shape[1], right.shape[1])
    hsym  = (left[:, :wmin] == right[:, :wmin]).mean() if wmin > 0 else 0.0
    # vertical (mirror across horizontal axis)
    top   = g[:h//2, :]
    bot   = np.flipud(g[h - h//2:, :])
    hmin  = min(top.shape[0], bot.shape[0])
    vsym  = (top[:hmin, :] == bot[:hmin, :]).mean() if hmin > 0 else 0.0
    return float(hsym), float(vsym)

def _connected_components(g):
    # 4-connectivity; counts components over all non-background colors
    H, W = g.shape
    visited = np.zeros_like(g, dtype=bool)
    def neigh(r,c):
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            rr,cc = r+dr, c+dc
            if 0<=rr<H and 0<=cc<W: yield rr,cc
    comp = 0
    for r in range(H):
        for c in range(W):
            if visited[r,c]: continue
            color = g[r,c]
            # treat background as its own large region? Here: count all colors.
            stack=[(r,c)]; visited[r,c]=True
            size=0
            while stack:
                rr,cc = stack.pop()
                size += 1
                for nr,nc in neigh(rr,cc):
                    if not visited[nr,nc] and g[nr,nc]==color:
                        visited[nr,nc]=True
                        stack.append((nr,nc))
            comp += 1
    return comp

def _color_transitions(g):
    # Count adjacent unequal pairs horizontally + vertically, normalized
    H, W = g.shape
    th = (g[:,1:] != g[:,:-1]).sum() if W>1 else 0
    tv = (g[1:,:] != g[:-1,:]).sum() if H>1 else 0
    denom = (H*(W-1) if W>1 else 0) + (W*(H-1) if H>1 else 0)
    return float((th + tv) / denom) if denom>0 else 0.0

def _bbox_stats(g):
    # For all non-background cells (or all cells), compute bbox fill ratio.
    # Here: consider all cells; ARC variants might prefer nonzero mask only.
    H, W = g.shape
    # nonzero mask
    mask = (g != 0)
    if not mask.any():
        return 0.0, 0.0, 0.0
    ys, xs = np.where(mask)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    fill = mask.sum() / float(h*w)
    return float(h/H), float(w/W), float(fill)

# ---------------------------
# ARC feature encoder (sklearn transformer)
# ---------------------------
class ARCFeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Deterministic, tiny feature set for ARC grids:
    - palette histogram (10-dim)
    - grid shape (H, W, aspect, area_norm)
    - symmetry scores (horiz/vert)
    - connected components count (normalized)
    - color transition density
    - bbox stats (hfrac, wfrac, fill)
    -> total ~ 10 + 4 + 2 + 1 + 1 + 3 = 21 dims
    """
    def __init__(self, n_colors=10):
        self.n_colors = n_colors

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for cell in X:
            g = _load_grid(cell)
            H, W = g.shape
            pal = _palette_hist(g, self.n_colors)
            aspect = (W / H) if H>0 else 0.0
            area_norm = (H*W) / 400.0  # normalize by 20x20 ref (ARC cap)
            hs, vs = _symmetry_scores(g)
            cc = _connected_components(g)
            cc_norm = cc / 400.0
            trans = _color_transitions(g)
            hfrac, wfrac, fill = _bbox_stats(g)

            f = np.hstack([
                pal,
                np.array([H, W, aspect, area_norm], dtype=float),
                np.array([hs, vs], dtype=float),
                np.array([cc_norm], dtype=float),
                np.array([trans], dtype=float),
                np.array([hfrac, wfrac, fill], dtype=float),
            ])
            feats.append(f.astype(float))
        return np.vstack(feats)

# ---------------------------
# Trainer (DeFi-like CLI)
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True,
                    help="CSV with columns: grid (JSON or .npy path), label")
    ap.add_argument("--out_path", default=".artifacts/arc_mapper.joblib")
    ap.add_argument("--C", type=float, default=8.0)
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calibration_method", choices=["auto","isotonic","sigmoid"], default="auto")
    ap.add_argument("--calibration_cv", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.labels_csv)
    need = {"grid","label"}
    if not need.issubset(df.columns):
        sys.exit(f"[train_arc_mapper] labels_csv must have columns {need}, got {df.columns.tolist()}")

    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip()
    df = df.dropna(subset=["grid","label"])
    if df.empty:
        sys.exit("[train_arc_mapper] No rows after cleaning.")

    X = df["grid"].tolist()
    y = df["label"].tolist()

    # Base classifier identical vibe to DeFi trainer
    base = LogisticRegression(max_iter=args.max_iter, C=args.C, class_weight="balanced", random_state=0)

    model = base
    if args.calibrate:
        cnt = Counter(y); m = min(cnt.values())
        method = args.calibration_method; cv = args.calibration_cv
        if method == "auto":
            if m >= max(3, cv):
                method, cv = "isotonic", max(3, cv)
            elif m >= 2:
                method, cv = "sigmoid", max(2, min(m, cv))
            else:
                print("[train_arc_mapper] Not enough per-class samples for calibration; skipping.", file=sys.stderr)
        if method in ("isotonic","sigmoid"):
            try:
                model = CalibratedClassifierCV(estimator=base, method=method, cv=cv)
            except TypeError:
                model = CalibratedClassifierCV(base_estimator=base, method=method, cv=cv)

    pipe = make_pipeline(ARCFeatureEncoder(n_colors=10), model)
    pipe.fit(X, y)
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    joblib.dump(pipe, args.out_path)
    print(f"[train_arc_mapper] Wrote mapper to {args.out_path} (n={len(X)})")

if __name__ == "__main__":
    main()
