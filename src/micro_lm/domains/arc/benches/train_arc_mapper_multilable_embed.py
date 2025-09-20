#!/usr/bin/env python3
"""
Train an ARC text → multi-label primitive mapper with SBERT embeddings +
One-vs-Rest Logistic Regression (independent sigmoid heads).

Why:
- ARC prompts can imply multiple primitives in sequence (e.g., "rot90 then flip_h").
- This trainer outputs independent per-primitive probabilities, enabling a
  shortlist of primitives (>= threshold) for your WDD/Stage-10/11 parser to
  order and audit.

Usage
-----
PYTHONPATH=. python3 src/micro_lm/domains/arc/benches/train_arc_mapper_multilable_embed.py \
  --labels_csv tests/fixtures/arc/arc_mapper_multilabel.csv \
  --out_path .artifacts/arc_mapper_multilabel.joblib \
  --sbert sentence-transformers/all-MiniLM-L6-v2 \
  --C 8 --max_iter 2000 --calibrate --calibration_method sigmoid

Data schema
-----------
CSV with columns:
  - prompt : str
  - label  : str  (single or multi; use comma ',' or pipe '|' as delimiter)
Examples:
  prompt,label
  "rotate the grid 90 clockwise",rot90
  "rotate 90 then flip horizontally","rot90|flip_h"
  "color the border red then tile 2x2","color_map,tile"

Outputs
-------
- .joblib Pipeline[SBERTEncoder → MultiLabelOvR(LogisticRegression [± calibration])]
- A JSON sidecar with class names and default 0.50 thresholds (optional).

Notes on calibration
--------------------
- For multi-label, we calibrate each binary head independently via CalibratedClassifierCV
  when --calibrate is set. If class support is tiny, we fall back to the uncalibrated head.
"""

import argparse, sys, json, os
from typing import List, Tuple
import pandas as pd
from collections import Counter, defaultdict

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
import joblib

# --- SBERT encoder (same contract as your single-label trainer) ----------------
try:
    from micro_lm.shared.encoders import SBERTEncoder  # preferred repo path
except Exception:
    try:
        from encoders import SBERTEncoder  # fallback if project has top-level encoders.py
    except Exception:
        # inline minimal wrapper (identical API to your existing trainer)
        from sklearn.base import BaseEstimator, TransformerMixin
        from sentence_transformers import SentenceTransformer
        import numpy as np
        class SBERTEncoder(BaseEstimator, TransformerMixin):
            def __init__(self, model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
                         normalize: bool = True, batch_size: int = 64):
                self.model_name_or_path = model_name_or_path
                self.normalize = normalize
                self.batch_size = batch_size
                self._model = None
            def _lazy(self):
                if self._model is None:
                    self._model = SentenceTransformer(self.model_name_or_path)
                return self._model
            def fit(self, X, y=None):
                self._lazy(); return self
            def transform(self, X):
                if not isinstance(X, (list, tuple)):
                    X = list(X)
                emb = self._lazy().encode(
                    X, batch_size=self.batch_size, normalize_embeddings=self.normalize,
                    convert_to_numpy=True, show_progress_bar=False
                )
                import numpy as np
                return np.asarray(emb)

# --- IO helpers ----------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="train_arc_mapper_multilable_embed",
        description="Train ARC multi-label mapper (text → {primitives}) with SBERT + OvR LR"
    )
    ap.add_argument("--labels_csv", required=True,
                    help="CSV with columns: prompt,label (label may contain multiple, comma or pipe-delimited)")
    ap.add_argument("--out_path", default=".artifacts/arc_mapper_multilabel.joblib",
                    help="Output .joblib path for the trained pipeline")
    ap.add_argument("--class_json", default=".artifacts/arc_mapper_multilabel_classes.json",
                    help="Optional JSON to write class names and default thresholds")
    ap.add_argument("--C", type=float, default=8.0,
                    help="Inverse regularization strength for LogisticRegression")
    ap.add_argument("--max_iter", type=int, default=2000,
                    help="Max iterations for LogisticRegression")
    ap.add_argument("--calibrate", action="store_true",
                    help="Enable per-class probability calibration")
    ap.add_argument("--calibration_method", choices=["auto","isotonic","sigmoid"],
                    default="auto", help="Calibration method (per head)")
    ap.add_argument("--calibration_cv", type=int, default=3,
                    help="CV folds for calibration (per head)")
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="SentenceTransformer model name or local path")
    return ap

def parse_multilabel(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    # accept comma or pipe; trim whitespace
    if "|" in s and "," in s:
        parts = [p.strip() for p in s.replace(",", "|").split("|")]
    elif "|" in s:
        parts = [p.strip() for p in s.split("|")]
    else:
        parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]

def load_and_clean(labels_csv: str) -> Tuple[List[str], List[List[str]]]:
    df = pd.read_csv(labels_csv)
    need = {"prompt","label"}
    if not need.issubset(df.columns):
        sys.exit(f"[train_arc_mapper_multilable_embed] labels_csv must have columns {need}, got {df.columns.tolist()}")

    df = df.copy()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["label"]  = df["label"].astype(str)
    df = df.dropna(subset=["prompt","label"]).query("prompt.str.len() > 0", engine="python")
    if df.empty:
        sys.exit("[train_arc_mapper_multilable_embed] No non-empty prompts after cleaning.")

    X = df["prompt"].tolist()
    Y = [parse_multilabel(s) for s in df["label"].tolist()]
    # basic sanity: no empty label sets
    empties = sum(1 for y in Y if len(y)==0)
    if empties:
        print(f"[train_arc_mapper_multilable_embed] WARNING: {empties} rows had empty label sets after parsing; dropping.", file=sys.stderr)
        XY = [(x,y) for x,y in zip(X,Y) if len(y)>0]
        if not XY:
            sys.exit("[train_arc_mapper_multilable_embed] All rows empty after parsing labels.")
        X, Y = map(list, zip(*XY))
    return X, Y

# --- Calibration wrappers (per head) -------------------------------------------
def _wrap_calibrated_binary_lr(C: float, max_iter: int, method: str, cv: int, y_binary) -> object:
    """
    Build a binary LR head, optionally wrapped in CalibratedClassifierCV.
    Chooses safe fallbacks when class support is tiny.
    """
    base = LogisticRegression(
        max_iter=max_iter, C=C, class_weight="balanced", random_state=0
    )
    # calibration feasibility check: both classes must be present
    import numpy as np
    yb = np.asarray(y_binary)
    pos = int((yb == 1).sum()); neg = int((yb == 0).sum())
    if pos == 0 or neg == 0:
        # fall back to uncalibrated if only one class present
        return base

    if method == "auto":
        m = min(pos, neg)
        if m >= max(3, cv):
            method_use, cv_use = "isotonic", max(3, cv)
        elif m >= 2:
            method_use, cv_use = "sigmoid", max(2, min(m, cv))
        else:
            return base
    else:
        method_use, cv_use = method, cv

    try:
        return CalibratedClassifierCV(estimator=base, method=method_use, cv=cv_use)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, method=method_use, cv=cv_use)

def make_ovr_with_optional_calibration(args, Y_bin, classes_: List[str]):
    """
    Construct an OvR estimator where each head can be independently calibrated.
    Strategy:
      - If --calibrate: fit heads one-by-one so we can pass that head's y_binary to the calibrator.
      - Else: use a standard OneVsRestClassifier(LogisticRegression).
    """
    if not args.calibrate:
        return OneVsRestClassifier(
            LogisticRegression(max_iter=args.max_iter, C=args.C, class_weight="balanced", random_state=0)
        )

    # Custom OvR with per-head calibrators
    # We implement a thin wrapper object exposing fit/decision_function/predict_proba like OvR.
    from sklearn.base import BaseEstimator, ClassifierMixin, clone
    import numpy as np

    class CalibratedOvR(BaseEstimator, ClassifierMixin):
        def __init__(self, classes, C, max_iter, method, cv):
            self.classes = list(classes)
            self.C = C; self.max_iter = max_iter
            self.method = method; self.cv = cv
            self.heads = []  # list of fitted estimators in class order

        def fit(self, X, Y):
            # Y expected shape: (n_samples, n_classes) binary
            self.heads = []
            for j, cls in enumerate(self.classes):
                yj = Y[:, j]
                est = _wrap_calibrated_binary_lr(self.C, self.max_iter, self.method, self.cv, yj)
                # Fit this head on current X/Yj
                est.fit(X, yj)
                self.heads.append(est)
            return self

        def decision_function(self, X):
            # Not all calibrated estimators expose decision_function; we fallback to predict_proba
            outs = []
            for est in self.heads:
                if hasattr(est, "decision_function"):
                    outs.append(est.decision_function(X))
                elif hasattr(est, "predict_proba"):
                    import numpy as np
                    proba = est.predict_proba(X)
                    # binary: [:,1] - [:,0] as a pseudo-logit margin
                    outs.append(proba[:, 1] - proba[:, 0])
                else:
                    outs.append(est.predict(X).astype(float))
            return np.vstack(outs).T  # shape (n_samples, n_classes)

        def predict_proba(self, X):
            outs = []
            for est in self.heads:
                if hasattr(est, "predict_proba"):
                    outs.append(est.predict_proba(X)[:, 1])
                else:
                    # approximate via decision_function + sigmoid if needed
                    import numpy as np
                    if hasattr(est, "decision_function"):
                        z = est.decision_function(X)
                        outs.append(1.0 / (1.0 + np.exp(-z)))
                    else:
                        outs.append(est.predict(X).astype(float))
            import numpy as np
            return np.vstack(outs).T  # shape (n_samples, n_classes)

        def predict(self, X):
            import numpy as np
            P = self.predict_proba(X)
            return (P >= 0.5).astype(int)

        @property
        def classes_(self):
            return self.classes

    return CalibratedOvR(classes_, args.C, args.max_iter, args.calibration_method, args.calibration_cv)

# --- main ----------------------------------------------------------------------
def main():
    ap = build_argparser()
    args = ap.parse_args()

    # Load & parse multilabel data
    X, Y_multi = load_and_clean(args.labels_csv)

    # Binarize labels
    mlb = MultiLabelBinarizer()
    Y_bin = mlb.fit_transform(Y_multi)
    classes = list(mlb.classes_)
    k = len(classes)
    if k < 2:
        sys.exit("[train_arc_mapper_multilable_embed] Need at least 2 unique labels for multi-label training.")

    # Build the multi-label OvR head (optionally calibrated)
    ovr_est = make_ovr_with_optional_calibration(args, Y_bin, classes)

    # Build pipeline: SBERTEncoder → OvR(LogisticRegression ± calibration)
    pipe = make_pipeline(SBERTEncoder(args.sbert), ovr_est)
    pipe.fit(X, Y_bin)

    # Persist model
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    joblib.dump(pipe, args.out_path)

    # Optional: write class list + default thresholds
    if args.class_json:
        os.makedirs(os.path.dirname(args.class_json) or ".", exist_ok=True)
        with open(args.class_json, "w") as f:
            json.dump({
                "classes": classes,
                "default_thresholds": {c: 0.50 for c in classes}
            }, f, indent=2)

    # Training summary
    counts = Counter([c for row in Y_multi for c in row])
    support = ", ".join([f"{c}:{counts[c]}" for c in classes])
    print(f"[train_arc_mapper_multilable_embed] wrote {args.out_path} "
          f"(n={len(X)}, k={k}, calibrate={args.calibrate})")
    print(f"[train_arc_mapper_multilable_embed] classes: {classes}")
    print(f"[train_arc_mapper_multilable_embed] support: {support}")

if __name__ == "__main__":
    main()
