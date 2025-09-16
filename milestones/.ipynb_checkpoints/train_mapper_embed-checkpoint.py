#!/usr/bin/env python3
import argparse, sys, joblib
from collections import Counter
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
from milestones.encoders import SBERTEncoder

# Lightweight sentence embeddings (≈22MB model), great for synonyms


"""
PYTHONPATH=. python3 milestones/train_mapper_embed.py \
  --labels_csv tests/fixtures/defi_mapper_labeled_mini.csv \
  --out_path .artifacts/defi_mapper_embed.joblib \
  --sbert sentence-transformers/all-mpnet-base-v2 \
  --C 8 --max_iter 2000 --calibrate
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out_path", default=".artifacts/defi_mapper.joblib")
    ap.add_argument("--C", type=float, default=8.0)
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calibration_method", choices=["auto","isotonic","sigmoid"], default="auto")
    ap.add_argument("--calibration_cv", type=int, default=3)
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    df = pd.read_csv(args.labels_csv)
    need = {"prompt","label"}
    if not need.issubset(df.columns):
        sys.exit(f"[train_mapper_embed] labels_csv must have columns {need}, got {df.columns.tolist()}")
    df = df.copy()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["label"]  = df["label"].astype(str).str.strip()
    df = df.dropna(subset=["prompt","label"])
    df = df[df["prompt"].str.len() > 0]
    if df.empty:
        sys.exit("[train_mapper_embed] No non-empty prompts after cleaning.")

    X = df["prompt"].tolist()
    y = df["label"].tolist()

    # Base classifier
    base = LogisticRegression(max_iter=args.max_iter, C=args.C, class_weight="balanced", random_state=0)

    # Tiny-data friendly calibration
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
                print("[train_mapper_embed] Not enough samples per class for calibration; skipping.", file=sys.stderr)
        if method in ("isotonic","sigmoid"):
            try:
                # sklearn >= 1.3 uses 'estimator'
                model = CalibratedClassifierCV(estimator=base, method=method, cv=cv)
            except TypeError:
                # older sklearn used 'base_estimator'
                model = CalibratedClassifierCV(base_estimator=base, method=method, cv=cv)
    

    pipe = make_pipeline(SBERTEncoder(args.sbert), model)
    pipe.fit(X, y)
    joblib.dump(pipe, args.out_path)
    print(f"[train_mapper_embed] Wrote mapper to {args.out_path} (n={len(X)})")

if __name__ == "__main__":
    main()
