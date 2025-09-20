#!/usr/bin/env python3
"""
Train an ARC text→label mapper using sentence embeddings + (optionally
calibrated) Logistic Regression — mirroring the DeFi trainer, but with
ARC‑specific defaults and artifact names.

Usage examples
--------------
PYTHONPATH=. python3 src/micro_lm/domains/arc/benches/train_arc_mapper_embed.py \
  --labels_csv tests/fixtures/arc/arc_mapper_labeled.csv \
  --out_path .artifacts/arc_mapper.joblib \
  --sbert sentence-transformers/all-MiniLM-L6-v2 \
  --C 8 --max_iter 2000 --calibrate

Notes
-----
- Input CSV must have columns: prompt,label
- Output is a joblib Pipeline[SBERTEncoder → (Calibrated)LogisticRegression]
- Calibration chooses a safe method automatically based on per‑class counts.
"""
import argparse, sys, joblib
from collections import Counter
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline

# This module is expected to exist in your repo, identical to the DeFi trainer.
# It should wrap a SentenceTransformer model and implement .fit/.transform.
# Prefer the shared encoder if available; otherwise fall back to a local wrapper.
try:
    from micro_lm.shared.encoders import SBERTEncoder  # typical repo path
except Exception:
    try:
        from encoders import SBERTEncoder  # fallback if project has a top-level encoders.py
    except Exception:
        # last-resort inline wrapper mirroring the DeFi trainer's SBERTEncoder
        from sklearn.base import BaseEstimator, TransformerMixin
        from sentence_transformers import SentenceTransformer
        import numpy as np

        class SBERTEncoder(BaseEstimator, TransformerMixin):
            def __init__(self, model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2", normalize: bool = True, batch_size: int = 64):
                self.model_name_or_path = model_name_or_path
                self.normalize = normalize
                self.batch_size = batch_size
                self._model = None

            def _lazy(self):
                if self._model is None:
                    self._model = SentenceTransformer(self.model_name_or_path)
                return self._model

            def fit(self, X, y=None):
                self._lazy()
                return self

            def transform(self, X):
                if not isinstance(X, (list, tuple)):
                    X = list(X)
                model = self._lazy()
                emb = model.encode(
                    X,
                    batch_size=self.batch_size,
                    normalize_embeddings=self.normalize,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                if not isinstance(emb, np.ndarray):
                    emb = np.asarray(emb)
                return emb


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="train_arc_mapper_embed",
        description="Train ARC mapper (text → primitive label) with SBERT + LR"
    )
    ap.add_argument("--labels_csv", required=True,
                    help="CSV with columns: prompt,label")
    ap.add_argument("--out_path", default=".artifacts/arc_mapper.joblib",
                    help="Where to write the trained mapper pipeline (.joblib)")
    ap.add_argument("--C", type=float, default=8.0,
                    help="Inverse regularization strength for LogisticRegression")
    ap.add_argument("--max_iter", type=int, default=2000,
                    help="Max iterations for LogisticRegression")
    ap.add_argument("--calibrate", action="store_true",
                    help="Enable probability calibration (isotonic/sigmoid auto)")
    ap.add_argument("--calibration_method", choices=["auto","isotonic","sigmoid"],
                    default="auto", help="Override calibration method")
    ap.add_argument("--calibration_cv", type=int, default=3,
                    help="CV folds for calibration when applicable")
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="SentenceTransformer model name or local path")
    return ap


def load_and_clean(labels_csv: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(labels_csv)
    need = {"prompt", "label"}
    if not need.issubset(df.columns):
        sys.exit(f"[train_arc_mapper_embed] labels_csv must have columns {need}, got {df.columns.tolist()}")

    df = df.copy()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["label"]  = df["label"].astype(str).str.strip()
    df = df.dropna(subset=["prompt","label"]).query("prompt.str.len() > 0", engine="python")
    if df.empty:
        sys.exit("[train_arc_mapper_embed] No non-empty prompts after cleaning.")

    return df["prompt"].tolist(), df["label"].tolist()


def make_model(args: argparse.Namespace):
    base = LogisticRegression(
        max_iter=args.max_iter,
        C=args.C,
        class_weight="balanced",
        random_state=0
    )

    model = base
    if args.calibrate:
        # Pick a safe calibration strategy based on class support
        # to avoid overfitting / failures with tiny classes
        # (mirrors the DeFi trainer behaviour).
        # We'll inspect y later; for now, we set placeholders.
        model = (base, "pending_calibration")  # sentinel
    return model


def wrap_with_calibration(base_estimator, y: list[str], method: str, cv: int):
    cnt = Counter(y)
    m = min(cnt.values())

    if method == "auto":
        if m >= max(3, cv):
            method, cv = "isotonic", max(3, cv)
        elif m >= 2:
            method, cv = "sigmoid", max(2, min(m, cv))
        else:
            print("[train_arc_mapper_embed] Not enough samples/class for calibration; skipping.", file=sys.stderr)
            return base_estimator

    if method in ("isotonic", "sigmoid"):
        try:
            return CalibratedClassifierCV(estimator=base_estimator, method=method, cv=cv)
        except TypeError:  # older sklearn
            return CalibratedClassifierCV(base_estimator=base_estimator, method=method, cv=cv)

    return base_estimator


def main():
    ap = build_argparser()
    args = ap.parse_args()

    X, y = load_and_clean(args.labels_csv)

    base = LogisticRegression(max_iter=args.max_iter, C=args.C, class_weight="balanced", random_state=0)
    if args.calibrate:
        clf = wrap_with_calibration(base, y, args.calibration_method, args.calibration_cv)
    else:
        clf = base

    pipe = make_pipeline(SBERTEncoder(args.sbert), clf)
    pipe.fit(X, y)

    joblib.dump(pipe, args.out_path)

    # Small, useful training summary
    n = len(X)
    k = len(set(y))
    print(f"[train_arc_mapper_embed] Wrote mapper to {args.out_path} (n={n}, k={k}, calibrate={args.calibrate})")


if __name__ == "__main__":
    main()
