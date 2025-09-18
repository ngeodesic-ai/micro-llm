
#!/usr/bin/env python3
"""
mapper_end_to_end.py — Train & Predict (Notebook-friendly)

What it does
------------
1) Train a simple SBERT(+logistic regression) mapper from a labeled CSV.
2) Save the trained mapper to a .joblib file.
3) Load a JSONL of prompts and predict with a simple confidence threshold.
4) (Optional) If labels are also provided for the prompts, report quick metrics.

Inputs (expected columns / records)
-----------------------------------
- labels_csv (for training): CSV with columns: prompt,label
- prompts_jsonl (for inference): one JSON per line with key "prompt"
- labels_csv_pred (optional, for evaluation of predictions): CSV with columns: prompt,label

Example (run inside a Jupyter cell)
-----------------------------------
from mapper_end_to_end import train_mapper, predict_prompts

# 1) Train
train_mapper(
    labels_csv="tests/fixtures/defi/defi_mapper_labeled_5k.csv",
    out_path=".artifacts/defi_mapper.joblib",
    sbert_model="sentence-transformers/all-MiniLM-L6-v2",
    C=8.0, max_iter=2000,
    calibrate=True, calibration_method="auto", calibration_cv=3
)

# 2) Predict (and compute metrics if labels_csv_pred is given)
predict_prompts(
    mapper_path=".artifacts/defi_mapper.joblib",
    prompts_jsonl="tests/fixtures/defi/defi_mapper_5k_prompts.jsonl",
    labels_csv_pred="tests/fixtures/defi/defi_mapper_labeled_5k.csv",   # optional
    threshold=0.6,
    out_rows_csv=".artifacts/m8_rows_simple.csv"
)
"""

from __future__ import annotations
import json, csv, sys, os, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
import numpy as np

# --- sklearn bits
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# --- sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None
    print("[WARN] sentence-transformers not importable right now. Install it to train/encode.", file=sys.stderr)


# ---------------- SBERT encoder (pipeline compatible) ----------------
class SBERTEncoder(BaseEstimator, TransformerMixin):
    """Lightweight sentence-embedding transformer for sklearn pipelines."""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64, normalize: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers is required to encode prompts.")
            self._model = SentenceTransformer(self.model_name)

    def fit(self, X, y=None):
        self._ensure_model()
        return self

    def transform(self, X):
        self._ensure_model()
        embs = self._model.encode(
            list(X),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(embs)


# ------------------------ I/O helpers ------------------------
def _read_prompts_jsonl(path: str) -> List[str]:
    prompts: List[str] = []
    with open(path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                p = rec.get("prompt", "").strip()
                if p:
                    prompts.append(p)
            except Exception:
                # tolerate occasional garbage lines
                continue
    return prompts

def _read_labels_csv(path: str) -> Dict[str, str]:
    gold: Dict[str, str] = {}
    df = pd.read_csv(path)
    if not {"prompt", "label"}.issubset(df.columns):
        raise ValueError(f"labels_csv must have columns ['prompt','label'], got {df.columns.tolist()}" )
    for _, row in df.iterrows():
        p = str(row["prompt"]).strip()
        y = str(row["label"]).strip()
        if p:
            gold[p] = y
    return gold


# ------------------------ Training ------------------------
def train_mapper(
    labels_csv: str,
    out_path: str = ".artifacts/defi_mapper.joblib",
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    C: float = 8.0,
    max_iter: int = 2000,
    calibrate: bool = True,
    calibration_method: str = "auto",  # 'auto' | 'isotonic' | 'sigmoid'
    calibration_cv: int = 3,
) -> str:
    """Train a SBERT + LogisticRegression pipeline, optionally calibrated, and dump to joblib."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    df = pd.read_csv(labels_csv)
    need = {"prompt","label"}
    if not need.issubset(df.columns):
        raise SystemExit(f"[train_mapper] labels_csv must have columns {need}, got {df.columns.tolist()}" )
    df = df.dropna(subset=["prompt","label"]).copy()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["label"]  = df["label"].astype(str).str.strip()
    df = df[df["prompt"].str.len() > 0]
    if df.empty:
        raise SystemExit("[train_mapper] No non-empty prompts after cleaning.")

    X = df["prompt"].tolist()
    y = df["label"].tolist()

    base = LogisticRegression(max_iter=max_iter, C=C, class_weight="balanced", random_state=0)

    model = base
    if calibrate:
        # pick a safe calibration automatically for tiny classes
        from collections import Counter
        cnt = Counter(y); m = min(cnt.values())
        method = calibration_method; cv = calibration_cv
        if method == "auto":
            if m >= max(3, cv):
                method, cv = "isotonic", max(3, cv)
            elif m >= 2:
                method, cv = "sigmoid", max(2, min(m, cv))
            else:
                print("[train_mapper] Not enough per-class samples for calibration; skipping.", file=sys.stderr)
                method = None
        if method in ("isotonic","sigmoid"):
            try:
                model = CalibratedClassifierCV(estimator=base, method=method, cv=cv)  # sklearn>=1.3
            except TypeError:
                model = CalibratedClassifierCV(base_estimator=base, method=method, cv=cv)  # older sklearn

    pipe = make_pipeline(SBERTEncoder(sbert_model), model)
    pipe.fit(X, y)
    joblib.dump(pipe, out_path)
    print(f"[train_mapper] wrote: {out_path}  (n={len(X)})")
    return out_path


# ------------------------ Prediction & Metrics ------------------------
@dataclass
class PredictResult:
    rows_csv: Optional[str]
    metrics: Optional[dict]


def _predict_proba(mapper, prompts: List[str]) -> Tuple[List[str], np.ndarray]:
    if hasattr(mapper, "classes_"):
        classes = list(map(str, mapper.classes_))
    else:
        # try to infer from predict_proba later
        classes = None

    if hasattr(mapper, "predict_proba"):
        probs = mapper.predict_proba(prompts)
        probs = np.asarray(probs, dtype=float)
        if classes is None:
            classes = list(map(str, getattr(mapper, "classes_", [])))
        return classes, probs

    # decision_function -> softmax fallback
    if hasattr(mapper, "decision_function"):
        logits = mapper.decision_function(prompts)
        logits = np.asarray(logits, dtype=float)
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = ex / (ex.sum(axis=1, keepdims=True) + 1e-12)
        if classes is None:
            classes = list(map(str, getattr(mapper, "classes_", [])))
        return classes, probs

    # predict-only fallback (degenerate probs)
    preds = np.array(mapper.predict(prompts), dtype=object).reshape(-1, 1)
    classes = list(sorted(set(map(str, preds.flatten().tolist()))))
    idx = {c:i for i,c in enumerate(classes)}
    probs = np.zeros((len(prompts), len(classes)), dtype=float)
    for r, y in enumerate(preds.flatten().tolist()):
        probs[r, idx[str(y)]] = 1.0
    return classes, probs


def predict_prompts(
    mapper_path: str,
    prompts_jsonl: str,
    labels_csv_pred: Optional[str] = None,
    threshold: float = 0.6,
    out_rows_csv: Optional[str] = None,
) -> PredictResult:
    """Load mapper, score prompts, write per-row CSV, and (optionally) compute quick metrics."""
    mapper = joblib.load(mapper_path)
    prompts = _read_prompts_jsonl(prompts_jsonl)
    classes, probs = _predict_proba(mapper, prompts)

    top_idx = probs.argmax(axis=1)
    top_conf = probs.max(axis=1)
    fired = (top_conf >= threshold)
    preds = np.array([classes[i] for i in top_idx], dtype=object)
    pred_labels = np.where(fired, preds, "")

    rows = []
    if labels_csv_pred:
        gold = _read_labels_csv(labels_csv_pred)
    else:
        gold = {}

    for p, yhat, conf, fire in zip(prompts, pred_labels, top_conf, fired):
        rows.append({
            "prompt": p,
            "predicted": yhat,
            "confidence": float(conf),
            "abstain": (not bool(fire)),
            "threshold": float(threshold),
            "gold_label": gold.get(p, "")
        })

    rows_csv_path = None
    if out_rows_csv:
        os.makedirs(os.path.dirname(out_rows_csv) or ".", exist_ok=True)
        import csv as _csv
        with open(out_rows_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["prompt","gold_label","predicted","confidence","abstain","threshold"]
            )
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in w.fieldnames})
        rows_csv_path = out_rows_csv
        print(f"[predict] wrote rows: {rows_csv_path}  (n={len(rows)})")

    # quick metrics (if gold labels provided)
    metrics = None
    if gold:
        total = len(prompts)
        abstain_ct = sum(1 for r in rows if r["abstain"])
        fired_ct   = total - abstain_ct
        correct_on_fired = sum(1 for r in rows if (not r["abstain"]) and r["predicted"] == r["gold_label"])
        overall_correct  = sum(1 for r in rows if r["predicted"] == r["gold_label"])  # empty pred never equals gold

        accuracy_on_fired = (correct_on_fired / fired_ct) if fired_ct else None
        overall_accuracy  = overall_correct / total if total else None

        metrics = {
            "threshold": float(threshold),
            "total": total,
            "abstain": abstain_ct,
            "abstain_rate": abstain_ct / total if total else None,
            "coverage": fired_ct / total if total else None,
            "fired": fired_ct,
            "correct_on_fired": correct_on_fired,
            "accuracy_on_fired": accuracy_on_fired,
            "overall_correct": overall_correct,
            "overall_accuracy": overall_accuracy,
        }
        print("[metrics]", metrics)

    return PredictResult(rows_csv=rows_csv_path, metrics=metrics)


# ------------------------ CLI (optional) ------------------------
def _as_bool(x: str) -> bool:
    return str(x).strip().lower() in {"1","true","t","yes","y"}

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train & Predict (end-to-end)")
    ap.add_argument("--labels_csv", help="Training CSV with columns: prompt,label")
    ap.add_argument("--out_path", default=".artifacts/defi_mapper.joblib")
    ap.add_argument("--sbert_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--C", type=float, default=8.0)
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--calibrate", default="true")
    ap.add_argument("--calibration_method", choices=["auto","isotonic","sigmoid"], default="auto")
    ap.add_argument("--calibration_cv", type=int, default=3)
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--labels_csv_pred", default="", help="Optional eval labels (prompt,label) for prompts_jsonl")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--out_rows_csv", default=".artifacts/m8_rows_simple.csv")
    args = ap.parse_args()

    model_path = train_mapper(
        labels_csv=args.labels_csv,
        out_path=args.out_path,
        sbert_model=args.sbert_model,
        C=args.C,
        max_iter=args.max_iter,
        calibrate=_as_bool(args.calibrate),
        calibration_method=args.calibration_method,
        calibration_cv=args.calibration_cv
    )
    predict_prompts(
        mapper_path=model_path,
        prompts_jsonl=args.prompts_jsonl,
        labels_csv_pred=(args.labels_csv_pred or None),
        threshold=args.threshold,
        out_rows_csv=args.out_rows_csv
    )

if __name__ == "__main__":
    main()
