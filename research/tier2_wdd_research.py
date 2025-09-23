#!/usr/bin/env python3
"""
Tier‑2 WDD Research Scaffold
----------------------------
Goal:
  Evaluate a simple Warped Distance Detector (WDD) over PCA(SBERT) latents for DeFi prompts.
  - Embed prompts with SBERT
  - PCA → d=9 (optionally whiten)
  - Build class prototypes (means)
  - Score examples by margin = best(sim) - second_best(sim) (or negative distances)
  - Calibrate an abstain threshold via permutation nulls
  - Report accuracy, abstain rate, PR-like curves vs threshold
  - (Optional) Compare to an existing mapper .joblib if provided

Usage:
  PYTHONPATH=. python3 research/tier2_wdd_research.py \
    --labels_csv tests/fixtures/defi/defi_mapper_labeled_5k.csv \
    --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl \
    --sbert sentence-transformers/all-MiniLM-L6-v2 \
    --out_dir .artifacts/wdd \
    --dims 9 --whiten \
    --n_null 128 \
    --compare_mapper .artifacts/defi_mapper.joblib

Notes:
  - This is CPU-friendly. SBERT embedding is the slowest step.
  - No seaborn deps; outputs are CSV + JSON so you can plot in your notebooks or CI.
"""
import argparse, json, csv, os, math, random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

# Optional: SBERT encoder (requires internet on first run if model not cached)
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None

SEED = 42
rng = np.random.default_rng(SEED)

def load_labels_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect columns: prompt,label
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    assert {"prompt","label"} <= set(df.columns), "labels_csv must have columns: prompt,label"
    return df

def load_prompts_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                J = json.loads(line)
                p = J.get("prompt","")
                rows.append({"prompt": p})
            except Exception:
                pass
    return pd.DataFrame(rows)

def merge_labels_prompts(labels_df: pd.DataFrame, prompts_df: pd.DataFrame) -> pd.DataFrame:
    # If prompts_jsonl has a subset/superset, left-join on prompt
    if prompts_df is None or prompts_df.empty:
        return labels_df.copy()
    df = prompts_df.merge(labels_df[["prompt","label"]], on="prompt", how="left")
    # drop rows without labels (optional)
    df = df.dropna(subset=["label"])
    return df

class SBERTFeaturizer:
    def __init__(self, model_name: str):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not available in this environment.")
        self.model_name = model_name
        self.enc = SentenceTransformer(model_name)
    def encode(self, texts: List[str]) -> np.ndarray:
        return np.array(self.enc.encode(texts, show_progress_bar=True))

def pca_whiten(X: np.ndarray, d: int, whiten: bool) -> Tuple[np.ndarray, PCA, StandardScaler]:
    # Standardize features prior to PCA for stability
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xn = scaler.fit_transform(X)
    pca = PCA(n_components=d, whiten=whiten, random_state=SEED)
    Y = pca.fit_transform(Xn)
    return Y, pca, scaler

def build_class_prototypes(Y: np.ndarray, labels: List[str]) -> Dict[str, np.ndarray]:
    protos = {}
    for cls in sorted(set(labels)):
        idx = [i for i,l in enumerate(labels) if l==cls]
        if idx:
            protos[cls] = Y[idx].mean(axis=0)
    return protos

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A_norm @ B_norm.T

def wdd_scores(Y: np.ndarray, labels: List[str], protos: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    classes = sorted(protos.keys())
    P = np.stack([protos[c] for c in classes], axis=0)  # [C, d]
    sims = cosine_sim_matrix(Y, P)                      # [N, C]
    top1_idx = np.argmax(sims, axis=1)
    top1 = [classes[i] for i in top1_idx]
    # margin = best - second best
    part = np.partition(sims, -2, axis=1)
    second_best = part[:, -2]
    best = sims[np.arange(sims.shape[0]), top1_idx]
    margin = best - second_best
    return margin, top1, sims

def permutation_null_margins(Y: np.ndarray, labels: List[str], protos: Dict[str, np.ndarray], n_null: int=128) -> np.ndarray:
    """Build null margins by shuffling labels -> random prototypes."""
    classes = sorted(set(labels))
    margins = []
    for _ in range(n_null):
        # reassign class means randomly
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        protos_null = build_class_prototypes(Y, shuffled)
        m, _, _ = wdd_scores(Y, labels, protos_null)
        margins.append(m)
    return np.concatenate(margins)

def evaluate_at_threshold(df: pd.DataFrame, thr: float) -> Dict[str, float]:
    fired = df["margin"] >= thr
    preds = df["top1"].where(fired, None)
    gold = df["label"]
    total = len(df)
    fired_n = int(fired.sum())
    abstain_n = total - fired_n
    correct_on_fired = int(((preds == gold) & fired).sum())
    overall_correct = int((preds == gold).sum())
    return {
        "threshold": float(thr),
        "total": total,
        "fired": fired_n,
        "abstain": abstain_n,
        "abstain_rate": abstain_n / total,
        "accuracy_on_fired": (correct_on_fired / fired_n) if fired_n else 0.0,
        "overall_correct": overall_correct,
        "overall_accuracy": overall_correct / total,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--prompts_jsonl", default=None)
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--dims", type=int, default=9)
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--n_null", type=int, default=128)
    ap.add_argument("--out_dir", default=".artifacts/wdd")
    ap.add_argument("--thresholds", default="auto")  # "auto" builds from null; or comma list
    ap.add_argument("--compare_mapper", default=None) # optional .joblib pipeline to compare
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    labels_df = load_labels_csv(args.labels_csv)
    prompts_df = load_prompts_jsonl(args.prompts_jsonl) if args.prompts_jsonl else None
    data = merge_labels_prompts(labels_df, prompts_df)
    texts = data["prompt"].tolist()
    gold = data["label"].tolist()

    enc = SBERTFeaturizer(args.sbert)
    X = enc.encode(texts)  # [N, F]

    Y, pca, scaler = pca_whiten(X, d=args.dims, whiten=args.whiten)
    protos = build_class_prototypes(Y, gold)

    margin, top1, sims = wdd_scores(Y, gold, protos)
    df = pd.DataFrame({
        "prompt": texts,
        "label": gold,
        "top1": top1,
        "margin": margin,
    })

    # Null calibration → choose thresholds
    null_margins = permutation_null_margins(Y, gold, protos, n_null=args.n_null)
    # simple z-score
    mu, sd = float(null_margins.mean()), float(null_margins.std() + 1e-9)
    df_stats = {
        "null_mean": mu,
        "null_std": sd,
        "z_scores_mean": float(((margin - mu) / (sd+1e-9)).mean()),
        "n_null": int(len(null_margins))
    }

    # Build threshold grid
    if args.thresholds == "auto":
        # pick z-based thresholds and raw margins
        zs = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        thr_grid = [mu + z*sd for z in zs]
        # also add empirical quantiles of margin
        qs = [0.5, 0.7, 0.8, 0.9, 0.95]
        thr_grid += list(np.quantile(margin, qs))
    else:
        thr_grid = [float(x) for x in args.thresholds.split(",")]

    # Evaluate grid
    metrics = [evaluate_at_threshold(df, thr) for thr in sorted(set(thr_grid))]
    met_df = pd.DataFrame(metrics).sort_values("threshold")
    met_path = out_dir / "wdd_metrics.csv"
    met_df.to_csv(met_path, index=False)

    # Choose threshold by simple utility: maximize overall_accuracy with abstain_rate <= 0.20
    filt = met_df[met_df["abstain_rate"] <= 0.20].copy()
    if not filt.empty:
        best_row = filt.sort_values(["overall_accuracy","accuracy_on_fired","threshold"], ascending=[False,False,True]).iloc[0].to_dict()
    else:
        best_row = met_df.sort_values(["overall_accuracy","accuracy_on_fired","threshold"], ascending=[False,False,True]).iloc[0].to_dict()

    summary = {
        "ok": True,
        "dims": args.dims,
        "whiten": bool(args.whiten),
        "stats": df_stats,
        "chosen": best_row,
        "outputs": {"metrics_csv": str(met_path)},
    }

    # Optional: compare to mapper .joblib
    if args.compare_mapper and Path(args.compare_mapper).exists():
        try:
            pipe = joblib.load(args.compare_mapper)
            # Assume pipeline.transform can handle raw texts; otherwise embed explicitly
            # If it's SBERT+LR pipeline (as in M8), pipe.predict_proba takes raw texts
            pred = pipe.predict(texts)
            acc = float((pred == np.array(gold)).mean())
            summary["compare_mapper"] = {"model": args.compare_mapper, "accuracy": acc}
        except Exception as e:
            summary["compare_mapper_error"] = str(e)

    # Save rows with margins for further analysis
    rows_path = out_dir / "wdd_rows.csv"
    df.to_csv(rows_path, index=False)

    sum_path = out_dir / "wdd_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({"ok": True, "summary": str(sum_path), "metrics": str(met_path), "rows": str(rows_path)}, indent=2))

if __name__ == "__main__":
    main()
