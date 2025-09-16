#!/usr/bin/env python3
"""
Milestone 7 — Mapper Threshold Sweep (Evaluation Only)

Purpose
-------
Given a trained mapper (e.g., from M6) and a set of evaluation prompts
(optionally with labels), sweep confidence thresholds and compute:
- abstain_rate, coverage
- (if labels provided) accuracy_on_fired, overall_accuracy

No training happens here. This script only:
  load_mapper -> score prompts -> sweep thresholds -> emit summary/report/csv

PYTHONPATH=. python3 milestones/train_mapper_embed.py \
  --labels_csv tests/fixtures/defi_mapper_labeled_large.csv \
  --out_path .artifacts/defi_mapper_embed.joblib \
  --sbert sentence-transformers/all-mpnet-base-v2 \
  --C 8 --max_iter 2000 --calibrate

python3 milestones/defi_milestone7.py \
  --mapper_path .artifacts/defi_mapper_embed.joblib \
  --prompts_jsonl tests/fixtures/defi_mapper_stress_prompts.jsonl \
  --labels_csv  tests/fixtures/defi_mapper_stress_labeled.csv \
  --thresholds "0.2,0.25,0.3,0.35,0.4" \
  --max_abstain_rate 0.20 \
  --min_overall_acc 0.85 \
  --choose_by utility \
  --per_class_thresholds .artifacts/m7_per_class_thresholds.json \
  --rows_csv .artifacts/m7_rows.csv \
  --out_summary .artifacts/m7_sum.json \
  --out_csv .artifacts/m7_metrics.csv

column -s, -t < .artifacts/m7_metrics.csv | sed -n '1,12p'
awk -F, 'NR==1 || ($5=="False" && $2!=$3)' .artifacts/m7_rows.csv | sed -n '1,20p'
jq '.chosen' .artifacts/defi_milestone7_summary.json
"""

import argparse, csv, json, os, sys, time
from typing import Any, Dict, List

try:
    import joblib
except Exception:
    joblib = None

DEFAULT_CLASSES = ["deposit_asset", "withdraw_asset", "swap_asset", "check_balance"]


# --- shim for older artifacts trained with __main__.SBERTEncoder ---
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class SBERTEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 batch_size=64, normalize=True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    def fit(self, X, y=None):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        embs = self._model.encode(
            list(X),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(embs)
# -------------------------------------------------------------------


# -------------------------- I/O helpers --------------------------

def load_mapper(path: str):
    if not joblib:
        raise RuntimeError("joblib not available; please install it to load the mapper.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mapper not found: {path}")
    return joblib.load(path)


def read_prompts_jsonl(path: str) -> List[str]:
    prompts: List[str] = []
    if path == "-":
        for line in sys.stdin:
            try:
                rec = json.loads(line)
                p = rec.get("prompt", "").strip()
                if p:
                    prompts.append(p)
            except Exception:
                continue
        return prompts
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                p = rec.get("prompt", "").strip()
                if p:
                    prompts.append(p)
            except Exception:
                continue
    return prompts


def read_labels_csv(path: str) -> Dict[str, str]:
    """
    Return mapping: prompt -> gold_label
    CSV columns expected: prompt,label
    """
    gold: Dict[str, str] = {}
    if not path:
        return gold
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        import pandas as pd
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            p = str(row["prompt"]).strip()
            y = str(row["label"]).strip()
            if p:
                gold[p] = y
        return gold
    except Exception:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for rec in reader:
                p = str(rec.get("prompt", "")).strip()
                y = str(rec.get("label", "")).strip()
                if p:
                    gold[p] = y
        return gold


# -------------------------- Scoring --------------------------

def mapper_predict_scores(mapper, prompts: List[str], class_names: List[str]) -> List[Dict[str, float]]:
    """
    Returns per-prompt score dicts (class -> score). Uses predict_proba if available,
    else decision_function -> softmax; else predict -> 1.0 for predicted class.
    """
    import numpy as np
    scores: List[Dict[str, float]] = []

    if hasattr(mapper, "predict_proba"):
        probs = mapper.predict_proba(prompts)
        classes = list(getattr(mapper, "classes_", class_names))
        for row in probs:
            row_map = {str(c): float(p) for c, p in zip(classes, row)}
            for cname in class_names:
                row_map.setdefault(cname, 0.0)
            scores.append(row_map)
        return scores

    if hasattr(mapper, "decision_function"):
        logits = mapper.decision_function(prompts)
        logits = np.array(logits, dtype=float)
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)
        classes = list(getattr(mapper, "classes_", class_names))
        for row in logits:
            row = np.array(row, dtype=float)
            ex = np.exp(row - row.max())
            prob = ex / (ex.sum() + 1e-12)
            row_map = {str(c): float(p) for c, p in zip(classes, prob)}
            for cname in class_names:
                row_map.setdefault(cname, 0.0)
            scores.append(row_map)
        return scores

    # Fallback: predict-only
    preds = mapper.predict(prompts)
    for y in preds:
        row_map = {c: 0.0 for c in class_names}
        row_map[str(y)] = 1.0
        scores.append(row_map)
    return scores


def top1_sequence(score_map: Dict[str, float], class_names: List[str], thr: float) -> List[str]:
    top = max(class_names, key=lambda c: score_map.get(c, 0.0))
    conf = score_map.get(top, 0.0)
    return [top] if conf >= thr else []

def write_rows_csv(prompts, scores, class_names, thr, gold, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt","gold_label","predicted","confidence","abstain","threshold"])
        for p, smap in zip(prompts, scores):
            top = max(class_names, key=lambda c: smap.get(c, 0.0))
            conf = float(smap.get(top, 0.0))
            fire = conf >= thr
            pred = top if fire else ""
            w.writerow([p, gold.get(p, ""), pred, f"{conf:.4f}", (not fire), thr])

def metrics_for_threshold(
    prompts: List[str],
    scores: List[Dict[str, float]],
    class_names: List[str],
    thr: float,
    gold: Dict[str, str],
    per_thr: Dict[str, float] | None = None,   # <— new
) -> Dict[str, Any]:
    total = len(prompts)
    abstain = 0
    fired = 0
    correct = 0
    for p, smap in zip(prompts, scores):
        # top-1 predicted class and its confidence
        top = max(class_names, key=lambda c: smap.get(c, 0.0))
        score = smap.get(top, 0.0)
    
        # per-class threshold override (fallback to global)
        eff_thr = (per_thr or {}).get(top, thr)
    
        # fire vs abstain using the effective threshold
        abstain = (score < eff_thr)
        seq = top1_sequence(smap, class_names, thr)
        if not seq:
            abstain += 1
        else:
            fired += 1
            if p in gold and gold[p] == seq[0]:
                correct += 1
    abstain_rate = abstain / max(1, total)
    coverage = fired / max(1, total)
    acc_on_fired = (correct / max(1, fired)) if fired else None
    overall_acc = correct / max(1, total)
    return {
        "threshold": thr,
        "total": total,
        "abstain": abstain,
        "abstain_rate": abstain_rate,
        "coverage": coverage,
        "fired": fired,
        "correct_on_fired": correct,
        "accuracy_on_fired": acc_on_fired,
        "overall_correct": correct,
        "overall_accuracy": overall_acc,
    }


# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser(description="Milestone 7 — Threshold sweep (evaluation only)")
    ap.add_argument("--mapper_path", type=str, required=True, help="Path to trained mapper (.joblib)")
    ap.add_argument("--prompts_jsonl", type=str, required=True, help="JSONL with {prompt: ...} per line (or '-' for stdin)")
    ap.add_argument("--labels_csv", type=str, default="", help="Optional CSV with columns: prompt,label")
    ap.add_argument("--class_names", type=str, default=",".join(DEFAULT_CLASSES), help="Comma-separated class names")
    ap.add_argument("--thresholds", type=str, default="0.5,0.6,0.7,0.8,0.9", help="Comma-separated confidence thresholds to sweep")
    ap.add_argument("--max_abstain_rate", type=float, default=0.10, help="Pass if chosen abstain_rate ≤ this")
    ap.add_argument("--min_overall_acc", type=float, default=0.85, help="(If labels) also require overall_accuracy ≥ this")
    ap.add_argument("--choose_by", type=str, default="abstain_then_acc", choices=["abstain_then_acc","utility"],
            help="abstain_then_acc = pick highest thr with abstain≤cut; utility = maximize harmonic mean of coverage acc_on_fired")

    ap.add_argument("--out_summary", type=str, default=".artifacts/defi_milestone7_summary.json")
    ap.add_argument("--out_report", type=str, default=".artifacts/defi_milestone7_report.md")
    ap.add_argument("--out_csv", type=str, default=".artifacts/defi_milestone7_metrics.csv")
    ap.add_argument("--rows_csv", type=str, default=".artifacts/defi_milestone7_rows.csv",
                help="Write per-prompt results at the chosen threshold")
    ap.add_argument(
        "--per_class_thresholds",
        type=str,
        default="",
        help="Optional path to JSON mapping {class_name: threshold} to override the global threshold per class"
    )
    args = ap.parse_args()

    # Ensure output dirs
    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)
    if args.out_report:
        os.makedirs(os.path.dirname(args.out_report) or ".", exist_ok=True)
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    # Load artifacts
    mapper = load_mapper(args.mapper_path)
    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    prompts = read_prompts_jsonl(args.prompts_jsonl)
    if not prompts:
        print("No prompts loaded.", file=sys.stderr)
        sys.exit(2)
    gold = read_labels_csv(args.labels_csv) if args.labels_csv else {}

    per_thr = {}
    if args.per_class_thresholds:
        with open(args.per_class_thresholds, "r") as f:
            per_thr = json.load(f)

    # Score once
    scores = mapper_predict_scores(mapper, prompts, class_names)

    # Sweep thresholds
    thr_list = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    metrics = [metrics_for_threshold(prompts, scores, class_names, thr, gold) for thr in thr_list]

    thr_list = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    metrics = [
        metrics_for_threshold(prompts, scores, class_names, thr, gold, per_thr)
        for thr in thr_list
    ]
    
    # Choose operating point: highest thr with abstain_rate <= 0.10; else minimal abstain
    cut = args.max_abstain_rate
    admissible = [m for m in metrics if m["abstain_rate"] <= cut]
    if admissible:
        admissible.sort(key=lambda m: (m["threshold"], m.get("accuracy_on_fired") or 0.0), reverse=True)
        chosen = admissible[0]
    else:
        if args.choose_by == "utility":
            def util(m):
                a = (m["accuracy_on_fired"] or 0.0)
                c = m["coverage"]
                return 0.0 if a == 0 or c == 0 else 2*a*c/(a+c)  # harmonic mean
            chosen = max(metrics, key=util)
        else:
            metrics.sort(key=lambda m: (m["abstain_rate"], -(m.get("accuracy_on_fired") or 0.0)))
            chosen = metrics[0]
    # Decide milestone status
    has_labels = bool(gold)
    pass_abstain = (chosen["abstain_rate"] <= args.max_abstain_rate)
    pass_accuracy = True if not has_labels else ((chosen.get("overall_accuracy") or 0.0) >= args.min_overall_acc)
    status = "pass" if (pass_abstain and pass_accuracy) else "fail"

    # Write outputs
    summary = {
        "ok": True,
        "milestone": "defi_milestone7",
        "timestamp": int(time.time()),
        "prompts": len(prompts),
        "has_labels": has_labels,
        "thresholds": thr_list,
        "metrics": metrics,
        "chosen": {
            "threshold": chosen["threshold"],
            "abstain_rate": chosen["abstain_rate"],
            "coverage": chosen["coverage"],
            "accuracy_on_fired": chosen.get("accuracy_on_fired"),
            "overall_accuracy": chosen.get("overall_accuracy"),
        },
        "status": status,
        "rails": None
    }

    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["threshold","total","abstain","abstain_rate","coverage","fired","correct_on_fired","accuracy_on_fired","overall_correct","overall_accuracy"]
            writer.writerow(header)
            for m in metrics:
                writer.writerow([m.get(h) for h in header])

    if args.out_report:
        lines = []
        lines.append("# Milestone 7 — Mapper Threshold Sweep (Eval)")
        lines.append(f"- Prompts: **{len(prompts)}**")
        lines.append(f"- Labels provided: **{has_labels}**")
        lines.append("")
        lines.append("## Per-threshold metrics")
        lines.append("| thr | abstain_rate | coverage | acc_on_fired | overall_acc |")
        lines.append("|---:|---:|---:|---:|---:|")
        for m in metrics:
            ar = f"{m['abstain_rate']:.3f}"
            cov = f"{m['coverage']:.3f}"
            aof = "—" if m["accuracy_on_fired"] is None else f"{m['accuracy_on_fired']:.3f}"
            ovr = f"{m['overall_accuracy']:.3f}"
            lines.append(f"| {m['threshold']:.2f} | {ar} | {cov} | {aof} | {ovr} |")
        lines.append("")
        lines.append("## Chosen operating point")
        lines.append(f"- **threshold:** `{chosen['threshold']:.2f}`")
        lines.append(f"- **abstain_rate:** `{chosen['abstain_rate']:.3f}`")
        lines.append(f"- **coverage:** `{chosen['coverage']:.3f}`")
        if chosen.get("accuracy_on_fired") is not None:
            lines.append(f"- **accuracy_on_fired:** `{chosen['accuracy_on_fired']:.3f}`")
            lines.append(f"- **overall_accuracy:** `{chosen['overall_accuracy']:.3f}`")
        with open(args.out_report, "w") as f:
            f.write("\n".join(lines))

    # Rows CSV at chosen threshold (easy failure drilldown)
    if args.rows_csv:
        write_rows_csv(prompts, scores, class_names, chosen["threshold"], gold, args.rows_csv)



    # Friendly stdout
    print(json.dumps({"ok": True, "summary": args.out_summary, "report": args.out_report,
                       "csv": args.out_csv, "rows_csv": args.rows_csv, "status": status}, indent=2))

    # Optional CI gate: fail process if status == "fail"
    # sys.exit(0 if status == "pass" else 1)


if __name__ == "__main__":
    main()
