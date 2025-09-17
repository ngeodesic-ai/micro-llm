#!/usr/bin/env python3
"""
Mapper Bench — Stage 7
- Swappable backends: --backend wordmap|sbert
- Threshold grid and metrics (acc, abstain rate, coverage)
- Emits: summary.json, metrics.csv, rows.csv, report.md under .artifacts/defi/mapper_bench/
"""
from __future__ import annotations
import argparse, json, csv, time, sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from micro_lm.core.mapper.base import load_backend

def read_prompts_jsonl(fp: Path) -> List[str]:
    rows=[]
    with fp.open() as f:
        for line in f:
            J=json.loads(line)
            rows.append(J["prompt"] if isinstance(J, dict) and "prompt" in J else line.strip())
    return rows

def read_labels_csv(fp: Path) -> List[str]:
    out=[]
    with fp.open() as f:
        R=csv.DictReader(f)
        for r in R:
            out.append(r["label"])
    return out

def write_rows_csv(fp: Path, prompts: List[str], gold: List[str], preds: List[Tuple[str|None,float]], thr: float):
    with fp.open("w", newline="") as f:
        W = csv.writer(f)
        W.writerow(["prompt","gold_label","predicted","confidence","abstain","threshold"])
        for p,g,(lab,conf) in zip(prompts, gold, preds):
            W.writerow([p,g,lab or "", f"{conf:.4f}", str(lab is None), thr])

def compute_metrics(gold: List[str], preds: List[Tuple[str|None,float]]) -> Dict[str,Any]:
    total = len(gold)
    abstain = sum(1 for (lab,_) in preds if lab is None)
    fired = total - abstain
    correct = sum(1 for g,(lab,_) in zip(gold,preds) if lab==g)
    acc_on_fired = (correct / fired) if fired else 0.0
    overall_acc = (correct / total) if total else 0.0
    return {"total": total, "abstain": bool(abstain), "abstain_count": abstain,
            "abstain_rate": abstain/total if total else 0.0,
            "coverage": fired/total if total else 0.0,
            "fired": fired, "correct_on_fired": correct, "accuracy_on_fired": acc_on_fired,
            "overall_correct": correct, "overall_accuracy": overall_acc}

def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir or ".artifacts/defi/mapper_bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = read_prompts_jsonl(Path(args.prompts_jsonl))
    gold    = read_labels_csv(Path(args.labels_csv))
    assert len(prompts)==len(gold), "prompts and labels must align"

    metrics_rows = []
    chosen = None
    best_util = -1e9

    thrs = [float(x) for x in (args.thresholds.split(",") if args.thresholds else ["0.35","0.7"])]
    for thr in thrs:
        backend = load_backend(args.backend, confidence_threshold=thr, model_path=args.model_path)
        preds = backend.predict(prompts)
        M = compute_metrics(gold, preds)
        M.update({"threshold": thr})
        metrics_rows.append(M)

        util = (M["overall_accuracy"] * 100.0) - (M["abstain_rate"] * 10.0)
        if util > best_util:
            best_util = util
            chosen = {"threshold": thr,
                      "abstain_rate": M["abstain_rate"],
                      "coverage": M["coverage"],
                      "accuracy_on_fired": M["accuracy_on_fired"],
                      "overall_accuracy": M["overall_accuracy"]}
        # Write per-threshold rows for inspection
        write_rows_csv(out_dir / f"rows_thr_{thr}.csv", prompts, gold, preds, thr)

    # Write metrics.csv
    with (out_dir/"metrics.csv").open("w", newline="") as f:
        W = csv.DictWriter(f, fieldnames=["threshold","total","abstain","abstain_count","abstain_rate","coverage","fired","correct_on_fired","accuracy_on_fired","overall_correct","overall_accuracy"])
        W.writeheader()
        for r in metrics_rows:
            W.writerow(r)

    summary = {"bench":"mapper_bench","backend":args.backend,"thresholds":thrs,
               "chosen":chosen, "timestamp": int(time.time())}
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))

    # Simple report
    lines = ["# Mapper Bench — Report","","Backend: "+args.backend,""]
    for r in metrics_rows:
        lines.append(f"- thr={r['threshold']:.2f} overall_acc={r['overall_accuracy']:.4f} abstain_rate={r['abstain_rate']:.4f} coverage={r['coverage']:.4f}")
    (out_dir/"report.md").write_text("\n".join(lines))

    # Gate (optional)
    if args.min_overall_acc is not None:
        if chosen and chosen["overall_accuracy"] < float(args.min_overall_acc):
            print(f"[mapper_bench] FAIL gate: overall_acc={chosen['overall_accuracy']:.3f} < {args.min_overall_acc}", file=sys.stderr)
            return 2
    print(f"[mapper_bench] PASS. chosen={json.dumps(chosen)}")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="wordmap", help="wordmap|sbert")
    ap.add_argument("--model_path", default=".artifacts/defi_mapper.joblib")
    ap.add_argument("--prompts_jsonl", default="tests/fixtures/defi/mapper_smoke_prompts.jsonl")
    ap.add_argument("--labels_csv",    default="tests/fixtures/defi/mapper_smoke_labels.csv")
    ap.add_argument("--thresholds",    default="0.3,0.35,0.4,0.7")
    ap.add_argument("--out_dir",       default="")
    ap.add_argument("--min_overall_acc", default=None)
    args = ap.parse_args()
    raise SystemExit(run(args))

if __name__ == "__main__":
    main()
