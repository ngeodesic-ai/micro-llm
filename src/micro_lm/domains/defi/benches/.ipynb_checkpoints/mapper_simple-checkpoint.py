from __future__ import annotations
import argparse, json, csv, time, sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from micro_lm.core.mapper.base import load_backend
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

"""
python3 src/micro_lm/domains/defi/benches/mapper_simple.py \
  --backend sbert \
  --model_path .artifacts/defi_mapper.joblib \
  --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl \
  --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv \
  --thresholds 0.5,0.55,0.6,0.65,0.7 \
  --min_overall_acc 0.75 \
  --out_dir .artifacts/defi/mapper_bench
"""

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

def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir or ".artifacts/defi/mapper_bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = read_prompts_jsonl(Path(args.prompts_jsonl))
    gold    = read_labels_csv(Path(args.labels_csv))

    thr = 0.5
    backend = load_backend(args.backend, confidence_threshold=thr, model_path=args.model_path)

    sub_prompts = prompts[0:10]
    preds = backend.predict(sub_prompts)
    print(preds)

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend",       default="wordmap", help="wordmap|sbert")
    ap.add_argument("--model_path",    default=".artifacts/defi_mapper.joblib")
    ap.add_argument("--prompts_jsonl", default="tests/fixtures/defi/mapper_smoke_prompts.jsonl")
    ap.add_argument("--labels_csv",    default="tests/fixtures/defi/mapper_smoke_labels.csv")
    ap.add_argument("--thresholds",    default="0.3,0.35,0.4,0.7")
    ap.add_argument("--out_dir",       default="")
    ap.add_argument("--min_overall_acc", default=None)
    
    notebook_args = [
        "--backend", "sbert",
        "--model_path", ".artifacts/defi_mapper.joblib",
        "--prompts_jsonl", "tests/fixtures/defi/defi_mapper_5k_prompts.jsonl",
        "--labels_csv", "tests/fixtures/defi/defi_mapper_labeled_5k.csv",
        "--thresholds", "0.5,0.55,0.6,0.65,0.7",
        "--min_overall_acc", "0.75",
        "--out_dir", ".artifacts/defi/mapper_bench",
    ]
    
    return ap.parse_args(notebook_args)

def main():
    args = get_args()
    raise SystemExit(run(args))

if __name__ == "__main__":
    main()
