#!/usr/bin/env python3
"""
Audit Bench (Tier-1, tautology-free)
Thin CLI wrapper around verify.run_audit(...) that mirrors audit_bench_metrics.py outputs.

Example:
python3 -m micro_lm.domains.defi.benches.audit_bench \
  --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl \
  --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv \
  --sbert sentence-transformers/all-MiniLM-L6-v2 \
  --n_max 4 --tau_span 0.50 --tau_rel 0.60 --tau_abs 0.93 \
  --L 160 --beta 8.6 --sigma 0.0 \
  --out_dir .artifacts/defi/audit_bench \
  --competitive_eval
"""
from __future__ import annotations
import argparse, json, os, csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Local domain adapter
try:
    # When placed inside package: micro_lm/domains/defi/verify.py
    from micro_lm.domains.defi.verify import run_audit
except Exception:
    # Fallback for local dev if running as a standalone script next to verify.py
    from verify import run_audit  # type: ignore


def _read_prompts_jsonl(path: str) -> List[str]:
    P = Path(path)
    if not P.exists():
        raise FileNotFoundError(f"prompts file not found: {path}")
    out: List[str] = []
    with P.open("r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            p = (obj.get("prompt") or "").strip()
            if p:
                out.append(p)
    return out


def _read_labels_csv(path: str) -> List[str]:
    P = Path(path)
    if not P.exists():
        raise FileNotFoundError(f"labels file not found: {path}")
    out: List[str] = []
    with P.open(newline="") as f:
        r = csv.DictReader(f)
        # Expect a column named 'label'
        if "label" not in r.fieldnames:
            raise ValueError(f"labels csv must have a 'label' column; got fields={r.fieldnames}")
        for row in r:
            out.append((row.get("label") or "").strip())
    return out


def _write_rows_csv(rows: List[Dict[str, Any]], path: str) -> None:
    # Normalize a compact row schema
    # Expected keys in each row (best-effort): prompt, gold, pred, score, spans, reason, ok
    # We'll adapt to whatever run_audit returns.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Collect headers
    headers = set()
    for r in rows:
        headers.update(r.keys())
    ordered = ["prompt","gold","pred","score","ok","reason","spans","tags"]
    for h in sorted(headers):
        if h not in ordered:
            ordered.append(h)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Audit Bench — evidence-only, tautology-free")
    ap.add_argument("--prompts_jsonl", required=True, help="JSONL with {prompt: ...}")
    ap.add_argument("--labels_csv",    required=True, help="CSV with header 'label'")
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--n_max", type=int, default=4)
    ap.add_argument("--topk_per_prim", type=int, default=3)
    ap.add_argument("--tau_span", type=float, default=0.50)
    ap.add_argument("--tau_rel",  type=float, default=0.60)
    ap.add_argument("--tau_abs",  type=float, default=0.93)
    ap.add_argument("--L", type=int, default=160, help="Kaiser window length")
    ap.add_argument("--beta", type=float, default=8.6, help="Kaiser beta")
    ap.add_argument("--sigma", type=float, default=0.0, help="optional emb jitter")
    ap.add_argument("--competitive_eval", action="store_true")
    ap.add_argument("--out_dir", default=".artifacts/defi/audit_bench")
    args = ap.parse_args()

    prompts = _read_prompts_jsonl(args.prompts_jsonl)
    gold = _read_labels_csv(args.labels_csv)

    prompts = prompts[0:200]
    gold = gold[0:200]

    if len(prompts) != len(gold):
        raise ValueError(f"prompts vs labels length mismatch: {len(prompts)} != {len(gold)}")

    res = run_audit(
        prompts=prompts,
        gold_labels=gold,
        sbert_model=args.sbert,
        n_max=args.n_max,
        topk_per_prim=args.topk_per_prim,
        tau_span=args.tau_span,
        tau_rel=args.tau_rel,
        tau_abs=args.tau_abs,
        L=args.L,
        beta=args.beta,
        sigma=args.sigma,
        competitive_eval=args.competitive_eval,
    )

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = out_dir / "rows_audit.csv"
    metrics_json = out_dir / "metrics_audit.json"

    _write_rows_csv(res.get("rows", []), str(rows_csv))
    with open(metrics_json, "w") as f:
        json.dump(res.get("metrics", {}), f, indent=2)

    print(json.dumps({
        "ok": True,
        "rows": str(rows_csv),
        "metrics": str(metrics_json),
        "n": len(prompts)
    }))

if __name__ == "__main__":
    main()
