
#!/usr/bin/env python3
"""
Milestone 10 (Audit) — Evidence-only, tautology-free verifier harness.

This spins off the audit path into a milestone-style script so you can keep
the original M10 intact. It mirrors the CLI of audit_bench but emits
milestone-like artifacts (summary + report).

Example:
python3 milestones/defi_milestone10_audit.py   --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl   --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv   --sbert sentence-transformers/all-MiniLM-L6-v2   --n_max 4 --tau_span 0.50 --tau_rel 0.60 --tau_abs 0.93   --L 160 --beta 8.6 --sigma 0.0   --out_dir .artifacts/defi_m10_audit


python3 scripts/milestones/defi_milestone10_audit.py \
  --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl \
  --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv \
  --sbert sentence-transformers/all-MiniLM-L6-v2 \
  --n_max 4 --tau_span 0.50 --tau_rel 0.60 --tau_abs 0.93 \
  --L 160 --beta 8.6 --sigma 0.0 \
  --out_dir .artifacts/defi_m10_audit \
  --competitive_eval

"""
from __future__ import annotations
import argparse, json, csv, os, sys
from pathlib import Path
from typing import List, Dict, Any

# Prefer package path; fall back to local verify.py for quick runs
try:
    from micro_lm.domains.defi.verify import run_audit  # type: ignore
except Exception:
    from verify import run_audit  # type: ignore

def _read_prompts_jsonl(path: str) -> List[str]:
    P = Path(path)
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
    out: List[str] = []
    with P.open(newline="") as f:
        r = csv.DictReader(f)
        if "label" not in r.fieldnames:
            raise ValueError(f"labels csv must have a 'label' column; got fields={r.fieldnames}")
        for row in r:
            out.append((row.get("label") or "").strip())
    return out

def _write_rows_csv(rows: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # gather headers
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

def _write_report_md(metrics: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# Milestone 10 — Audit (Tautology-Free)\n\n")
        f.write("This report reflects the evidence-only verifier (no mapper coupling).\n\n")
        keys = ["coverage","abstain_rate","span_yield_rate","abstain_no_span_rate",
                "abstain_with_span_rate","hallucination_rate","multi_accept_rate"]
        for k in keys:
            if k in metrics:
                f.write(f"- **{k}** = {metrics[k]}\n")
        f.write("\n## Params\n\n")
        for k,v in (metrics.get("params") or {}).items():
            f.write(f"- {k}: {v}\n")

def main():
    ap = argparse.ArgumentParser(description="Milestone 10 (Audit) — evidence-only verifier harness")
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--labels_csv",    required=True)
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--n_max", type=int, default=4)
    ap.add_argument("--topk_per_prim", type=int, default=3)
    ap.add_argument("--tau_span", type=float, default=0.50)
    ap.add_argument("--tau_rel",  type=float, default=0.60)
    ap.add_argument("--tau_abs",  type=float, default=0.93)
    ap.add_argument("--L", type=int, default=160)
    ap.add_argument("--beta", type=float, default=8.6)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--competitive_eval", action="store_true")
    ap.add_argument("--out_dir", default=".artifacts/defi_m10_audit")
    args = ap.parse_args()

    prompts = _read_prompts_jsonl(args.prompts_jsonl)
    gold = _read_labels_csv(args.labels_csv)

    prompts = prompts[0:20]
    gold = gold[0:20]
    
    if len(prompts) != len(gold):
        raise ValueError(f"prompts vs labels length mismatch: {len(prompts)} != {len(gold)}")

    res = run_audit(prompts=prompts, gold_labels=gold,
                    sbert_model=args.sbert, n_max=args.n_max, topk_per_prim=args.topk_per_prim,
                    tau_span=args.tau_span, tau_rel=args.tau_rel, tau_abs=args.tau_abs,
                    L=args.L, beta=args.beta, sigma=args.sigma, competitive_eval=args.competitive_eval)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows_csv    = out_dir / "rows_audit.csv"
    metrics_js  = out_dir / "metrics_audit.json"
    summary_js  = out_dir / "defi_milestone10_audit_summary.json"
    report_md   = out_dir / "defi_milestone10_audit_report.md"

    # rows + metrics
    _write_rows_csv(res.get("rows", []), str(rows_csv))
    with open(metrics_js, "w") as f:
        json.dump(res.get("metrics", {}), f, indent=2)

    # simple milestone-style summary
    M = res.get("metrics", {})
    summary = {
        "milestone": "defi_milestone10_audit",
        "ok": True,  # audit is informational; gate in CI separately if desired
        "n": len(prompts),
        "outputs": {
            "rows_csv": str(rows_csv),
            "metrics_json": str(metrics_js),
            "report_md": str(report_md)
        },
        "metrics": M
    }
    with open(summary_js, "w") as f:
        json.dump(summary, f, indent=2)

    # human report
    _write_report_md(M, str(report_md))

    print(json.dumps({
        "ok": True,
        "summary": str(summary_js),
        "report": str(report_md)
    }))

if __name__ == "__main__":
    main()
