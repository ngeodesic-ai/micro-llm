#!/usr/bin/env python3
"""
DeFi Rails Bench — Stage 6 scaffold

Purpose:
- Exercise mapper→rails integration on a fixed smoke slice.
- Emit deterministic reasons and a summary artifact.

Acceptance (from Stage-6 plan):
- No regression vs Stage-5 gate on the smoke slice.
- Deterministic verify.reasons across >=3 runs.
- Artifacts written to .artifacts/defi/rails_bench/

Docs: TIER1_FREEZE.md (Stage 5), OVERVIEW.md (structure), TIER1REFACTOR.md (stages).

python3 benches/rails_bench.py \
  --runs 3 --rails stage11 --gate_min 0.66

"""
from __future__ import annotations
import argparse, json, os, time, csv, sys
from pathlib import Path
from typing import Any, Dict, List

ALLOWED_REASONS = {"local:verified", "shim:accept:stage-10", "shim:accept:stage-11", "ltv"}

def _try_import_runner():
    try:
        from micro_lm.core.runner import run_micro  # type: ignore
        return run_micro
    except Exception as e:
        print("[rails_bench] ERROR: Could not import micro_lm.core.runner.run_micro.", file=sys.stderr)
        print("Install micro-lm package or run inside the repo with `pip install -e .`", file=sys.stderr)
        raise

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _write_json(fp: Path, obj: dict):
    fp.write_text(json.dumps(obj, indent=2))

def _write_report_md(fp: Path, summary: dict, rows: List[dict]):
    md = ["# DeFi Rails Bench — Report", "", f"Runs: {summary.get('runs', 0)}",
          f"Total cases: {summary.get('total', 0)}",
          f"ok: {summary.get('ok', 0)}",
          f"ok_acc: {summary.get('ok_acc', 0):.3f}",
          "", "## Cases"]
    for r in rows:
        md.append(f"- prompt: `{r['prompt']}` → pred=`{r.get('pred','')}` reason=`{r.get('reason','')}` ok={r.get('ok', False)}")
    fp.write_text("\n".join(md))

def _write_metrics_csv(fp: Path, rows: List[dict]):
    with fp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "pred", "reason", "ok"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ["prompt", "pred", "reason", "ok"]})

def get_smoke_prompts() -> List[str]:
    # Minimal, deterministic slice. Replace/extend with benches/defi_smoke.jsonl if desired.
    return [
        "deposit 10 ETH into aave",
        "swap 2 ETH to USDC on uniswap v3",
        "borrow 500 USDC with ETH collateral at 70% ltv",
    ]

def evaluate_rows(rows: List[dict]) -> Dict[str, Any]:
    ok = sum(1 for r in rows if r.get("ok"))
    total = len(rows)
    return {
        "ok": ok,
        "total": total,
        "ok_acc": (ok / total) if total else 0.0,
    }

def run(args: argparse.Namespace) -> int:
    run_micro = _try_import_runner()
    out_dir = Path(args.out_dir or ".artifacts/defi/rails_bench")
    _ensure_dir(out_dir)

    prompts = get_smoke_prompts()
    rows: List[dict] = []
    for i in range(args.runs):
        for p in prompts:
            res = run_micro(
                domain="defi",
                prompt=p,
                context=json.loads(args.context) if args.context else {},
                policy=json.loads(args.policy) if args.policy else {},
                rails=args.rails,
                T=args.T,
                backend=args.backend,
            )
            pred = res.get("pred", "")
            reason = (res.get("verify", {}) or {}).get("reason", "")
            # Gate: reason must be in allowed set (determinism check left to CI over multiple runs)
            if reason not in ALLOWED_REASONS:
                print(f"[rails_bench] WARN: unexpected reason '{reason}' for prompt: {p}", file=sys.stderr)
            ok = bool(res.get("ok", False))
            rows.append({"prompt": p, "pred": pred, "reason": reason, "ok": ok})

    # Aggregate & write artifacts
    summary = evaluate_rows(rows)
    summary.update({
        "bench": "rails_bench",
        "runs": args.runs,
        "rails": args.rails,
        "backend": args.backend,
        "T": args.T,
        "timestamp": int(time.time())
    })

    _write_json(out_dir / "summary.json", summary)
    _write_report_md(out_dir / "report.md", summary, rows)
    _write_metrics_csv(out_dir / "metrics.csv", rows)

    # Stage-5 gate (non-regression): ok_acc must not drop below 0.66 on this slice
    gate_min = float(args.gate_min)
    if summary["ok_acc"] < gate_min:
        print(f"[rails_bench] FAIL gate: ok_acc={summary['ok_acc']:.3f} < {gate_min}", file=sys.stderr)
        return 2
    print(f"[rails_bench] PASS gate: ok_acc={summary['ok_acc']:.3f} ≥ {gate_min}")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rails", default="stage11")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--T", type=int, default=180)
    ap.add_argument("--backend", default="sbert")
    ap.add_argument("--policy", default="")
    ap.add_argument("--context", default="")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--gate_min", default="0.66")
    args = ap.parse_args()
    raise SystemExit(run(args))

if __name__ == "__main__":
    main()
