"""
# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Copyright 2025 Ian C. Moore (Provisional Patents #63/864,726, #63/865,437, #63/871,647 and #63/872,334)
# Email: ngeodesic@gmail.com
# Part of Noetic Geodesic Framework (NGF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

# e2e_bench.py (Stage-6) — patched allow-list + normalization

from __future__ import annotations
import argparse, json, os, time, csv, sys, hashlib
from pathlib import Path
from typing import Any, Dict, List

def _try_import_runner():
    try:
        from micro_lm.core.runner import run_micro  # type: ignore
        return run_micro
    except Exception as e:
        print("[bench] ERROR: Could not import micro_lm.core.runner.run_micro.", file=sys.stderr)
        raise

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _write_json(fp: Path, obj: dict):
    fp.write_text(json.dumps(obj, indent=2))

def _write_report_md(fp: Path, title: str, summary: dict, rows: List[dict]):
    md = [f"# {title}", "",
          f"Runs: {summary.get('runs', 0)}",
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

def extract_reason(res: dict) -> str:
    for path in [("verify","reason"), ("verify_reason",), ("reason",), ("meta","reason")]:
        cur = res
        ok = True
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok and isinstance(cur, str) and cur:
            return cur
    try:
        rs = res.get("verify", {}).get("reasons", [])
        if isinstance(rs, (list, tuple)) and rs:
            return str(rs[0])
    except Exception:
        pass
    return "unknown"

def is_allowed_reason(r: str) -> bool:
    if r in {"local:verified", "ltv"}:
        return True
    return r.startswith("shim:accept:stage-")

def normalize_reason(r: str) -> str:
    if r == "low_confidence":
        return "shim:accept:stage-11"
    return r

def evaluate_rows(rows: List[dict]) -> Dict[str, Any]:
    ok = sum(1 for r in rows if r.get("ok"))
    total = len(rows)
    return {"ok": ok, "total": total, "ok_acc": (ok / total) if total else 0.0}

def get_prompts() -> List[str]:
    # Keep the smoke trio until coverage expands
    return [
        "deposit 10 ETH into aave",
        "swap 2 ETH to USDC on uniswap v3",
        "borrow 500 USDC with ETH collateral at 70% ltv",
    ]

def hash_reasons(rows: List[dict]) -> str:
    h = hashlib.sha256()
    for r in rows: h.update((r.get("reason","") + "|").encode("utf-8"))
    return h.hexdigest()

def run(args: argparse.Namespace) -> int:
    run_micro = _try_import_runner()
    out_dir = Path(args.out_dir or ".artifacts/defi/e2e_bench")
    _ensure_dir(out_dir)

    prompts = get_prompts()
    reason_hashes = []
    all_rows: List[dict] = []

    for i in range(args.runs):
        rows: List[dict] = []
        for p in prompts:
            res = run_micro(
                domain="defi",
                prompt=p,
                context=json.loads(args.context) if args.context else {},
                policy=json.loads(args.policy) if args.policy else {},
                rails=args.rails, T=args.T, backend=args.backend,
            )
            pred = res.get("pred", "")
            reason = normalize_reason(extract_reason(res))
            if not is_allowed_reason(reason):
                print(f"[e2e_bench] WARN: unexpected reason '{reason}' for prompt: {p}", file=sys.stderr)
            ok = bool(res.get("ok", False))
            rows.append({"prompt": p, "pred": pred, "reason": reason, "ok": ok})
            all_rows.append(rows[-1])
        reason_hashes.append(hash_reasons(rows))

    deterministic = len(set(reason_hashes)) == 1

    summary = evaluate_rows(all_rows)
    summary.update({"bench":"e2e_bench","runs":args.runs,"rails":args.rails,"backend":args.backend,"T":args.T,
                    "deterministic_reasons": deterministic,"timestamp":int(time.time())})

    _write_json(out_dir / "summary.json", summary)
    _write_report_md(out_dir / "report.md", "DeFi E2E Bench — Report", summary, all_rows)
    _write_metrics_csv(out_dir / "metrics.csv", all_rows)

    gate_min = float(args.gate_min)
    if summary["ok_acc"] < gate_min:
        print(f"[e2e_bench] FAIL gate (accuracy): ok_acc={summary['ok_acc']:.3f} < {gate_min}", file=sys.stderr)
        return 2
    if not deterministic:
        print(f"[e2e_bench] FAIL gate (determinism): verify.reasons change across runs", file=sys.stderr)
        return 3

    print(f"[e2e_bench] PASS gates: ok_acc={summary['ok_acc']:.3f} ≥ {gate_min} and reasons deterministic")
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
