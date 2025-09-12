#!/usr/bin/env python3
"""
Milestone-3: Denoise + Safety Guards

Objectives
- Turn on Stage-11 denoising and confirm outputs remain stable and correct.
- Verify that enabling denoiser does not change the intended top-1 primitive
  vs. a baseline run with denoiser disabled.
- Capture optional stage11 metrics (e.g., SNR, noise_floor hits) if exposed.

Gates (pass/fail)
- For each scenario:
  1) Baseline (denoise=False) top-1 is stable across N runs AND equals expected primitive.
  2) Denoised  (denoise=True ) top-1 is stable across N runs AND equals expected primitive.
  3) The denoised top-1 equals the baseline top-1 (no hallucination introduced).
- Script exits 0 and writes .artifacts/defi_milestone3_summary.json on success.

Notes
- We intentionally avoid passing denoiser hyperparameters since these may vary by your rails
  implementation. If you want knobs, add them under policy["rails"].

python3 milestones/defi_milestone3.py \
  --rails stage11 \
  --runs 5 \
  --policy '{"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7},"ltv_max":0.75}' \
  --context '{"oracle":{"age_sec":5,"max_age_sec":30}}'

python3 milestones/inspect_summary.py .artifacts/defi_milestone3_summary.json

pytest -k m3_stability_smoke -v
  
"""

from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
from typing import Any, Dict, List

from micro_llm.pipelines.runner import run_micro

ARTIFACTS_DIR = Path(".artifacts")
SUMMARY_PATH = ARTIFACTS_DIR / "defi_milestone3_summary.json"

def _hash_ctx(ctx: Dict[str, Any]) -> str:
    import json, hashlib
    return hashlib.md5(json.dumps(ctx, sort_keys=True).encode("utf-8")).hexdigest()[:8]

def _load_json_arg(arg: str | None) -> Dict[str, Any]:
    if not arg:
        return {}
    arg = arg.strip()
    if arg.startswith("{"):
        return json.loads(arg)
    p = Path(arg)
    if p.exists():
        return json.loads(p.read_text())
    raise ValueError(f"Could not parse JSON or find file: {arg}")

def _run_repeated(prompt: str, context: Dict[str, Any], policy: Dict[str, Any],
                  rails: str, T: int, runs: int) -> Dict[str, Any]:
    outs: List[Dict[str, Any]] = []
    for i in range(runs):
        res = run_micro(domain="defi", prompt=prompt, context=context, policy=policy, rails=rails, T=T)
        seq = res.get("plan", {}).get("sequence", [])
        top1 = seq[0] if seq else None
        outs.append({
            "iteration": i,
            "top1": top1,
            "flags": res.get("flags", {}),
            "verify": res.get("verify", {}),
            "aux": {
                "prior": res.get("aux", {}).get("prior"),
                "mapper_confidence": res.get("aux", {}).get("mapper_confidence"),
                "features": res.get("aux", {}).get("features"),
                "stage10": res.get("aux", {}).get("stage10"),
                "stage11": res.get("aux", {}).get("stage11"),  # optional metrics
            }
        })
    top1_list = [o["top1"] for o in outs]
    stable = len(set(top1_list)) == 1
    return {
        "prompt": prompt,
        "context_hash": _hash_ctx(context),
        "policy": policy,
        "rails": rails,
        "T": T,
        "stable_top1": stable,
        "top1_list": top1_list,
        "outputs": outs,
    }

def _eval_expect(expect: Dict[str, Any], out: Dict[str, Any], runs: int) -> (bool, str):
    kind = expect.get("kind")
    want = expect.get("primitive")
    if kind == "stable_top1_nonempty":
        if not out["stable_top1"]:
            return False, f"top-1 not stable across {runs} runs: {out['top1_list']}"
        if not out["top1_list"] or out["top1_list"][0] is None:
            return False, "top-1 is None (empty plan)"
        if want and out["top1_list"][0] != want:
            return False, f"expected top-1={want}, got={out['top1_list'][0]}"
        return True, ""
    elif kind == "nonexec_abstain":
        # For non-exec, we require empty plan and a specific flag if present
        if any(t is not None for t in out["top1_list"]):
            return False, f"expected abstain (empty top-1), got {out['top1_list']}"
        # Flags are stored per-iteration; check any iteration exposes abstain flag
        flags0 = (out.get("outputs") or [{}])[0].get("flags", {})
        if not flags0.get("abstain_non_exec", False):
            # Non-fatal: some rails may not surface this exact flag name
            return True, "note: abstain flag not surfaced"
        return True, ""
    else:
        return False, f"unknown expect.kind: {kind}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rails", default="stage11", help="rails backend (default: stage11)")
    ap.add_argument("--T", type=int, default=180, help="trace length (default: 180)")
    ap.add_argument("--runs", type=int, default=5, help="repeated runs per condition (default: 5)")
    ap.add_argument("--policy", default=None, help="policy JSON (or path)")
    ap.add_argument("--context", default=None, help="context JSON (or path) applied to ALL cases")
    args = ap.parse_args()

    base_policy = _load_json_arg(args.policy)
    base_context = _load_json_arg(args.context)

    # Ensure mapper defaults (so it "just works")
    base_policy.setdefault("mapper", {})
    base_policy["mapper"].setdefault("model_path", ".artifacts/defi_mapper.joblib")
    base_policy["mapper"].setdefault("confidence_threshold", 0.7)

    # Canonical scenarios with expected primitives
    scenarios = [
        {
            "name": "deposit_eth",
            "prompt": "deposit 10 ETH into aave",
            "expect": {"kind": "stable_top1_nonempty", "primitive": "deposit_asset"},
            "context": {"risk":{"hf":1.22}, "oracle":{"age_sec":5,"max_age_sec":30}},
            "policy": {"ltv_max":0.75},
        },
        {
            "name": "swap_eth_usdc",
            "prompt": "swap 2 ETH for USDC",
            "expect": {"kind": "stable_top1_nonempty", "primitive": "swap_asset"},
            "context": {"oracle":{"age_sec":5,"max_age_sec":30}},
            "policy": {"ltv_max":0.75},
        },
        # Keep a non-exec probe to ensure abstain still abstains under denoise
        {
            "name": "nonexec_abstain",
            "prompt": "check balance",
            "expect": {"kind": "nonexec_abstain"},
            "context": {"oracle":{"age_sec":5,"max_age_sec":30}},
            "policy": {"ltv_max":0.75},
        },
    ]

    results = []
    failures = []

    for sc in scenarios:
        prompt = sc["prompt"]
        merged_ctx = {**base_context, **sc.get("context", {})}

        # Baseline (denoise=False)
        pol_base = {**base_policy, **sc.get("policy", {})}
        pol_base.setdefault("rails", {})
        pol_base["rails"]["denoise"] = False

        out_base = _run_repeated(prompt, merged_ctx, pol_base, rails=args.rails, T=args.T, runs=args.runs)
        ok_base, reason_base = _eval_expect(sc["expect"], out_base, args.runs)

        # Treatment (denoise=True)
        pol_dn = {**base_policy, **sc.get("policy", {})}
        pol_dn.setdefault("rails", {})
        pol_dn["rails"]["denoise"] = True

        out_dn = _run_repeated(prompt, merged_ctx, pol_dn, rails=args.rails, T=args.T, runs=args.runs)
        ok_dn, reason_dn = _eval_expect(sc["expect"], out_dn, args.runs)

        # Consistency gate: if both are executable, require same top-1
        consistent = True
        reason_consistency = ""
        if sc["expect"]["kind"] == "stable_top1_nonempty":
            if out_base["top1_list"] and out_dn["top1_list"]:
                if out_base["top1_list"][0] != out_dn["top1_list"][0]:
                    consistent = False
                    reason_consistency = f"mismatch base={out_base['top1_list'][0]} vs denoise={out_dn['top1_list'][0]}"

        ok = ok_base and ok_dn and consistent
        reason = "; ".join([r for r in [reason_base, reason_dn, reason_consistency] if r])

        results.append({
            "name": sc["name"],
            "ok": ok,
            "reason": reason,
            "baseline": out_base,
            "denoised": out_dn,
        })
        if not ok:
            failures.append((sc["name"], reason))

    # Build milestone report
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "milestone": "defi_milestone3",
        "status": "pass" if not failures else "fail",
        "rails": args.rails,
        "T": args.T,
        "runs": args.runs,
        "scenarios": results,
    }
    SUMMARY_PATH.write_text(json.dumps(report, indent=2))

    if failures:
        print(json.dumps({"ok": False, "summary": str(SUMMARY_PATH), "failures": failures}, indent=2))
        sys.exit(1)

    print(json.dumps({"ok": True, "summary": str(SUMMARY_PATH)}, indent=2))
    sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(1)
