#!/usr/bin/env python3
"""
Milestone-1: Hybrid Mapper + Prior Injection

Objective
- Use trained mapper (joblib) + rule fallback to map prompt → primitive slots.
- Feed mapper confidence into adapter priors.
- Ensure Stage-10 ordering respects priors (top-1 deterministic on clean prompts).

What this script asserts
- Canonical prompts map to the intended primitive as top-1.
- Non-exec/unsupported prompts abstain (or mark an abstain flag).
- Optional visibility checks for aux fields (prior, mapper_confidence) if present.

Pass/Fail gates
- All scenarios satisfy their expected "kind" ("top1", "abstain").
- Script exits 0 and writes .artifacts/defi_milestone1_summary.json

# from repo root
python3 milestones/defi_milestone1.py \
  --rails stage11 \
  --policy '{"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7},"ltv_max":0.75}'

pytest -v tests/test_m1_stability_smoke.py

"""

from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
from typing import Any, Dict

# Assumes micro_llm is installed in editable mode or on PYTHONPATH
from micro_llm.pipelines.runner import run_micro

ARTIFACTS_DIR = Path(".artifacts")
SUMMARY_PATH = ARTIFACTS_DIR / "defi_milestone1_summary.json"

def _hash_ctx(ctx: Dict[str, Any]) -> str:
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

def run_case(prompt: str, context: Dict[str, Any], policy: Dict[str, Any], rails: str, T: int) -> Dict[str, Any]:
    res = run_micro(domain="defi", prompt=prompt, context=context, policy=policy, rails=rails, T=T)
    # Shape result a bit for the milestone report
    seq = res.get("plan", {}).get("sequence", [])
    top1 = seq[0] if seq else None
    return {
        "prompt": prompt,
        "context_hash": _hash_ctx(context),
        "policy": policy,
        "rails": rails,
        "T": T,
        "top1": top1,
        "flags": res.get("flags", {}),
        "verify": res.get("verify", {}),
        "aux": {
            # Optional; only present if the package exposes these
            "prior": res.get("aux", {}).get("prior"),
            "mapper_confidence": res.get("aux", {}).get("mapper_confidence"),
            "features": res.get("aux", {}).get("features"),
            "stage10": res.get("aux", {}).get("stage10"),
        }
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rails", default="stage11", help="rails backend (default: stage11)")
    ap.add_argument("--T", type=int, default=180, help="trace length (default: 180)")
    ap.add_argument("--policy", default=None, help='policy JSON (or path). Example: \'{"ltv_max":0.75,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}\'')
    ap.add_argument("--context", default=None, help="context JSON (or path) applied to ALL cases")
    args = ap.parse_args()

    base_policy = _load_json_arg(args.policy)
    base_context = _load_json_arg(args.context)

    # Canonical scenarios for milestone-1 (simple and crisp)
    scenarios = [
        {
            "name": "deposit_eth",
            "prompt": "deposit 10 ETH into aave",
            "expect": {"kind": "top1", "primitive": "deposit_asset"},
            "context": {"risk":{"hf":1.22}, "oracle":{"age_sec":5,"max_age_sec":30}},
            "policy": {"ltv_max":0.75, "mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}},
        },
        {
            "name": "nonexec_abstain",
            "prompt": "check balance",
            "expect": {"kind": "abstain"},
            "context": {"oracle":{"age_sec":5,"max_age_sec":30}},
            "policy": {"ltv_max":0.75, "mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}},
        },
        # Optional: add more crisp mapper checks if you’ve trained the model on paraphrases
        # {
        #     "name": "swap_eth_usdc",
        #     "prompt": "swap 2 ETH for USDC",
        #     "expect": {"kind": "top1", "primitive": "swap_asset"},
        #     "context": {"oracle":{"age_sec":5,"max_age_sec":30}},
        #     "policy": {"ltv_max":0.75, "mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}},
        # },
    ]

    results = []
    failures = []

    for sc in scenarios:
        prompt = sc["prompt"]
        merged_ctx = {**base_context, **sc.get("context", {})}
        merged_pol = {**base_policy, **sc.get("policy", {})}
        out = run_case(prompt, merged_ctx, merged_pol, rails=args.rails, T=args.T)

        # Evaluate expectations
        expect = sc["expect"]
        kind = expect["kind"]
        ok = True
        reason = ""

        if kind == "top1":
            want = expect["primitive"]
            got = out["top1"]
            if got != want:
                ok = False
                reason = f"expected top1={want}, got={got}"
            # Optional visibility checks (won’t fail if missing):
            # - if prior exists, ensure it favors the intended primitive
            prior = out["aux"].get("prior")
            if ok and isinstance(prior, dict) and want in prior:
                if prior.get(want, 0.0) <= 0.0:
                    ok = False
                    reason = f"prior does not favor {want}"
        elif kind == "abstain":
            # We accept either an empty plan or a flag/verify indicating abstain
            is_empty_plan = out["top1"] is None
            flags = out.get("flags", {})
            verify = out.get("verify", {})
            abstainish = is_empty_plan or flags.get("abstain_non_exec") or (verify.get("ok") is False)
            if not abstainish:
                ok = False
                reason = "expected abstain-like outcome (empty plan / abstain flag / verify not ok)"
        else:
            ok = False
            reason = f"unknown expect.kind: {kind}"

        results.append({
            "name": sc["name"],
            "ok": ok,
            "reason": reason,
            "output": out,
        })
        if not ok:
            failures.append((sc["name"], reason))

    # Build milestone report
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "milestone": "defi_milestone1",
        "status": "pass" if not failures else "fail",
        "rails": args.rails,
        "T": args.T,
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
