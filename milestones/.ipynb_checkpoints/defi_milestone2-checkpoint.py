#!/usr/bin/env python3
"""
Milestone-2: Stage-11 Warp + Detect (no denoiser)

Objective
- Exercise the NGF Stage-11 *Warp → Detect* path (denoiser off).
- Show stable well identification (top-1 primitive) across repeated runs.
- Keep the same prompt→adapter→rails interface as M1.

What this script asserts
- For each canonical scenario, top-1 primitive is identical across N repeated runs.
- Optional: if rails expose warp/detect metrics in aux, capture them (non-fatal).

Pass/Fail gates
- All scenarios satisfy "stable_top1" (same top-1 across runs).
- Script exits 0 and writes .artifacts/defi_milestone2_summary.json

python3 milestones/defi_milestone2.py \
  --rails stage11 \
  --runs 5 \
  --policy '{"rails":{"denoise": false},"ltv_max":0.75}' \
  --context '{"oracle":{"age_sec":5,"max_age_sec":30}}'

python3 milestones/inspect_summary.py .artifacts/defi_milestone2_summary.json
Milestone: defi_milestone2
Status:    pass
Rails:     stage11
T:         180
runs:      5

- deposit_eth: ok=True
  prompt: deposit 10 ETH into aave
  top1_list: ['deposit_asset', 'deposit_asset', 'deposit_asset', 'deposit_asset', 'deposit_asset']
  stable_top1: True
  top1(first): deposit_asset

- swap_eth_usdc: ok=True
  prompt: swap 2 ETH for USDC
  top1_list: ['swap_asset', 'swap_asset', 'swap_asset', 'swap_asset', 'swap_asset']
  stable_top1: True
  top1(first): swap_asset

pytest -k m2_stability_smoke -v

"""

from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
from typing import Any, Dict, List

# Assumes micro_llm is installed in editable mode or on PYTHONPATH
from micro_llm.pipelines.runner import run_micro

ARTIFACTS_DIR = Path(".artifacts")
SUMMARY_PATH = ARTIFACTS_DIR / "defi_milestone2_summary.json"

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

def run_case(prompt: str, context: Dict[str, Any], policy: Dict[str, Any],
             rails: str, T: int, runs: int) -> Dict[str, Any]:
    """Run the same prompt multiple times and check top-1 stability."""
    outputs: List[Dict[str, Any]] = []
    for i in range(runs):
        res = run_micro(domain="defi", prompt=prompt, context=context, policy=policy, rails=rails, T=T)
        seq = res.get("plan", {}).get("sequence", [])
        top1 = seq[0] if seq else None
        outputs.append({
            "iteration": i,
            "top1": top1,
            "flags": res.get("flags", {}),
            "verify": res.get("verify", {}),
            "aux": {
                # These are optional; only recorded if present
                "prior": res.get("aux", {}).get("prior"),
                "mapper_confidence": res.get("aux", {}).get("mapper_confidence"),
                "features": res.get("aux", {}).get("features"),
                "stage10": res.get("aux", {}).get("stage10"),
                "stage11": res.get("aux", {}).get("stage11"),
            }
        })
    # Determine stability
    top1_list = [o["top1"] for o in outputs]
    stable = len(set(top1_list)) == 1
    return {
        "prompt": prompt,
        "context_hash": _hash_ctx(context),
        "policy": policy,
        "rails": rails,
        "T": T,
        "stable_top1": stable,
        "top1_list": top1_list,
        "outputs": outputs,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rails", default="stage11", help="rails backend (default: stage11)")
    ap.add_argument("--T", type=int, default=180, help="trace length (default: 180)")
    ap.add_argument("--runs", type=int, default=5, help="repeated runs per scenario (default: 5)")
    ap.add_argument("--policy", default=None,
                    help='policy JSON (or path). Example: \'{"rails":{"denoise": false},"ltv_max":0.75}\'')
    ap.add_argument("--context", default=None, help="context JSON (or path) applied to ALL cases")
    args = ap.parse_args()

    base_policy = _load_json_arg(args.policy)
    base_context = _load_json_arg(args.context)

    # Recommend denoiser off for this milestone; if your runner ignores it, the script still works.
    default_policy_hint = {"rails": {"denoise": False}}

    # Canonical scenarios emphasizing clean intent and rails behavior
    scenarios = [
        {
            "name": "deposit_eth",
            "prompt": "deposit 10 ETH into aave",
            "expect": {"kind": "stable_top1_nonempty", "primitive": "deposit_asset"},
            "context": {"risk":{"hf":1.22}, "oracle":{"age_sec":5,"max_age_sec":30}},
            "policy": {"ltv_max":0.75, **default_policy_hint},
        },
        {
            "name": "swap_eth_usdc",
            "prompt": "swap 2 ETH for USDC",
            "expect": {"kind": "stable_top1_nonempty", "primitive": "swap_asset"},
            "context": {"oracle":{"age_sec":5,"max_age_sec":30}},
            "policy": {"ltv_max":0.75, **default_policy_hint},
        },
        # Add more as needed (borrow, repay) if your mapper is trained on them
    ]

    results = []
    failures = []

    for sc in scenarios:
        prompt = sc["prompt"]
        merged_ctx = {**base_context, **sc.get("context", {})}
        merged_pol = {**default_policy_hint, **base_policy, **sc.get("policy", {})}
        out = run_case(prompt, merged_ctx, merged_pol, rails=args.rails, T=args.T, runs=args.runs)

        # Evaluate expectations
        expect = sc["expect"]
        kind = expect["kind"]
        ok = True
        reason = ""

        if kind == "stable_top1":
            if not out["stable_top1"]:
                ok = False
                reason = f"top-1 not stable across {args.runs} runs: {out['top1_list']}"
        elif kind == "stable_top1_nonempty":
            want = expect.get("primitive")
            if not out["stable_top1"]:
                ok = False
                reason = f"top-1 not stable across {args.runs} runs: {out['top1_list']}"
            elif out["top1_list"][0] is None:
                ok = False
                reason = "top-1 is None (empty plan)"
            elif want and out["top1_list"][0] != want:
                ok = False
                reason = f"expected top-1={want}, got={out['top1_list'][0]}"
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
        "milestone": "defi_milestone2",
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
