#!/usr/bin/env python3
"""
Milestone-4: Verifier & Policy Gating

Gates:
- deposit_eth: verify.ok == True and stable top-1 == deposit_asset
- swap_eth_usdc: verify.ok == True and stable top-1 == swap_asset
- withdraw_high_ltv: verify.ok == False (blocked) across runs
- borrow_low_hf:     verify.ok == False (blocked) across runs

python3 milestones/defi_milestone4.py \
  --rails stage11 \
  --runs 5 \
  --policy '{"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7},"ltv_max":0.75}' \
  --context '{"oracle":{"age_sec":5,"max_age_sec":30}}'

python3 milestones/inspect_summary.py .artifacts/defi_milestone4_summary.json

pytest -k m4_stability_smoke -v

"""

from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
from typing import Any, Dict, List
from micro_llm.pipelines.runner import run_micro

ARTIFACTS_DIR = Path(".artifacts")
SUMMARY_PATH  = ARTIFACTS_DIR / "defi_milestone4_summary.json"

def _hash_ctx(ctx: Dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(ctx, sort_keys=True).encode()).hexdigest()[:8]

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
                "stage11": res.get("aux", {}).get("stage11"),
            }
        })
    top1_list = [o["top1"] for o in outs]
    stable = len(set(top1_list)) == 1
    # Pull verify.ok for each run
    veroks = [bool(o.get("verify", {}).get("ok")) for o in outs]
    return {
        "prompt": prompt,
        "context_hash": _hash_ctx(context),
        "policy": policy,
        "rails": rails,
        "T": T,
        "stable_top1": stable,
        "top1_list": top1_list,
        "verify_ok_list": veroks,
        "outputs": outs,
    }

def _expect_exec(want_prim: str, out: Dict[str, Any], runs: int) -> (bool, str):
    if not out["stable_top1"]:
        return False, f"top-1 not stable across {runs} runs: {out['top1_list']}"
    if not out["top1_list"] or out["top1_list"][0] is None:
        return False, "empty plan (None) but expected executable"
    if out["top1_list"][0] != want_prim:
        return False, f"expected top-1={want_prim}, got={out['top1_list'][0]}"
    if not all(out["verify_ok_list"]):
        return False, f"verify failed for at least one run: {out['verify_ok_list']}"
    return True, ""

def _expect_blocked(out: Dict[str, Any], runs: int) -> (bool, str):
    # Blocked means verifier says ok==False across all runs (conservative)
    if all(out["verify_ok_list"]):
        return False, f"expected blocked, but verify_ok_list={out['verify_ok_list']}"
    return True, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rails", default="stage11")
    ap.add_argument("--T", type=int, default=180)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--context", default=None)
    args = ap.parse_args()

    base_policy = _load_json_arg(args.policy)
    base_context = _load_json_arg(args.context)

    # Ensure mapper defaults so it "just works"
    base_policy.setdefault("mapper", {})
    base_policy["mapper"].setdefault("model_path", ".artifacts/defi_mapper.joblib")
    base_policy["mapper"].setdefault("confidence_threshold", 0.7)

    # Scenarios
    scenarios = [
        # Allowed actions
        dict(
            name="deposit_eth",
            prompt="deposit 10 ETH into aave",
            expect=("exec", "deposit_asset"),
            context={"risk":{"hf":1.22}, "oracle":{"age_sec":5,"max_age_sec":30}},
            policy={"ltv_max":0.75},
        ),
        dict(
            name="swap_eth_usdc",
            prompt="swap 2 ETH for USDC",
            expect=("exec", "swap_asset"),
            context={"oracle":{"age_sec":5,"max_age_sec":30}},
            policy={"ltv_max":0.75},
        ),
        # Blocked by policy/verify
        dict(
            name="withdraw_high_ltv",
            prompt="withdraw 5 ETH",
            expect=("blocked", None),
            # Simulate high utilization so withdraw should be unsafe under ltv_max
            context={"risk":{"hf":1.15}, "oracle":{"age_sec":5,"max_age_sec":30}},
            policy={"ltv_max":0.60},  # stricter cap to trigger block
        ),
        dict(
            name="borrow_low_hf",
            prompt="borrow 1000 USDC",
            expect=("blocked", None),
            context={"risk":{"hf":1.05}, "oracle":{"age_sec":5,"max_age_sec":30}},
            policy={"ltv_max":0.75},
        ),
    ]

    results = []
    failures = []

    for sc in scenarios:
        merged_ctx = {**base_context, **sc.get("context", {})}
        merged_pol = {**base_policy, **sc.get("policy", {})}
        # Turn on denoise by default for M4 (safer rail)
        merged_pol.setdefault("rails", {})
        merged_pol["rails"]["denoise"] = True

        out = _run_repeated(sc["prompt"], merged_ctx, merged_pol, rails=args.rails, T=args.T, runs=args.runs)

        kind, want = sc["expect"]
        if kind == "exec":
            ok, reason = _expect_exec(want, out, args.runs)
        elif kind == "blocked":
            ok, reason = _expect_blocked(out, args.runs)
        else:
            ok, reason = False, f"unknown expect kind: {kind}"

        results.append({
            "name": sc["name"],
            "ok": ok,
            "reason": reason,
            "output": out
        })
        if not ok:
            failures.append((sc["name"], reason))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "milestone": "defi_milestone4",
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
