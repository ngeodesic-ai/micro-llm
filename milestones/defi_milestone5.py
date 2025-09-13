#!/usr/bin/env python3
# milestones/defi_milestone5.py
import argparse, json, pathlib, sys, time, hashlib
from typing import Dict, Any, List, Tuple
from micro_llm.pipelines.runner import run_micro

"""

python3 milestones/defi_milestone5.py \
  --rails stage11 \
  --runs 5 \
  --policy '{"ltv_max":0.75,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}' \
  --context '{"oracle":{"age_sec":5,"max_age_sec":30}}'
python3 milestones/inspect_summary.py .artifacts/defi_milestone5_summary.json

"""

ARTIF = pathlib.Path(".artifacts")
ARTIF.mkdir(exist_ok=True, parents=True)

SUITE = [
    # exec paths
    {"name": "deposit_eth",     "prompt": "deposit 10 ETH into aave",
     "expect_top1": "deposit_asset", "expect_verify_ok": True},
    {"name": "swap_eth_usdc",   "prompt": "swap 2 ETH for USDC",
     "expect_top1": "swap_asset",    "expect_verify_ok": True},

    # abstain/verify trips
    {"name": "withdraw_high_ltv", "prompt": "withdraw 5 ETH",
     "expect_top1": None, "expect_verify_ok": False, "expect_reason_contains": "ltv"},
    {"name": "borrow_low_hf",     "prompt": "borrow 1000 USDC",
     "expect_top1": None, "expect_verify_ok": False, "expect_reason_contains": "hf"},

    # non-exec abstain
    {"name": "nonexec_abstain", "prompt": "check balance",
     "expect_top1": None, "expect_verify_ok": False, "expect_reason_contains": "abstain_non_exec"},
]


def ctx_hash(ctx: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(ctx, sort_keys=True).encode()).hexdigest()[:8]

def run_once(prompt: str, context: Dict[str, Any], policy: Dict[str, Any], rails: str, T: int) -> Dict[str, Any]:
    res = run_micro("defi", prompt, context=context, policy=policy, rails=rails, T=T)

    import os, sys, json as _json
    if os.getenv("MICROLLM_DEBUG"):
        print("[M5] res.verify:", _json.dumps(res.get("verify"), indent=2), file=sys.stderr)
        print("[M5] res.flags :", _json.dumps(res.get("flags"), indent=2), file=sys.stderr)
        print("[M5] res.plan  :", _json.dumps((res.get("plan") or {}).get("sequence"), indent=2), file=sys.stderr)

    
    seq = (res.get("plan") or {}).get("sequence") or []
    top1 = seq[0] if seq else None
    out = {
        "prompt": prompt,
        "context_hash": ctx_hash(context),
        "policy": policy,
        "rails": rails,
        "T": T,
        "top1": top1,
        "flags": res.get("flags"),
        "verify": res.get("verify"),
        "aux": {
            "prior": (res.get("aux") or {}).get("prior"),
            "mapper_confidence": (res.get("aux") or {}).get("mapper_confidence"),
        },
    }
    return out

def run_stability(prompt: str, runs: int, context: Dict[str, Any], policy: Dict[str, Any], rails: str, T: int) -> Dict[str, Any]:
    tops = []
    for _ in range(runs):
        r = run_once(prompt, context, policy, rails, T)
        tops.append(r["top1"])
    stable = len(set(tops)) == 1
    return {"top1_list": tops, "stable_top1": stable, "top1_first": tops[0] if tops else None}

def check_expectations(name: str, single: Dict[str, Any], exp: Dict[str, Any]) -> Tuple[bool, str]:
    # top1 expectation
    et = exp.get("expect_top1", "...skip...")
    if et != "...skip..." and single["top1"] != et:
        return False, f"{name}: expected top1={et}, got={single['top1']}"
    # verify expectation
    v = single["verify"] or {}
    evo = exp.get("expect_verify_ok")
    if evo is not None and bool(v.get("ok")) != bool(evo):
        return False, f"{name}: expected verify.ok={evo}, got={v}"
    # reason contains (soft check)
    substr = exp.get("expect_reason_contains")
    if substr:
        reason = (v.get("reason") or "").lower()
        if substr not in reason:
            return False, f"{name}: expected verify.reason to contain '{substr}', got='{reason}'"
    return True, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rails", default="stage11")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--T", type=int, default=180)
    ap.add_argument("--context", default='{"oracle":{"age_sec":5,"max_age_sec":30}}')
    ap.add_argument("--policy",  default='{"ltv_max":0.75, "mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}')
    args = ap.parse_args()

    ctx_base = json.loads(args.context)
    pol_base = json.loads(args.policy)

    started = time.time()
    scenarios_out: List[Dict[str, Any]] = []
    overall_ok = True
    failures: List[str] = []

    # A) Baseline stability
    for case in SUITE:
        name = case["name"]
        single = run_once(case["prompt"], ctx_base, pol_base, args.rails, args.T)
        stab = run_stability(case["prompt"], args.runs, ctx_base, pol_base, args.rails, args.T)
        ok, why = check_expectations(name, single, case)
        if not ok:
            overall_ok = False
            failures.append(why)
        scenarios_out.append({"name": name, "ok": ok, "reason": "" if ok else why,
                              "output": single, "stability": stab})

    # B) Denoise A/B for exec prompts only
    denoise_pol = json.loads(args.policy)
    denoise_pol.setdefault("rails", {})["denoise"] = True
    for case in [c for c in SUITE if c.get("expect_top1")]:
        name = case["name"] + "_denoised"
        single = run_once(case["prompt"], ctx_base, denoise_pol, args.rails, args.T)
        ok, why = check_expectations(name, single, case)
        if not ok:
            overall_ok = False
            failures.append(why)
        scenarios_out.append({"name": name, "ok": ok, "reason": "" if ok else why, "output": single})

    # Summaries
    summary = {
        "milestone": "defi_milestone5",
        "status": "pass" if overall_ok else "fail",
        "rails": args.rails,
        "T": args.T,
        "runs": args.runs,
        "scenarios": scenarios_out,
        "failures": failures,
        "elapsed_sec": round(time.time() - started, 3),
    }
    summary_path = ARTIF / "defi_milestone5_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Human report
    lines = []
    lines.append(f"# Milestone 5 Report\n")
    lines.append(f"- Status: {'✅ pass' if overall_ok else '❌ fail'}")
    lines.append(f"- Rails: `{args.rails}`  •  T={args.T}  •  runs={args.runs}\n")
    for sc in scenarios_out:
        lines.append(f"## {sc['name']} — {'OK' if sc['ok'] else 'FAIL'}")
        o = sc["output"]
        lines.append(f"- prompt: `{o['prompt']}`")
        lines.append(f"- top1: `{o['top1']}`  •  verify.ok: `{(o.get('verify') or {}).get('ok')}`  •  reason: `{(o.get('verify') or {}).get('reason','')}`")
        if "stability" in sc:
            st = sc["stability"]
            lines.append(f"  - top1_list: {st['top1_list']}  •  stable: {st['stable_top1']}  •  top1(first): {st['top1_first']}")
        if not sc["ok"]:
            lines.append(f"  - WHY: {sc['reason']}")
        lines.append("")
    report_path = ARTIF / "defi_milestone5_report.md"
    report_path.write_text("\n".join(lines))

    # Print single JSON line so tests can parse
    print(json.dumps({"ok": overall_ok, "summary": str(summary_path), "report": str(report_path)}))

    sys.exit(0 if overall_ok else 2)

if __name__ == "__main__":
    main()
