#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeFi Milestone 10 — Robustness & Stress Benchmark
-------------------------------------------------
Builds on Milestone 9 (verifier + edge guards) and integrates with Stage-11 rails.
Focus:
  - Run baseline vs perturbed prompts (typos, jitter, synonyms).
  - Ensure verifier still blocks edge cases (zero false approvals).
  - Measure accuracy drop, abstain, and latency drift.
  
python3 milestones/defi_milestone10.py \
  --rails stage11 --runs 3 --seed 42 --perturb --perturb_k 3 \
  --policy '{"ltv_max":0.75,"hf_min":1.0,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}' \
  --context '{"oracle":{"age_sec":5,"max_age_sec":30}}'
python3 milestones/inspect_summary.py .artifacts/defi_milestone10_summary.json

  
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, random, re, time
from typing import Dict, Any, List, Optional

# ---- Safe imports ----
try:
    from micro_lm.pipelines.runner import run_micro
except Exception:
    run_micro = None

try:
    from defi_verify import verify_with_mapper
except Exception:
    try:
        from micro_lm.verify.defi_verify import verify_with_mapper
    except Exception:
        verify_with_mapper = None

# ---- Suites ----
EXEC_SUITE = [
    {"name": "ok_deposit", "type": "exec_ok", "prompt": "deposit 10 ETH into aave", "expect_top1": "deposit_asset"},
    {"name": "ok_swap",    "type": "exec_ok", "prompt": "swap 2 ETH for USDC",      "expect_top1": "swap_asset"},
]

EDGE_SUITE = [
    {"name": "edge_ltv_withdraw_unsafe", "type": "edge", "prompt": "withdraw 5 ETH", "expect_reason_any": ["ltv"]},
    {"name": "edge_hf_health_breach",    "type": "edge", "prompt": "increase borrow to the maximum", "expect_reason_any": ["hf"]},
    {"name": "edge_oracle_stale_price",  "type": "edge", "prompt": "borrow 1000 USDC",
     "expect_reason_any": ["oracle", "stale"],
     "context_patch": {"oracle": {"age_sec": 120, "max_age_sec": 30}}},
]


def perturb_prompt(p: str) -> str:
    """Simple perturbations: typos, spacing, casing."""
    r = random.random()
    if r < 0.3 and len(p) > 4:
        i = random.randint(1, len(p)-2)
        return p[:i] + p[i+1] + p[i] + p[i+2:]  # swap
    elif r < 0.6:
        return " " + p + " "
    else:
        return p.upper()
    

def run_suite(args, policy: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    scenarios = EDGE_SUITE + EXEC_SUITE
    results = []
    for sc in scenarios:
        prompt = sc["prompt"]
        if args.perturb:
            prompt = perturb_prompt(prompt)
        result = run_micro(prompt, policy, context, rails=args.rails)
        verify_block = verify_with_mapper(
            plan=result.get("plan"), state=result.get("state"),
            policy=policy, mapper_conf=result.get("aux", {}).get("mapper_conf")
        )
        result["verify"] = verify_block
        results.append(dict(name=sc["name"], type=sc["type"], prompt=prompt, output=result))
    return results


def evaluate(results: List[Dict[str, Any]], base_latency: float, args) -> Dict[str, Any]:
    acc_ok = sum(1 for r in results if r["type"]=="exec_ok" and r["output"]["verify"]["ok"])
    total_ok = sum(1 for r in results if r["type"]=="exec_ok")
    acc = acc_ok / max(1, total_ok)

    false_edges = [r["name"] for r in results if r["type"]=="edge" and r["output"]["verify"]["ok"]]
    abstain = sum(1 for r in results if r["output"]["verify"].get("reason")=="abstain_non_exec")
    abstain_rate = abstain / len(results)

    latency_now = sum(r["output"].get("latency", 0) for r in results) / max(1,len(results))
    delta_latency = (latency_now - base_latency) / max(1e-9, base_latency)

    return dict(
        accuracy_ok=acc,
        false_edges=false_edges,
        abstain_rate=abstain_rate,
        delta_latency=delta_latency
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rails", type=str, default="stage11")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--policy", type=str, required=True)
    ap.add_argument("--context", type=str, required=True)
    ap.add_argument("--perturb", action="store_true")
    ap.add_argument("--perturb_k", type=int, default=3)
    ap.add_argument("--max_drop_pp", type=float, default=1.5)
    ap.add_argument("--abstain_target", type=float, default=0.10)
    ap.add_argument("--max_latency_p95_increase", type=float, default=0.10)
    ap.add_argument("--summary", type=str, default=".artifacts/defi_milestone10_summary.json")
    ap.add_argument("--report", type=str, default=".artifacts/defi_milestone10_report.md")
    args = ap.parse_args()

    random.seed(args.seed)
    policy = json.loads(args.policy)
    context = json.loads(args.context)

    # Baseline run
    base_results = run_suite(args, policy, context)
    base_latency = sum(r["output"].get("latency", 0) for r in base_results)/max(1,len(base_results))

    all_results = []
    for i in range(args.runs):
        perturbed = run_suite(args, policy, context)
        all_results.extend(perturbed)

    eval_block = evaluate(all_results, base_latency, args)
    ok = (
        eval_block["accuracy_ok"] >= 1.0 - args.max_drop_pp/100.0
        and not eval_block["false_edges"]
        and eval_block["abstain_rate"] <= args.abstain_target
        and eval_block["delta_latency"] <= args.max_latency_p95_increase
    )

    summary = dict(
        milestone="defi_milestone10",
        status="pass" if ok else "fail",
        rails=args.rails,
        runs=args.runs,
        results=all_results,
        metrics=eval_block
    )

    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    with open(args.summary,"w") as f: json.dump(summary,f,indent=2)
    with open(args.report,"w") as f:
        f.write(f"# Milestone 10 Report\n\n")
        f.write(json.dumps(eval_block, indent=2))

    print(json.dumps(dict(ok=ok, summary=args.summary, report=args.report)))


if __name__=="__main__":
    main()
