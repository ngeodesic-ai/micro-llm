#!/usr/bin/env python3
"""
DeFi Stage-11 style benchmark to validate well mapping (no hallucinations) and F1=1.0

- Runs canonical prompts for each DeFi primitive using micro_llm.pipelines.runner.run_micro
- Checks that the predicted top1 primitive matches the expected one
- Repeats N runs per prompt to verify stability (same top1 across runs)
- Emits a summary JSON and a human-readable report

Example:
  python3 defi_benchmark_latest.py \
    --rails stage11 \
    --runs 5 \
    --context '{"risk":{"hf":1.0},"oracle":{"age_sec":5,"max_age_sec":30}}' \
    --policy '{"ltv_max":0.75, "mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}'

Outputs (by default):
  .artifacts/defi_benchmark_latest_summary.json
  .artifacts/defi_benchmark_latest_report.md

Notes:
- Core set defaults to primitives proven by current mapper examples: deposit, withdraw, borrow, repay, swap, and a non-exec probe.
- Optional collateral primitives (add/remove) can be included with --include-extras once your mapper supports them.
"""
import argparse
import json
import os
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Tuple

# Import the pipeline runner
from micro_llm.pipelines.runner import run_micro

ARTIFACTS_DIR = ".artifacts"
DEFAULT_SUMMARY = os.path.join(ARTIFACTS_DIR, "defi_benchmark_latest_summary.json")
DEFAULT_REPORT = os.path.join(ARTIFACTS_DIR, "defi_benchmark_latest_report.md")

# Canonical prompts per primitive (choose short, unambiguous phrasing)
CORE_TESTS = [
    ("deposit_asset", "deposit 10 ETH into aave"),
    ("swap_asset", "swap 2 ETH for USDC"),
    ("withdraw_asset", "withdraw 5 ETH"),
    ("borrow_asset", "borrow 1000 USDC"),
    ("repay_loan", "repay 500 USDC"),
    (None, "check balance"),  # non-exec probe should abstain
]

EXTRA_TESTS = [
    ("add_collateral", "add 1 ETH as collateral"),
    ("remove_collateral", "remove 0.5 ETH collateral"),
]


def top1_from_result(res: Dict[str, Any]) -> Optional[str]:
    """Extract the top-1 primitive (or None for abstain) from run_micro result."""
    seq = res.get("plan", {}).get("sequence", [])
    if not seq:
        return None
    return seq[0]


def stable(values: List[Any]) -> bool:
    return len(set(values)) == 1


# --- drop-in replacement ---
def compute_f1(golds, preds, abstain_token="__ABSTAIN__"):
    """
    golds/preds are lists of labels where 'None' means abstain.
    Returns: micro_f1 (non-abstain), macro_f1 (non-abstain), abstain_f1.
    """

    def norm(x):
        return abstain_token if x is None else x

    G = [norm(g) for g in golds]
    P = [norm(p) for p in preds]

    # Label sets
    labels_all = sorted(set(G) | set(P))
    labels_non_abstain = [l for l in labels_all if l != abstain_token]

    # Per-label counts
    def counts_for(label):
        tp = sum(1 for g, p in zip(G, P) if g == label and p == label)
        fp = sum(1 for g, p in zip(G, P) if g != label and p == label)
        fn = sum(1 for g, p in zip(G, P) if g == label and p != label)
        return tp, fp, fn

    # Abstain F1
    tp_a, fp_a, fn_a = counts_for(abstain_token)
    abstain_f1 = 0.0 if (2*tp_a + fp_a + fn_a) == 0 else (2*tp_a) / (2*tp_a + fp_a + fn_a)

    # Macro F1 over non-abstain labels
    per_label_f1 = []
    for lbl in labels_non_abstain:
        tp, fp, fn = counts_for(lbl)
        f1 = 0.0 if (2*tp + fp + fn) == 0 else (2*tp) / (2*tp + fp + fn)
        per_label_f1.append(f1)
    macro_f1 = 0.0 if not per_label_f1 else sum(per_label_f1) / len(per_label_f1)

    # Micro F1 over non-abstain labels
    # (aggregate TP/FP/FN across non-abstain labels)
    tp_sum = fp_sum = fn_sum = 0
    for lbl in labels_non_abstain:
        tp, fp, fn = counts_for(lbl)
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
    micro_f1 = 0.0 if (2*tp_sum + fp_sum + fn_sum) == 0 else (2*tp_sum) / (2*tp_sum + fp_sum + fn_sum)

    return micro_f1, macro_f1, abstain_f1


# --- end replacement ---

def run_suite(
    rails: str,
    runs: int,
    context: Dict[str, Any],
    policy: Dict[str, Any],
    include_extras: bool,
    T: int,
) -> Dict[str, Any]:
    tests = CORE_TESTS + (EXTRA_TESTS if include_extras else [])

    results = {}
    golds: List[Optional[str]] = []
    preds: List[Optional[str]] = []

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    for expected, prompt in tests:
        top1_list: List[Optional[str]] = []
        aux_samples: List[Dict[str, Any]] = []

        for _ in range(runs):
            res = run_micro(
                domain="defi",
                prompt=prompt,
                context=context,
                policy=policy,
                rails=rails,
                T=T,
            )
            top1 = top1_from_result(res)
            top1_list.append(top1)
            # capture some helpful aux stats (prior & mapper confidence)
            aux = {
                "prior": res.get("aux", {}).get("prior"),
                "mapper_confidence": res.get("aux", {}).get("mapper_confidence"),
                "verify": res.get("verify", {}),
            }
            aux_samples.append(aux)

        results[prompt] = {
            "expected": expected,
            "top1_list": top1_list,
            "stable_top1": stable(top1_list),
            "top1(first)": top1_list[0],
            "aux": aux_samples,
        }

        # scoring uses the first run (determinism is evaluated separately)
        golds.append(expected)
        preds.append(top1_list[0])

    micro_f1, macro_f1, abstain_f1 = compute_f1(golds, preds)

    ok = all(g == p for g, p in zip(golds, preds)) and all(
        block["stable_top1"] for block in results.values()
    )

    return {
        "ok": ok,
        "rails": rails,
        "runs": runs,
        "F1": {
            "micro": micro_f1,
            "macro": macro_f1,
            "abstain_label_f1": abstain_f1,
        },
        "results": results,
    }


def write_outputs(summary_path: str, report_path: str, payload: Dict[str, Any]) -> None:
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)

    lines = []
    lines.append(f"Rails:     {payload['rails']}")
    lines.append(f"Runs/Case: {payload['runs']}")
    lines.append(f"OK:        {payload['ok']}")
    f1 = payload["F1"]
    lines.append(f"F1 (micro/macro): {f1['micro']:.3f} / {f1['macro']:.3f}")
    lines.append("")

    for prompt, block in payload["results"].items():
        lines.append(f"- prompt:   {prompt}")
        lines.append(f"  expected: {block['expected']}")
        lines.append(f"  stable:   {block['stable_top1']}")
        lines.append(f"  top1:     {block['top1(first)']}")
        lines.append(f"  top1_list: {block['top1_list']}")
        lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description="DeFi Stage-11 style benchmark")
    p.add_argument("--rails", default="stage11")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--T", type=int, default=180)
    p.add_argument("--context", default='{}', help="JSON dict")
    p.add_argument("--policy", default='{}', help="JSON dict")
    p.add_argument("--include-extras", action="store_true", help="Include add/remove collateral probes")
    p.add_argument("--summary", default=DEFAULT_SUMMARY)
    p.add_argument("--report", default=DEFAULT_REPORT)
    args = p.parse_args()

    try:
        context = json.loads(args.context)
    except Exception:
        raise SystemExit("--context must be a JSON object")

    try:
        policy = json.loads(args.policy)
    except Exception:
        raise SystemExit("--policy must be a JSON object")

    payload = run_suite(
        rails=args.rails,
        runs=args.runs,
        context=context,
        policy=policy,
        include_extras=args.include_extras,
        T=args.T,
    )

    # Ensure artifacts dir exists
    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)

    write_outputs(args.summary, args.report, payload)

    # Print a concise console summary
    print(json.dumps({
        "ok": payload["ok"],
        "rails": payload["rails"],
        "runs": payload["runs"],
        "F1": payload["F1"],
        "summary": args.summary,
        "report": args.report,
    }, indent=2))


if __name__ == "__main__":
    main()
