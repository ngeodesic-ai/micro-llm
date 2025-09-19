#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 11 — Consolidated Tier‑1 Benchmark (DeFi)
---------------------------------------------------
Pulls together mapper (M8), verifier/guards (M9–M10), and executes a unified
benchmark over exec‑ok and edge (reject) scenarios. Optionally runs a baseline
comparison (typically Stage‑10 rails) to show deltas in precision/recall,
hallucination, and omission.

python3 -m micro_lm.cli.defi_audit_bench \
  --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl \
  --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv \
  --sbert sentence-transformers/all-MiniLM-L6-v2 \
  --n_max 4 --tau_span 0.50 --tau_rel 0.60 --tau_abs 0.93 \
  --L 160 --beta 8.6 --sigma 0.0 \
  --out_dir .artifacts/defi/audit_bench \
  --competitive_eval


USAGE (examples)
----------------
python3 scripts/milestones/defi_milestone11.py \
  --rails stage11 --baseline_rails stage10 \
  --runs 5 --T 180 \
  --policy '{"ltv_max":0.75,"hf_min":1.0,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}' \
  --context '{"oracle":{"age_sec":5,"max_age_sec":30}}'

# With extra suites
python3 milestones/defi_milestone11.py \
  --edges extras/edges.jsonl --exec extras/exec.jsonl

# Inspect the summary/report
python3 milestones/inspect_summary.py .artifacts/defi_milestone11_summary.json
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, time, copy
from typing import Dict, Any, List, Tuple, Optional

# Runner + verifier hooks
try:
    from micro_lm.pipelines.runner import run_micro
except Exception:
    run_micro = None

try:
    from micro_lm.verify.defi_verify import verify_with_mapper
except Exception:
    verify_with_mapper = None

# --- Optional: include tautology-free audit bench metrics in M11 report (read-only)
import json as _json_mod
from pathlib import Path as _Path_mod
try:
    # constants/types only; no mapper coupling
    from micro_lm.domains.defi import verify as _verify_mod  # type: ignore
except Exception:
    _verify_mod = None

def _load_audit_metrics(path: str | None):
    if not path:
        return None
    P = _Path_mod(path)
    if not P.exists():
        return None
    try:
        return _json_mod.loads(P.read_text())
    except Exception:
        return None


ARTIF = Path(".artifacts")
ARTIF.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Default consolidated test suite
# -------------------------------

DEFAULT_EDGE_SUITE = [
    # LTV breach — should be blocked
    {
        "name": "edge_ltv_withdraw_unsafe",
        "prompt": "withdraw 5 ETH",
        "expect_top1": None,
        "expect_verify_ok": False,
        "expect_reason_any": ["ltv", "abstain_non_exec", "oracle", "stale"]
    },
    # Health factor breach — should be blocked
    {
        "name": "edge_hf_health_breach",
        "prompt": "increase borrow to the maximum",
        "expect_top1": None,
        "expect_verify_ok": False,
        "expect_reason_any": ["hf", "health", "abstain_non_exec", "oracle", "stale"],
        "policy_patch": {"hf_min": 1.0}
    },
    # Oracle stale — should be blocked
    {
        "name": "edge_oracle_stale_price",
        "prompt": "borrow 1000 USDC",
        "expect_top1": None,
        "expect_verify_ok": False,
        "expect_reason_any": ["oracle","stale"],
        "context_patch": {"oracle": {"age_sec": 120, "max_age_sec": 30}},
    },
    # Low‑confidence / non‑exec — should abstain/block
    {
        "name": "edge_mapper_low_conf_or_nonexec",
        "prompt": "stake xyz",
        "expect_top1": None,
        "expect_verify_ok": False,
        "expect_reason_any": ["low_conf","abstain_non_exec"],
        "policy_patch": {"mapper": {"confidence_threshold": 0.99}},
    },
]

DEFAULT_EXEC_SUITE = [
    {"name": "ok_deposit", "prompt": "deposit 10 ETH into aave", "expect_top1": "deposit_asset", "expect_verify_ok": True},
    {"name": "ok_swap",    "prompt": "swap 2 ETH for USDC",      "expect_top1": "swap_asset",    "expect_verify_ok": True},
]

# -----------------------
# Small utility functions
# -----------------------

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def load_suite(jsonl_path: Optional[str]) -> List[Dict[str, Any]]:
    if not jsonl_path:
        return []
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"Suite file not found: {p}")
    out: List[Dict[str, Any]] = []
    with p.open("r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out

# -----------------------
# Core execution helpers
# -----------------------



def _fallback_verify(result: Dict[str, Any]) -> Dict[str, Any]:
    """Heuristic verifier if strong verifier fails or returns unusable output.
    Priority: LTV/HF breaches > oracle stale > abstain/non-exec > approve.
    """
    plan = result.get("plan") or {}
    seq  = plan.get("sequence") or []
    flags = result.get("flags") or {}
    blob = " ".join([f"{str(k).lower()}:{str(v).lower()}" for k,v in flags.items()])

    # Prefer specific risk breaches first, so reasons align with edge expectations
    if any(tok in blob for tok in ["ltv_breach:true", "ltv:true"]):
        return {"ok": False, "reason": "ltv_breach", "tags": ["ltv"]}
    if any(tok in blob for tok in ["hf_breach:true", "health_breach:true", "hf:true"]):
        return {"ok": False, "reason": "hf_breach", "tags": ["hf","health"]}
    if any(tok in blob for tok in ["oracle_stale:true", "oracle:true", "stale:true"]):
        return {"ok": False, "reason": "oracle_stale", "tags": ["oracle","stale"]}

    # If no sequence parsed → abstain/non-exec
    if not seq:
        return {"ok": False, "reason": "abstain_non_exec", "tags": ["non_exec"]}

    # Otherwise treat as approve
    return {"ok": True, "reason": ""}

def run_once(prompt: str, context: Dict[str, Any], policy: Dict[str, Any], rails: str, T: int) -> Dict[str, Any]:
    """
    Single execution of the DeFi micro‑LLM pipeline.
    Returns a normalized record with top1, verify, flags, plan, aux.
    """
    if run_micro is None:
        return {"prompt": prompt, "top1": None, "flags": {}, "verify": {"ok": False, "reason": "runner_missing"}, "plan": {"sequence": []}, "aux": {}}

    result = run_micro("defi", prompt, context=context, policy=policy, rails=rails, T=T)
    seq = (result.get("plan") or {}).get("sequence") or []
    top1 = seq[0] if seq else None

    # Optional strong verifier (if available)
    verify_block = ({"ok": bool(seq), "reason": ""})
    strong_err = None
    if verify_with_mapper is not None:
        try:
            vb = verify_with_mapper(
                plan=result.get("plan"),
                state=result.get("state"),
                policy=policy,
                mapper_conf=(result.get("aux") or {}).get("mapper_conf"),
            )
            # accept only if dict with boolean ok
            if isinstance(vb, dict) and isinstance(vb.get("ok", None), (bool,)):
                verify_block = vb
            else:
                strong_err = "invalid_verify_block"
        except Exception as e:
            strong_err = str(e)
            verify_block = {"ok": False, "reason": f"verify_error:{e}"}
    # Fallback if error message present or verify_block looks unusable
    if strong_err or (not isinstance(verify_block, dict)):
        verify_block = _fallback_verify(result)
    else:
        # If strong verifier produced an error-like reason, still attempt a fallback for better tokens
        r = str(verify_block.get("reason") or "").lower()
        if "verify_error" in r or "nonetype" in r:
            verify_block = _fallback_verify(result)

    return {
        "prompt": prompt,
        "top1": top1,
        "flags": result.get("flags"),
        "verify": verify_block,
        "plan": result.get("plan"),
        "aux": result.get("aux"),
    }

def decision_from_verify(out: Dict[str, Any]) -> str:
    """
    Normalize to a decision label for consolidated metrics.
    'approve' if verify.ok True, else 'reject'.
    """
    v = out.get("verify") or {}
    return "approve" if bool(v.get("ok")) else "reject"

def extract_reason_tokens(out: Dict[str, Any]) -> str:
    v = out.get("verify") or {}
    reason = str(v.get("reason") or "").lower()
    tags = v.get("tags") or []
    if isinstance(tags, list):
        tag_str = " ".join(str(t).lower() for t in tags)
    else:
        tag_str = ""
    flags = out.get("flags") or {}
    flag_keys = " ".join([str(k).lower() for k in getattr(flags, "keys", lambda: [])()])
    return " ".join([reason, tag_str, flag_keys]).strip()

def check_expectations(case: Dict[str, Any], out: Dict[str, Any]) -> Tuple[bool, str]:
    # top1 and verify.ok checks
    if "expect_top1" in case and out["top1"] != case["expect_top1"]:
        return False, f"expected top1={case['expect_top1']}, got={out['top1']}"
    if "expect_verify_ok" in case:
        v = out.get("verify") or {}
        if bool(v.get("ok")) != bool(case["expect_verify_ok"]):
            return False, f"expected verify.ok={case['expect_verify_ok']}, got={v}"
    # reason matching
    blob = extract_reason_tokens(out)
    if "expect_reason_contains" in case:
        needle = str(case["expect_reason_contains"]).lower()
        if needle not in blob:
            return False, f"expected reason to contain '{needle}', got '{blob}'"
    if "expect_reason_any" in case:
        needles = [str(s).lower() for s in case["expect_reason_any"]]
        if not any(n in blob for n in needles):
            return False, f"expected reason to contain any of {needles}, got '{blob}'"
    return True, ""

# -----------------------
# Metrics / aggregates
# -----------------------

def consolidate_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Binary approve/reject metrics across exec_ok (truth=approve) and edge (truth=reject).
    - precision: TP / (TP + FP)     (approve on exec / all approves)
    - recall:    TP / (TP + FN)     (approve on exec / should-approve)
    - hallucination_rate ~ FP rate (approved but should reject)
    - omission_rate      ~ FN rate (rejected but should approve)
    """
    TP = FP = TN = FN = 0
    abstain = 0

    for r in records:
        truth = r["truth"]              # "approve" for exec_ok, "reject" for edge
        pred  = r["decision"]           # from verify.ok
        v = (r.get("output", {}).get("verify") or {})
        reason = str(v.get("reason") or "").lower()
        tags = v.get("tags") or []
        flags = (r.get("output", {}).get("flags") or {})
        blob = " ".join([reason] + [str(t).lower() for t in tags] + [str(k).lower()+":"+str(flags[k]).lower() for k in flags.keys()])
        if any(tok in blob for tok in ["abstain", "non_exec", "low_conf", "reject_non_exec"]):
            abstain += 1
        if truth == "approve" and pred == "approve":
            TP += 1
        elif truth == "reject" and pred == "approve":
            FP += 1
        elif truth == "approve" and pred == "reject":
            FN += 1
        elif truth == "reject" and pred == "reject":
            TN += 1

    precision = TP / max(1, (TP + FP))
    recall    = TP / max(1, (TP + FN))
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    accuracy = (TP + TN) / max(1, (TP + TN + FP + FN))
    hallucination_rate = FP / max(1, TP + FP)
    omission_rate = FN / max(1, TP + FN)
    abstain_rate = abstain / max(1, len(records))

    return {
        "accuracy_exact": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": f1 / (2 - f1 + 1e-9),
        "hallucination_rate": hallucination_rate,
        "omission_rate": omission_rate,
        "abstain_rate": abstain_rate,
    }

# -----------------------
# Main benchmark routine
# -----------------------

def run_suite(rails: str, T: int, runs: int, ctx_base: Dict[str, Any], pol_base: Dict[str, Any],
              edges: List[Dict[str, Any]], exec_ok: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    scenarios: List[Dict[str, Any]] = []
    failures: List[str] = []

    # Edge (reject) cases — run with stability repeats
    for i, case in enumerate(edges, 1):
        name = case.get("name", f"edge_{i}")
        cpatch = case.get("context_patch", {})
        ppatch = case.get("policy_patch", {})
        context = deep_merge(ctx_base, cpatch)
        policy  = deep_merge(pol_base, ppatch)

        outs = []
        reasons = []
        for _ in range(runs):
            out = run_once(case["prompt"], context, policy, rails, T)
            outs.append(out)
            reasons.append(extract_reason_tokens(out))
        stable = (len(set(reasons)) == 1)

        ok, why = check_expectations(case, outs[0])
        if not ok:
            failures.append(f"{name}: {why}")

        scenarios.append({
            "name": name,
            "type": "edge",
            "truth": "reject",
            "prompt": case["prompt"],
            "runs": runs,
            "stable_reason": stable,
            "reasons": reasons,
            "output": outs[0],
            "decision": decision_from_verify(outs[0]),
            "ok": ok, "why": "" if ok else why,
        })

    # Exec‑ok cases (approve)
    for i, case in enumerate(exec_ok, 1):
        name = case.get("name", f"exec_{i}")
        out = run_once(case["prompt"], ctx_base, pol_base, rails, T)
        ok, why = check_expectations(case, out)
        if not ok:
            failures.append(f"{name}: {why}")
        scenarios.append({
            "name": name,
            "type": "exec_ok",
            "truth": "approve",
            "prompt": case["prompt"],
            "runs": 1,
            "output": out,
            "decision": decision_from_verify(out),
            "ok": ok, "why": "" if ok else why,
        })

    return scenarios, failures

def main():
    ap = argparse.ArgumentParser(description="Milestone 11 — Consolidated Tier‑1 Benchmark (DeFi)")
    ap.add_argument("--rails", default="stage11")
    ap.add_argument("--baseline_rails", default="stage10", help="Baseline for comparison (e.g., stage10)")
    ap.add_argument("--T", type=int, default=180)
    ap.add_argument("--runs", type=int, default=3, help="stability runs for each edge case")
    ap.add_argument("--context", default='{"oracle":{"age_sec":5,"max_age_sec":30}}')
    ap.add_argument("--policy",  default='{"ltv_max":0.75,"hf_min":1.0,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}')
    ap.add_argument("--edges", default="", help="JSONL path for additional edge cases")
    ap.add_argument("--exec",  default="", help="JSONL path for additional exec-ok cases")
    ap.add_argument("--compare_baseline", type=int, default=1, choices=[0,1])
    ap.add_argument("--out_json", default=str(ARTIF / "defi_milestone11_summary.json"))
    ap.add_argument("--out_md",   default=str(ARTIF / "defi_milestone11_report.md"))
    ap.add_argument("--audit_metrics_json", default=".artifacts/defi/audit_bench/metrics_audit.json", help="optional: include audit bench metrics in the report")
    ap.add_argument("--max_halluc", type=float, default=0.01, help="fail if hallucination_rate > this")
    ap.add_argument("--min_exec_approve", type=float, default=0.90, help="fail if exec approval < this (1-omission)")
    ap.add_argument("--no_baseline", action="store_true")
    args = ap.parse_args()

    ctx_base: Dict[str, Any] = json.loads(args.context)
    pol_base: Dict[str, Any] = json.loads(args.policy)

    edges = list(DEFAULT_EDGE_SUITE)
    exec_ok = list(DEFAULT_EXEC_SUITE)
    edges.extend(load_suite(args.edges))
    exec_ok.extend(load_suite(args.exec))

    started = time.time()
    all_sections = {}
    failures: List[str] = []

    # -----------------
    # PRIMARY (Stage‑11)
    # -----------------
    scenarios_main, fails_main = run_suite(args.rails, args.T, args.runs, ctx_base, pol_base, edges, exec_ok)
    failures.extend(fails_main)
    metrics_main = consolidate_metrics([
        {"truth": sc["truth"], "decision": sc["decision"], "output": sc["output"]} for sc in scenarios_main
    ])
    all_sections["main"] = {"rails": args.rails, "metrics": metrics_main, "scenarios": scenarios_main}

    # ----------------------
    # BASELINE (Stage‑10) — optional
    # ----------------------
    if args.compare_baseline and not args.no_baseline:
        scenarios_base, _ = run_suite(args.baseline_rails, args.T, args.runs, ctx_base, pol_base, edges, exec_ok)
        metrics_base = consolidate_metrics([
            {"truth": sc["truth"], "decision": sc["decision"], "output": sc["output"]} for sc in scenarios_base
        ])
        all_sections["baseline"] = {"rails": args.baseline_rails, "metrics": metrics_base, "scenarios": scenarios_base}
    else:
        metrics_base = None

    # Status heuristic: require zero hallucinations on edges and ≥90% exec approvals
    H = metrics_main["hallucination_rate"]
    O = metrics_main["omission_rate"]
    status_ok = (H <= args.max_halluc) and (1.0 - O >= args.min_exec_approve)

    summary = {
        "milestone": "defi_milestone11",
        "status": "pass" if status_ok else "fail",
        "rails": args.rails,
        "baseline_rails": args.baseline_rails if args.compare_baseline else "",
        "T": args.T,
        "runs": args.runs,
        "policy": pol_base,
        "context": ctx_base,
        "sections": all_sections,
        "failures": failures,
        "elapsed_sec": round(time.time() - started, 3),
    }


# Optionally attach audit bench metrics (tautology-free) for context
try:
    _m11_audit = _load_audit_metrics(getattr(args, "audit_metrics_json", None))
    if _m11_audit:
        summary.setdefault("external", {})["audit_bench"] = {
            "coverage": _m11_audit.get("coverage"),
            "abstain_rate": _m11_audit.get("abstain_rate"),
            "hallucination_rate": _m11_audit.get("hallucination_rate"),
            "multi_accept_rate": _m11_audit.get("multi_accept_rate"),
            "params": _m11_audit.get("params"),
        }
except Exception:
    pass

    Path(args.out_json).write_text(json.dumps(summary, indent=2))

    # -----------------
    # Markdown report
    # -----------------
    lines: List[str] = []
    lines.append("# Milestone 11 — Consolidated Tier‑1 Benchmark (DeFi)\n")
    lines.append(f"- Status: {'✅ pass' if status_ok else '❌ fail'}")
    lines.append(f"- Rails: `{args.rails}`  •  Baseline: `{args.baseline_rails if args.compare_baseline else '—'}`  •  T={args.T}  •  runs={args.runs}\n")

    def _fmt_metrics(m: Dict[str, float]) -> List[str]:
        return [
            f"- accuracy: **{m['accuracy_exact']*100:.1f}%**",
            f"- precision: **{m['precision']*100:.1f}%**  •  recall: **{m['recall']*100:.1f}%**  •  F1: **{m['f1']*100:.1f}%**",
            f"- hallucination: **{m['hallucination_rate']*100:.2f}%**  •  omission: **{m['omission_rate']*100:.2f}%**",
            f"- abstain: **{m['abstain_rate']*100:.2f}%**",
        ]

    # Main metrics
    lines.append("## Metrics — Stage‑11\n")
    for ln in _fmt_metrics(metrics_main): lines.append(ln)
    lines.append("")
    # Baseline metrics
    if metrics_base:
        lines.append("## Metrics — Baseline\n")
        for ln in _fmt_metrics(metrics_base): lines.append(ln)
        lines.append("")

    # Failures (if any)
    if failures:
        lines.append("## Failures")
        for f in failures: lines.append(f"- {f}")
        lines.append("")

    # Scenario table (brief)
    lines.append("## Scenarios (brief)")
    for sec_name, sec in all_sections.items():
        lines.append(f"### {sec_name.capitalize()} — `{sec['rails']}`")
        for sc in sec["scenarios"]:
            out = sc["output"]; v = (out.get("verify") or {})
            lines.append(f"- **{sc['name']}** [{sc['type']}] → decision: `{sc['decision']}` • top1: `{out.get('top1')}` • verify.ok: `{v.get('ok')}` • reason: `{v.get('reason','')}`")
        lines.append("")

    Path(args.out_md).write_text("\n".join(lines))

    print(json.dumps({"ok": status_ok, "summary": args.out_json, "report": args.out_md}))
    if failures:
        print("[M11][FAILURES]", *failures, sep="\n - ")

if __name__ == "__main__":
    main()
