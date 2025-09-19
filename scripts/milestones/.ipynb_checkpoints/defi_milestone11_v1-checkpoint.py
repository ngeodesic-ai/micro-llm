#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 11 — Consolidated Tier-1 Benchmark (DeFi), v2
- Works with current run_micro output: keys ['ok','label','score','reason','artifacts']
- Robust top1 extraction (plan.sequence|steps → label → artifacts.mapper.intent)
- Clear debug prints, and expectation checks use `expect_top1` consistently
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, time, copy
from typing import Dict, Any, List, Tuple, Optional

# --- where run_micro lives today
from micro_lm.core.runner import run_micro  # signature: run_micro(domain, prompt, *, context, policy, rails, T) -> dict

# Optional strong verifier (won’t fail if absent)
try:
    from micro_lm.verify.defi_verify import verify_with_mapper  # type: ignore
except Exception:
    verify_with_mapper = None

ARTIF = Path(".artifacts")
ARTIF.mkdir(parents=True, exist_ok=True)

# -----------------------
# Default consolidated suites (minimal)
# -----------------------
DEFAULT_EDGE_SUITE = [
    {
        "name": "edge_ltv_withdraw_unsafe",
        "prompt": "withdraw 5 ETH",
        "expect_top1": None,
        "expect_verify_ok": False,
        "expect_reason_any": ["ltv", "abstain_non_exec", "oracle", "stale"],
    },
    {
        "name": "edge_hf_health_breach",
        "prompt": "increase borrow to the maximum",
        "expect_top1": None,
        "expect_verify_ok": False,
        "expect_reason_any": ["hf", "health", "abstain_non_exec", "oracle", "stale"],
        "policy_patch": {"hf_min": 1.0},
    },
    {
        "name": "edge_oracle_stale_price",
        "prompt": "borrow 1000 USDC",
        "expect_top1": None,
        "expect_verify_ok": False,
        "expect_reason_any": ["oracle", "stale"],
        "context_patch": {"oracle": {"age_sec": 120, "max_age_sec": 30}},
    },
    {
        "name": "edge_mapper_low_conf_or_nonexec",
        "prompt": "stake xyz",
        "expect_top1": None,
        "expect_verify_ok": False,
        "expect_reason_any": ["low_conf", "abstain_non_exec"],
        "policy_patch": {"mapper": {"confidence_threshold": 0.99}},
    },
]

DEFAULT_EXEC_SUITE = [
    {
        "name": "ok_deposit",
        "prompt": "deposit 10 ETH into aave",
        "expect_top1": "deposit_asset",
        "expect_verify_ok": True,
        "context_patch": {
            "account": {"balances": {"ETH": 20, "USDC": 5000}},
            "market": {"venues": ["aave", "uniswap"]},
        },
    },
    {
        "name": "ok_swap",
        "prompt": "swap 2 ETH for USDC",
        "expect_top1": "swap_asset",
        "expect_verify_ok": True,
        "context_patch": {
            "account": {"balances": {"ETH": 20, "USDC": 5000}},
            "market": {"venues": ["uniswap", "aave"]},
        },
    },
]

# -----------------------
# Utilities
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

def _extract_top1_from_seq(seq: List[Any]) -> Optional[str]:
    if not seq:
        return None
    first = seq[0]
    if isinstance(first, str):
        return first
    if isinstance(first, dict):
        for k in ("op", "action", "name", "type", "label"):
            v = first.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return None

def _normalize_top1(op: Optional[str]) -> Optional[str]:
    if not op:
        return None
    t = op.strip().lower()
    alias = {
        "deposit": "deposit_asset",
        "deposit_token": "deposit_asset",
        "swap": "swap_asset",
        "swap_exact_in": "swap_asset",
        "swap_exact_out": "swap_asset",
    }
    return alias.get(t, t)

def _fallback_verify(res: Dict[str, Any], context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Heuristic verifier when strong verifier is unavailable or abstains."""
    plan = res.get("plan") or {}
    seq  = plan.get("sequence") or plan.get("steps") or []
    flags = res.get("flags") or {}
    blob = " ".join([f"{k}:{str(v).lower()}" for k, v in flags.items()])

    if "ltv" in blob or "ltv_breach:true" in blob:
        return {"ok": False, "reason": "ltv_breach", "tags": ["ltv"]}
    if "hf" in blob or "health_breach:true" in blob or "hf_breach:true" in blob:
        return {"ok": False, "reason": "hf_breach", "tags": ["hf", "health"]}
    if "oracle" in blob or "stale" in blob:
        return {"ok": False, "reason": "oracle_stale", "tags": ["oracle", "stale"]}

    if context:
        try:
            o = context.get("oracle") or {}
            if float(o.get("age_sec", 0)) > float(o.get("max_age_sec", 1e9)):
                return {"ok": False, "reason": "oracle_stale", "tags": ["oracle", "stale"]}
        except Exception:
            pass

    if not seq:
        return {"ok": False, "reason": "abstain_non_exec", "tags": ["non_exec"]}
    return {"ok": True, "reason": ""}

def _verify_block(res: Dict[str, Any], context: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer strong verifier if available; otherwise construct from res/flags.
    if verify_with_mapper is not None:
        try:
            vb = verify_with_mapper(
                plan=res.get("plan"),
                state=res.get("state"),
                policy=policy,
                mapper_conf=(res.get("aux") or {}).get("mapper_conf"),
            )
            if isinstance(vb, dict) and isinstance(vb.get("ok"), bool):
                return vb
        except Exception:
            pass
    # Fallback: if runner reported ok/reason, use it; otherwise heuristics
    if isinstance(res.get("ok"), bool):
        rb = {"ok": bool(res["ok"]), "reason": (res.get("reason") or "")}
        if rb["ok"] or rb["reason"]:
            return rb
    return _fallback_verify(res, context)

# -----------------------
# Core execution
# -----------------------
def run_once(prompt: str, context: Dict[str, Any], policy: Dict[str, Any], rails: str, T: int) -> Dict[str, Any]:
    print(f"\n[DEBUG][run_once] prompt={prompt}, rails={rails}, T={T}")
    if not callable(run_micro):
        return {
            "top1": None,
            "verify": {"ok": False, "reason": "runner_missing"},
            "flags": {"runner_missing": True},
            "plan": {"sequence": []},
            "aux": {},
        }

    res = run_micro("defi", prompt, context=context, policy=policy, rails=rails, T=T)
    plan = res.get("plan") or {}
    seq  = plan.get("sequence") or plan.get("steps") or []

    mapper = (res.get("artifacts") or {}).get("mapper") or {}
    print("[DEBUG][mapper.intent]:", mapper.get("intent"))
    print("[DEBUG][mapper.raw]:", mapper)
    print("[DEBUG][run_once] run_micro output keys:", list(res.keys()))
    print("[DEBUG][run_once] label:", res.get("label"))

    # Build sequence fallback path: label → artifacts.mapper.intent
    extracted = _extract_top1_from_seq(seq)
    if not extracted or str(extracted).strip().lower() == "abstain":
        lbl = (res.get("label") or "").strip()
        if lbl and lbl.lower() != "abstain":
            seq = [lbl]
        else:
            intent = mapper.get("intent")
            if isinstance(intent, str) and intent.strip() and intent.lower() != "abstain":
                seq = [intent.strip()]

    top1 = _normalize_top1(_extract_top1_from_seq(seq))
    verify_blk = _verify_block(res, context, policy)

    out = {
        "top1": top1,
        "verify": verify_blk,
        "flags": res.get("flags") or {},
        "plan": plan if plan else {"sequence": seq if isinstance(seq, list) else []},
        "aux": res.get("aux") or {},
    }
    print("[DEBUG][run_once] Result top1:", out["top1"])
    print("[DEBUG][run_once] verify:", out["verify"])
    return out

def decision_from_verify(out: Dict[str, Any]) -> str:
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
    print(f"\n[DEBUG][check_expectations] case={case['name']}, expect_top1={case.get('expect_top1')}")
    print("[DEBUG][check_expectations] out.top1:", out.get("top1"))
    print("[DEBUG][check_expectations] verify:", out.get("verify"))

    if "expect_top1" in case and out.get("top1") != case["expect_top1"]:
        return False, f"expected top1={case['expect_top1']}, got={out.get('top1')}"
    if "expect_verify_ok" in case:
        v = out.get("verify") or {}
        if bool(v.get("ok")) != bool(case["expect_verify_ok"]):
            return False, f"expected verify.ok={case['expect_verify_ok']}, got={v}"
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
    TP = FP = TN = FN = 0
    abstain = 0
    for r in records:
        truth = r["truth"]     # "approve" or "reject"
        pred  = r["decision"]  # approve/reject
        v = (r.get("output", {}).get("verify") or {})
        reason = str(v.get("reason") or "").lower()
        tags = v.get("tags") or []
        flags = (r.get("output", {}).get("flags") or {})
        blob = " ".join([reason] + [str(t).lower() for t in tags] +
                        [f"{k}:{str(flags[k]).lower()}" for k in flags.keys()])
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
    omission_rate      = FN / max(1, TP + FN)
    abstain_rate       = abstain / max(1, len(records))
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

    # Exec-ok (approve)
    for i, case in enumerate(exec_ok, 1):
        name = case.get("name", f"exec_{i}")
        context = deep_merge(ctx_base, case.get("context_patch", {}))
        policy  = deep_merge(pol_base, case.get("policy_patch", {}))
        print(f"\n[DEBUG][run_suite] Running case: {name}")
        out = run_once(case["prompt"], context, policy, rails, T)
        ok, why = check_expectations(case, out)
        if not ok:
            failures.append(f"{name}: {why}")
        scenarios.append({
            "name": name, "type": "exec_ok", "truth": "approve",
            "prompt": case["prompt"], "runs": 1, "output": out,
            "decision": decision_from_verify(out),
            "ok": ok, "why": "" if ok else why,
        })

    # Edges (reject) — keep minimal for now
    for i, case in enumerate(edges, 1):
        name = case.get("name", f"edge_{i}")
        context = deep_merge(ctx_base, case.get("context_patch", {}))
        policy  = deep_merge(pol_base, case.get("policy_patch", {}))
        print(f"\n[DEBUG][run_suite] Running case: {name}")
        out = run_once(case["prompt"], context, policy, rails, T)
        ok, why = check_expectations(case, out)
        if not ok:
            failures.append(f"{name}: {why}")
        scenarios.append({
            "name": name, "type": "edge", "truth": "reject",
            "prompt": case["prompt"], "runs": runs, "output": out,
            "decision": decision_from_verify(out),
            "ok": ok, "why": "" if ok else why,
        })

    return scenarios, failures

def main():
    ap = argparse.ArgumentParser(description="Milestone 11 — Consolidated Tier-1 Benchmark (DeFi)")
    ap.add_argument("--rails", default="stage11")
    ap.add_argument("--baseline_rails", default="stage10")
    ap.add_argument("--T", type=int, default=180)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--context", default='{"oracle":{"age_sec":5,"max_age_sec":30}}')
    ap.add_argument("--policy",  default='{"ltv_max":0.75,"hf_min":1.0,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}')
    ap.add_argument("--edges", default="", help="JSONL path for extra edge cases")
    ap.add_argument("--exec",  default="", help="JSONL path for extra exec-ok cases")
    ap.add_argument("--compare_baseline", type=int, default=1, choices=[0,1])
    ap.add_argument("--no_baseline", action="store_true")
    ap.add_argument("--out_json", default=str(ARTIF / "defi_milestone11_summary.json"))
    ap.add_argument("--out_md",   default=str(ARTIF / "defi_milestone11_report.md"))
    args = ap.parse_args()

    ctx_base: Dict[str, Any] = json.loads(args.context)
    pol_base: Dict[str, Any] = json.loads(args.policy)

    edges = list(DEFAULT_EDGE_SUITE)
    exec_ok = list(DEFAULT_EXEC_SUITE)
    edges.extend(load_suite(args.edges))
    exec_ok.extend(load_suite(args.exec))

    started = time.time()
    all_sections: Dict[str, Any] = {}
    failures: List[str] = []

    # Primary rails
    scenarios_main, fails_main = run_suite(args.rails, args.T, args.runs, ctx_base, pol_base, edges, exec_ok)
    failures.extend(fails_main)
    metrics_main = consolidate_metrics([
        {"truth": sc["truth"], "decision": sc["decision"], "output": sc["output"]} for sc in scenarios_main
    ])
    all_sections["main"] = {"rails": args.rails, "metrics": metrics_main, "scenarios": scenarios_main}

    # Baseline rails (optional)
    metrics_base = None
    if args.compare_baseline and not args.no_baseline:
        scenarios_base, _ = run_suite(args.baseline_rails, args.T, args.runs, ctx_base, pol_base, edges, exec_ok)
        metrics_base = consolidate_metrics([
            {"truth": sc["truth"], "decision": sc["decision"], "output": sc["output"]} for sc in scenarios_base
        ])
        all_sections["baseline"] = {"rails": args.baseline_rails, "metrics": metrics_base, "scenarios": scenarios_base}

    H = metrics_main["hallucination_rate"]
    O = metrics_main["omission_rate"]
    status_ok = (H <= 0.01) and (1.0 - O >= 0.90)

    summary = {
        "milestone": "defi_milestone11",
        "status": "pass" if status_ok else "fail",
        "rails": args.rails,
        "baseline_rails": args.baseline_rails if (args.compare_baseline and not args.no_baseline) else "",
        "T": args.T,
        "runs": args.runs,
        "policy": pol_base,
        "context": ctx_base,
        "sections": all_sections,
        "failures": failures,
        "elapsed_sec": round(time.time() - started, 3),
    }

    Path(args.out_json).write_text(json.dumps(summary, indent=2))

    # Markdown report
    def _fmt_metrics(m: Dict[str, float]) -> List[str]:
        return [
            f"- accuracy: **{m['accuracy_exact']*100:.1f}%**",
            f"- precision: **{m['precision']*100:.1f}%**  •  recall: **{m['recall']*100:.1f}%**  •  F1: **{m['f1']*100:.1f}%**",
            f"- hallucination: **{m['hallucination_rate']*100:.2f}%**  •  omission: **{m['omission_rate']*100:.2f}%**",
            f"- abstain: **{m['abstain_rate']*100:.2f}%**",
        ]

    lines: List[str] = []
    lines.append("# Milestone 11 — Consolidated Tier-1 Benchmark (DeFi)\n")
    lines.append(f"- Status: {'✅ pass' if status_ok else '❌ fail'}")
    lines.append(f"- Rails: `{args.rails}`  •  Baseline: `{args.baseline_rails if metrics_base else '—'}`  •  T={args.T}  •  runs={args.runs}\n")
    lines.append("## Metrics — Stage-11\n")
    lines += _fmt_metrics(metrics_main) + [""]
    if metrics_base:
        lines.append("## Metrics — Baseline\n")
        lines += _fmt_metrics(metrics_base) + [""]

    if failures:
        lines.append("## Failures")
        for f in failures:
            lines.append(f"- {f}")
        lines.append("")

    lines.append("## Scenarios (brief)")
    for sec_name, sec in all_sections.items():
        lines.append(f"### {sec_name.capitalize()} — `{sec['rails']}`")
        for sc in sec["scenarios"]:
            out = sc["output"]; v = (out.get("verify") or {})
            lines.append(f"- **{sc['name']}** [{sc['type']}] → decision: `{sc['decision']}` • top1: `{out.get('top1')}` • verify.ok: `{v.get('ok')}` • reason: `{v.get('reason','')}`")
        lines.append("")

    Path(args.out_md).write_text("\n".join(lines))
    print(json.dumps({"ok": status_ok, "summary": args.out_json, "report": args.out_md}))

if __name__ == "__main__":
    main()
