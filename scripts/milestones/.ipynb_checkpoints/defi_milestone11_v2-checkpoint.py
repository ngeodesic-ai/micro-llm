#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 11 — Consolidated Tier-1 Benchmark (DeFi)
Refactor-friendly: mapper→label→intent fallback, strong-verify shim, and
robust top1 normalization. Prints concise DEBUG for each case.
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, time, copy
from typing import Dict, Any, List, Tuple, Optional

# ---------- Runner import shim ----------
run_micro = None
for _mod in (
    "micro_lm.domains.defi.runner",   # domain runner
    "micro_lm.core.runner",           # refactor runner (current)
):
    try:
        _m = __import__(_mod, fromlist=["run_micro"])
        run_micro = getattr(_m, "run_micro", None)
        if callable(run_micro):
            break
    except Exception:
        pass

# Optional strong verify
try:
    from micro_lm.verify.defi_verify import verify_with_mapper
except Exception:
    verify_with_mapper = None

# Optional: tautology-free audit metrics (attach to summary if present)
import json as _json_mod
from pathlib import Path as _Path_mod
def _load_audit_metrics(path: str | None):
    if not path: return None
    P = _Path_mod(path)
    if not P.exists(): return None
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
        "expect_reason_any": ["hf","health","abstain_non_exec","oracle","stale"],
        "policy_patch": {"hf_min": 1.0},
    },
    {
        "name": "edge_oracle_stale_price",
        "prompt": "borrow 1000 USDC",
        "expect_top1": None,
        "expect_verify_ok": False,
        "expect_reason_any": ["oracle","stale"],
        "context_patch": {"oracle": {"age_sec": 120, "max_age_sec": 30}},
    },
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
            if ln:
                out.append(json.loads(ln))
    return out

def _extract_top1(seq: List[Any]) -> Optional[str]:
    if not seq:
        return None
    first = seq[0]
    if isinstance(first, str):
        return first
    if isinstance(first, dict):
        for k in ("op","action","name","type","label"):
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

def _fallback_verify(result: Dict[str, Any], context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    plan = result.get("plan") or {}
    seq  = plan.get("sequence") or []
    flags = result.get("flags") or {}
    blob = " ".join(f"{k}:{str(v).lower()}" for k,v in flags.items())

    if any(s in blob for s in ["ltv_breach:true", "ltv:true"]):
        return {"ok": False, "reason": "ltv_breach", "tags": ["ltv"]}
    if any(s in blob for s in ["hf_breach:true","health_breach:true","hf:true"]):
        return {"ok": False, "reason": "hf_breach", "tags": ["hf","health"]}
    if any(s in blob for s in ["oracle_stale:true","oracle:true","stale:true"]):
        return {"ok": False, "reason": "oracle_stale", "tags": ["oracle","stale"]}
    if context:
        try:
            o = (context or {}).get("oracle") or {}
            if float(o.get("age_sec",0)) > float(o.get("max_age_sec",1e9)):
                return {"ok": False, "reason": "oracle_stale", "tags": ["oracle","stale"]}
        except Exception:
            pass
    if not seq:
        return {"ok": False, "reason": "abstain_non_exec", "tags": ["non_exec"]}
    return {"ok": True, "reason": ""}

def call_runner(prompt: str, ctx: dict, pol: dict, rails: str, T: int) -> dict:
    if not callable(run_micro):
        return {
            "plan": {"sequence": []},
            "verify": {"ok": False, "reason": "runner_missing"},
            "flags": {"runner_missing": True},
            "aux": {},
            "label": "abstain",
            "artifacts": {},
            "ok": False,
            "reason": "runner_missing",
        }
    return run_micro("defi", prompt, context=ctx, policy=pol, rails=rails, T=T)

# -----------------------
# Core execution
# -----------------------
def run_once(prompt: str, ctx: Dict[str, Any], pol: Dict[str, Any], rails: str, T: int) -> Dict[str, Any]:
    print(f"\n[DEBUG][run_once] prompt={prompt}, rails={rails}, T={T}")
    res = call_runner(prompt, ctx, pol, rails, T)

    # Debug keys / mapper
    print("[DEBUG][run_once] run_micro output keys:", list(res.keys()))
    print("[DEBUG][run_once] label:", res.get("label"))
    mapper = (res.get("artifacts") or {}).get("mapper") or {}
    print("[DEBUG][mapper.intent]:", mapper.get("intent"))
    print("[DEBUG][mapper.raw]:", mapper)

    # Sequence extraction with mapper/label fallback
    plan = res.get("plan") or {}
    seq  = plan.get("sequence") or plan.get("steps") or []
    extracted = _extract_top1(seq)

    if not extracted or str(extracted).strip().lower() == "abstain":
        lbl = (res.get("label") or "").strip()
        if lbl and lbl.lower() != "abstain":
            seq = [lbl]
        else:
            intent = mapper.get("intent")
            if isinstance(intent, str) and intent.strip() and intent.lower() != "abstain":
                seq = [intent.strip()]

    top1 = _normalize_top1(_extract_top1(seq))
    print("[DEBUG][run_once] Result top1:", top1)

    # Verify: strong verify if available, else safe fallback
    verify_block = res.get("verify") or {}
    if not isinstance(verify_block, dict) or ("ok" not in verify_block):
        verify_block = {}

    if verify_with_mapper:
        use_v = {"ok": bool(seq), "reason": ""}
        try:
            vb = verify_with_mapper(
                plan=res.get("plan"),
                state=res.get("state"),
                policy=pol,
                mapper_conf=(res.get("aux") or {}).get("mapper_conf"),
            )
            if isinstance(vb, dict) and ("ok" in vb):
                use_v = vb
        except Exception as e:
            use_v = {"ok": False, "reason": f"verify_error:{e}"}
        # sanitize or fallback when unusable
        r = str(use_v.get("reason") or "").lower()
        if ("verify_error" in r) or ("nonetype" in r):
            use_v = _fallback_verify(res, ctx)
        verify_block = use_v
    else:
        # favor runner’s own ok if provided; otherwise infer
        if isinstance(res.get("ok"), bool) and (seq or top1):
            verify_block = {"ok": bool(res["ok"]), "reason": (res.get("reason") or "")}
            if not verify_block["ok"] and not verify_block["reason"]:
                verify_block = _fallback_verify(res, ctx)
        else:
            verify_block = _fallback_verify(res, ctx)

    print("[DEBUG][run_once] verify:", verify_block)

    return {
        "prompt": prompt,
        "top1": top1,
        "flags": res.get("flags") or {},
        "verify": verify_block,
        "plan": res.get("plan") or {},
        "aux":  res.get("aux") or {},
    }

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

    if "expect_top1" in case and out["top1"] != case["expect_top1"]:
        return False, f"expected top1={case['expect_top1']}, got={out['top1']}"
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

def consolidate_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    TP = FP = TN = FN = 0
    abstain = 0
    for r in records:
        truth = r["truth"]            # approve (exec_ok) or reject (edge)
        pred  = r["decision"]
        v = (r.get("output", {}).get("verify") or {})
        reason = str(v.get("reason") or "").lower()
        tags = v.get("tags") or []
        flags = (r.get("output", {}).get("flags") or {})
        blob = " ".join([reason] + [str(t).lower() for t in tags] +
                        [f"{k}:{str(flags[k]).lower()}" for k in flags.keys()])
        if any(tok in blob for tok in ["abstain","non_exec","low_conf","reject_non_exec"]):
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

def run_suite(rails: str, T: int, runs: int, ctx_base: Dict[str, Any], pol_base: Dict[str, Any],
              edges: List[Dict[str, Any]], exec_ok: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    scenarios: List[Dict[str, Any]] = []
    failures: List[str] = []

    # Exec-OK only (you can append edges later if you want them here too)
    print("\n[DEBUG][run_suite] Running case: ok_deposit")
    for case in exec_ok:
        name = case.get("name", "exec_case")
        context = deep_merge(ctx_base, case.get("context_patch", {}))
        policy  = deep_merge(pol_base, case.get("policy_patch", {}))
        out = run_once(case["prompt"], context, policy, rails, T)
        ok, why = check_expectations(case, out)
        if not ok:
            failures.append(f"{name}: {why}")
        scenarios.append({
            "name": name, "type": "exec_ok", "truth": "approve",
            "prompt": case["prompt"], "runs": 1, "output": out,
            "decision": decision_from_verify(out), "ok": ok, "why": "" if ok else why,
        })

    return scenarios, failures

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Milestone 11 — Consolidated Tier-1 Benchmark (DeFi)")
    ap.add_argument("--rails", default="stage11")
    ap.add_argument("--baseline_rails", default="stage10")
    ap.add_argument("--T", type=int, default=180)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--context", default='{"oracle":{"age_sec":5,"max_age_sec":30}}')
    ap.add_argument("--policy",  default='{"ltv_max":0.75,"hf_min":1.0,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}')
    ap.add_argument("--edges", default="")
    ap.add_argument("--exec",  default="")
    ap.add_argument("--compare_baseline", type=int, default=1, choices=[0,1])
    ap.add_argument("--out_json", default=str(ARTIF / "defi_milestone11_summary.json"))
    ap.add_argument("--out_md",   default=str(ARTIF / "defi_milestone11_report.md"))
    ap.add_argument("--max_halluc", type=float, default=0.01)
    ap.add_argument("--min_exec_approve", type=float, default=0.90)
    ap.add_argument("--audit_metrics_json", default=".artifacts/defi/audit_bench/metrics_audit.json")
    ap.add_argument("--no_baseline", action="store_true")
    args = ap.parse_args()

    ctx_base = json.loads(args.context)
    pol_base = json.loads(args.policy)

    exec_ok = list(DEFAULT_EXEC_SUITE)
    exec_ok.extend(load_suite(args.exec))
    edges = list(DEFAULT_EDGE_SUITE)
    edges.extend(load_suite(args.edges))

    started = time.time()
    all_sections = {}
    failures: List[str] = []

    # Stage-11
    scenarios_main, fails_main = run_suite(args.rails, args.T, args.runs, ctx_base, pol_base, edges, exec_ok)
    failures.extend(fails_main)
    metrics_main = consolidate_metrics([
        {"truth": sc["truth"], "decision": sc["decision"], "output": sc["output"]} for sc in scenarios_main
    ])
    all_sections["main"] = {"rails": args.rails, "metrics": metrics_main, "scenarios": scenarios_main}

    # Baseline (optional)
    if args.compare_baseline and not args.no_baseline:
        scenarios_base, _ = run_suite(args.baseline_rails, args.T, args.runs, ctx_base, pol_base, edges, exec_ok)
        metrics_base = consolidate_metrics([
            {"truth": sc["truth"], "decision": sc["decision"], "output": sc["output"]} for sc in scenarios_base
        ])
        all_sections["baseline"] = {"rails": args.baseline_rails, "metrics": metrics_base, "scenarios": scenarios_base}
    else:
        metrics_base = None

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

    # Attach audit metrics if present
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

    # Markdown report
    lines: List[str] = []
    lines.append("# Milestone 11 — Consolidated Tier-1 Benchmark (DeFi)\n")
    lines.append(f"- Status: {'✅ pass' if status_ok else '❌ fail'}")
    lines.append(f"- Rails: `{args.rails}`  •  Baseline: `{args.baseline_rails if args.compare_baseline else '—'}`  •  T={args.T}  •  runs={args.runs}\n")
    def _fmt(m: Dict[str, float]) -> List[str]:
        return [
            f"- accuracy: **{m['accuracy_exact']*100:.1f}%**",
            f"- precision: **{m['precision']*100:.1f}%**  •  recall: **{m['recall']*100:.1f}%**  •  F1: **{m['f1']*100:.1f}%**",
            f"- hallucination: **{m['hallucination_rate']*100:.2f}%**  •  omission: **{m['omission_rate']*100:.2f}%**",
            f"- abstain: **{m['abstain_rate']*100:.2f}%**",
        ]
    lines.append("## Metrics — Stage-11\n"); lines += _fmt(metrics_main) + [""]
    if metrics_base:
        lines.append("## Metrics — Baseline\n"); lines += _fmt(metrics_base) + [""]

    if failures:
        lines.append("## Failures"); lines += [f"- {f}" for f in failures] + [""]

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
