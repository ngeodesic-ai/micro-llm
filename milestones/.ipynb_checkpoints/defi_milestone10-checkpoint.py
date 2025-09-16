#!/usr/bin/env python3
# milestones/defi_milestone10.py
from __future__ import annotations
import argparse, json, time, random, copy, pathlib, hashlib
from typing import Dict, Any, List, Tuple
from micro_lm.pipelines.runner import run_micro
import numpy as np



"""
# Clean run (no perturb):
python3 milestones/defi_milestone10.py \
  --rails stage11 --runs 5 \
  --policy '{"ltv_max":0.75,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}' \
  --context '{"oracle":{"age_sec":5,"max_age_sec":30}}'

python3 milestones/inspect_summary.py .artifacts/defi_milestone10_summary.json

# With perturbation robustness:
python3 milestones/defi_milestone10.py \
  --rails stage11 --runs 5 --perturb --perturb_k 3 \
  --policy '{"ltv_max":0.75,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}' \
  --context '{"oracle":{"age_sec":5,"max_age_sec":30}}'
python3 milestones/inspect_summary.py .artifacts/defi_milestone10_summary.json

# Freeze knobs in a small config file:
cat > configs/m10_defaults.json <<'JSON'
{"rails":"stage11","runs":5,"T":180,
 "policy":{"ltv_max":0.75,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}},
 "context":{"oracle":{"age_sec":5,"max_age_sec":30}},
 "perturb": true, "perturb_k": 3}
JSON

# Quick CSV view:
jq -r '["name","ok","top1","verify_ok"],
  (.scenarios[] | select(.name|endswith("_perturb")|not) |
   [ .name, (.ok|tostring), (.output.top1 // "None"), (.output.verify.ok|tostring) ]) | @csv' \
   .artifacts/defi_milestone10_summary.json > .artifacts/m10_scenarios.csv
"""

ARTIF = pathlib.Path(".artifacts"); ARTIF.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = ARTIF / "defi_milestone10_summary.json"
REPORT_PATH  = ARTIF / "defi_milestone10_report.md"

# --- Canonical scenarios (self-contained; no M9 import) ---
SCENARIOS: List[Dict[str, Any]] = [
    # exec paths
    {"name": "deposit_eth",   "prompt": "deposit 10 ETH into aave",
     "expect_top1": "deposit_asset", "expect_verify_ok": True},
    {"name": "swap_eth_usdc", "prompt": "swap 2 ETH for USDC",
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

def _np_json(o):
    if isinstance(o, (np.integer,)):   return int(o)
    if isinstance(o, (np.floating,)):  return float(o)
    if isinstance(o, (np.ndarray,)):   return o.tolist()
    return str(o)

def _ctx_hash(ctx: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(ctx, sort_keys=True).encode()).hexdigest()[:8]

def _json_arg(s: str | None) -> Dict[str, Any]:
    if not s: return {}
    s = s.strip()
    if s.startswith("{"): return json.loads(s)
    p = pathlib.Path(s); return json.loads(p.read_text()) if p.exists() else {}

def run_once(prompt: str, context: Dict[str, Any], policy: Dict[str, Any], rails: str, T: int) -> Dict[str, Any]:
    res = run_micro("defi", prompt, context=context, policy=policy, rails=rails, T=T)
    seq = (res.get("plan") or {}).get("sequence") or []
    top1 = seq[0] if seq else None
    verify = res.get("verify") or {}
    flags  = res.get("flags")  or {}
    aux    = res.get("aux")    or {}
    # keep only json-safe, scalar-ish aux bits
    out = {
        "prompt": prompt,
        "top1": top1,
        "flags": {k: flags[k] for k in flags if isinstance(flags[k], (str, int, float, bool))},
        "verify": {
            "ok": bool(verify.get("ok", False)),
            "reason": (verify.get("reason") or "")
        },
        "aux": {
            "mapper_confidence": float(aux.get("mapper_confidence") or 0.0)
        },
    }
    return out

def check_expect(single: Dict[str, Any], expect: Dict[str, Any]) -> Tuple[bool, str]:
    # top1
    expected_top1 = expect.get("expect_top1", "...skip...")
    if expected_top1 != "...skip..." and single["top1"] != expected_top1:
        return False, f"expected top1={expected_top1}, got={single['top1']}"
    # verify.ok
    evo = expect.get("expect_verify_ok")
    if evo is not None:
        ok = bool((single.get("verify") or {}).get("ok"))
        if ok != bool(evo):
            return False, f"expected verify.ok={evo}, got={single.get('verify')}"
    # reason substring (soft)
    sub = (expect.get("expect_reason_contains") or "").lower()
    if sub:
        reason = ((single.get("verify") or {}).get("reason") or "").lower()
        if sub not in reason:
            return False, f"expected reason contains '{sub}', got '{reason}'"
    return True, ""

def stability(prompt: str, runs: int, context: Dict[str, Any], policy: Dict[str, Any], rails: str, T: int) -> Dict[str, Any]:
    tops, oks = [], []
    for _ in range(runs):
        out = run_once(prompt, context, policy, rails, T)
        tops.append(out["top1"])
        oks.append(bool((out["verify"] or {}).get("ok")))
    return {"stable_top1": len(set(tops)) == 1, "top1_list": tops, "ok_list": oks}

# --- perturbation helpers (small & safe) ---
NUM_WORD_JITTER = [("deposit","add"),("swap","exchange"),("into","to"),("for","into")]
def perturb_prompt(p: str, seed: int) -> str:
    rng = random.Random(seed)
    words = p.split()
    if len(words) <= 1: return p
    # synonym swap
    for a, b in NUM_WORD_JITTER:
        if rng.random() < 0.5 and a in words:
            words = [b if w==a and rng.random()<0.7 else w for w in words]
    # tiny numeric jitter (e.g., 10→9.9 or 2→2.1)
    for i, w in enumerate(words):
        try:
            val = float(w)
            jitter = 1.0 + rng.choice([-0.01, 0.01])
            words[i] = f"{val*jitter:.3g}"
        except ValueError:
            pass
    return " ".join(words)

def perturb_context(ctx: Dict[str, Any], seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    ctx2 = copy.deepcopy(ctx)
    # nudge oracle age bounds by ±1s within sane bounds
    o = ctx2.setdefault("oracle", {})
    for k in ("age_sec","max_age_sec"):
        if k in o and isinstance(o[k], (int, float)):
            o[k] = max(1, int(round(o[k] * (1.0 + rng.choice([-0.05, 0.05])))))
    return ctx2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rails", default="stage11")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--T", type=int, default=180)
    ap.add_argument("--context", default='{"oracle":{"age_sec":5,"max_age_sec":30}}')
    ap.add_argument("--policy",  default='{"ltv_max":0.75, "mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}')
    ap.add_argument("--perturb", action="store_true", help="enable prompt/context perturbations")
    ap.add_argument("--perturb_k", type=int, default=3, help="number of perturbation variants per scenario")
    args = ap.parse_args()

    ctx_base = _json_arg(args.context) or {"oracle":{"age_sec":5,"max_age_sec":30}}
    pol_base = _json_arg(args.policy)  or {"ltv_max":0.75}
    started = time.time()

    scenarios_out: List[Dict[str, Any]] = []
    failures: List[str] = []
    overall_ok = True

    # A) Clean runs + stability
    for case in SCENARIOS:
        single = run_once(case["prompt"], ctx_base, pol_base, args.rails, args.T)
        ok, why = check_expect(single, case)
        stab = stability(case["prompt"], args.runs, ctx_base, pol_base, args.rails, args.T)
        if not ok or (case.get("expect_top1") and not stab["stable_top1"]):
            overall_ok = False
            failures.append(f"{case['name']}: {why or 'top1 not stable'}")
        scenarios_out.append({
            "name": case["name"],
            # inspector expects: sc["output"]["prompt"] and sc["output"]["top1"]
            "output": {
                "prompt": single["prompt"],
                "top1": single["top1"],
                "verify": {"ok": bool((single.get("verify") or {}).get("ok"))}
            },
            # keep these for humans / other tools
            "ok": ok and ((case.get("expect_top1") is None) or stab["stable_top1"]),
            "prompt": single["prompt"],          # convenience duplicate
            "top1": single["top1"],              # convenience duplicate
            "verify_ok": bool((single.get("verify") or {}).get("ok")),
            "stable_top1": stab["stable_top1"],
            "top1_list": stab["top1_list"],
            "reason": "" if ok else why
        })

    # B) Perturbation robustness (optional)
    if args.perturb:
        seed0 = 20259
        for case in SCENARIOS:
            case_ok = True
            pert_runs = []
            for j in range(max(1, args.perturb_k)):
                p_prompt = perturb_prompt(case["prompt"], seed0 + j)
                p_ctx    = perturb_context(ctx_base, seed0 + j)
                out      = run_once(p_prompt, p_ctx, pol_base, args.rails, args.T)
                ok_j, why_j = check_expect(out, case)
                if not ok_j:
                    case_ok = False
                pert_runs.append({"variant": j, "ok": ok_j, "why": "" if ok_j else why_j, "output": out})
            if not case_ok:
                overall_ok = False
                failures.append(f"{case['name']}: perturbation failures")
            rep = pert_runs[0]["output"]
            scenarios_out.append({
                "name": case["name"] + "_perturb",
                "ok": case_ok,
                "output": {
                    "prompt": rep["prompt"],
                    "top1": rep["top1"],
                    "verify": {"ok": bool((rep.get("verify") or {}).get("ok"))}
                },
                "runs": pert_runs
            })

    summary = {
        "milestone": "defi_milestone10",
        "status": "pass" if overall_ok else "fail",
        "rails": args.rails, "T": args.T, "runs": args.runs,
        "perturb": bool(args.perturb), "perturb_k": args.perturb_k,
        "scenarios": scenarios_out,
        "failures": failures,
        "elapsed_sec": round(time.time() - started, 3),
        "context_hash": _ctx_hash(ctx_base),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=_np_json))

    # Human report
    lines = []
    lines.append(f"# Milestone 10 Report\n")
    lines.append(f"- Status: {'✅ pass' if overall_ok else '❌ fail'}")
    lines.append(f"- Rails: `{args.rails}`  •  T={args.T}  •  runs={args.runs}  •  perturb={args.perturb} (k={args.perturb_k})\n")
    for sc in scenarios_out:
        name = sc["name"]; ok = sc.get("clean_ok", sc.get("ok", True))
        lines.append(f"## {name} — {'OK' if ok else 'FAIL'}")
        if "output" in sc:
            o = sc["output"]
            lines.append(f"- prompt: `{o['prompt']}`")
            lines.append(f"- top1: `{o['top1']}`  •  verify.ok: `{bool((o.get('verify') or {}).get('ok'))}`")
            if "stability" in sc:
                s = sc["stability"]
                lines.append(f"- stable_top1: `{s['stable_top1']}`  •  top1_list: {s['top1_list']}")
            if sc.get("reason"): lines.append(f"- reason: {sc['reason']}")
        if "runs" in sc:
            bad = [r for r in sc["runs"] if not r["ok"]]
            lines.append(f"- perturb variants: {len(sc['runs'])}  •  fails: {len(bad)}")
    REPORT_PATH.write_text("\n".join(lines))

    print(json.dumps({"ok": overall_ok, "summary": str(SUMMARY_PATH), "report": str(REPORT_PATH)}, indent=2))

if __name__ == "__main__":
    main()
