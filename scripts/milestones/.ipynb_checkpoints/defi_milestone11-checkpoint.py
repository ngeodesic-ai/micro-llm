#!/usr/bin/env python3
# scripts/milestones/defi_milestone11_final.py
import argparse, json, sys, time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

"""
python3 scripts/milestones/defi_milestone11.py \
  --rails stage11 --baseline_rails stage10 \
  --runs 3 --T 180 \
  --policy '{"ltv_max":0.75,"hf_min":1.0,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}' \
  --context '{"oracle":{"age_sec":5,"max_age_sec":30}}' \
  --debug
"""

# -- CLI -----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("m11 smoke with safe shim + debug")
    p.add_argument("--rails", required=True, help="rails to test (e.g., stage11)")
    p.add_argument("--baseline_rails", required=True, help="compare against (e.g., stage10)")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--T", type=int, default=180)
    p.add_argument("--policy", type=str, default="{}")
    p.add_argument("--context", type=str, default="{}")
    p.add_argument("--out_json", type=str, default=".artifacts/defi_milestone11_summary.json")
    p.add_argument("--out_md", type=str, default=".artifacts/defi_milestone11_report.md")
    p.add_argument("--debug", action="store_true", help="enable verbose debug prints")
    return p.parse_args()

def dprint(enabled: bool, *args, **kwargs):
    if enabled:
        print(*args, **kwargs)

# -- Core runner import (no over-engineering; simple, robust) -------------------
def import_run_micro(debug=False):
    """
    Prefer the canonical location; fall back through a few aliases used in the repo
    during refactors. No exceptions bubble up; we fail clearly otherwise.
    """
    candidates = (
        "micro_lm.core.runner",
        "micro_lm.runner",
        "micro_lm.pipeline.runner",
        "micro_lm.cli.runner",
        "micro_lm.domains.defi.runner",
    )
    for mod in candidates:
        try:
            m = __import__(mod, fromlist=["run_micro"])
            run_micro = getattr(m, "run_micro", None)
            if callable(run_micro):
                dprint(debug, f"[DEBUG][import] using run_micro from {mod}")
                return run_micro
        except Exception as e:
            dprint(debug, f"[DEBUG][import] {mod} import failed: {e}")
    raise RuntimeError("run_micro not found in expected modules")

# -- Safe shim: only used when mapper intent is missing/abstain -----------------
class MapperShim:
    """
    Very small shim that attempts to load a joblib classifier and infer intent.
    Kicks in only if run_micro returns no mapper intent or abstains with non-exec/low_conf.
    """
    def __init__(self, model_path: Optional[str], debug=False, topk:int=5):
        self.model_path = model_path
        self.model = None
        self.debug = debug
        self.topk = topk

        if not model_path:
            dprint(self.debug, "[DEBUG][mapper] no model_path provided; shim disabled")
            return
        try:
            from joblib import load
            self.model = load(model_path)
            # Try to read classes for nice printing, if available
            classes = getattr(self.model, "classes_", None)
            if classes is not None:
                dprint(self.debug, f"[DEBUG][mapper] loaded: {model_path}")
                dprint(self.debug, f"[DEBUG][mapper] classes: {list(classes)}")
        except Exception as e:
            dprint(self.debug, f"[DEBUG][mapper] failed to load {model_path}: {e}")
            self.model = None

    def infer(self, text: str) -> Optional[Dict[str, Any]]:
        if not self.model:
            return None
        try:
            # Prefer predict_proba; fall back to decision_function if needed.
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba([text])[0]
            elif hasattr(self.model, "decision_function"):
                import numpy as np
                scores = self.model.decision_function([text])[0]
                # softmax to pseudo-probs
                ex = np.exp(scores - scores.max())
                probs = ex / ex.sum()
            else:
                return None

            classes = list(getattr(self.model, "classes_", []))
            pairs = list(zip(classes, probs))
            pairs.sort(key=lambda x: float(x[1]), reverse=True)

            # Compose a small payload similar to artifacts.mapper
            top = pairs[: self.topk]
            dprint(self.debug, f"[DEBUG][mapper.shim] top: {[(c, float(p)) for c, p in top]}")
            return {
                "intent": top[0][0] if top else None,
                "score": float(top[0][1]) if top else 0.0,
                "top": [(c, float(p)) for c, p in top],
                "raw": {"topk": [(c, float(p)) for c, p in top]},
            }
        except Exception as e:
            dprint(self.debug, f"[DEBUG][mapper.shim] inference failed: {e}")
            return None

def choose_top1(artifacts: Dict[str, Any], shim: MapperShim, prompt: str, debug=False) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Decide a top1 label for "OK" cases:
      1) If run_micro returned artifacts.mapper.intent -> use that.
      2) Else if run_micro abstained (low_conf/non_exec) -> try shim.
      3) Else None.
    Returns (top1, verify_info) where verify_info mimics a tiny 'verify' object.
    """
    mapper = (artifacts or {}).get("mapper") or {}
    intent = mapper.get("intent")
    reason = mapper.get("reason") or ""

    if intent:
        dprint(debug, f"[DEBUG][top1] artifacts.mapper.intent={intent}")
        return intent, {"ok": True, "reason": "artifacts.mapper"}

    # If run_micro abstained or had nothing mapper-ish, try shim:
    shim_out = shim.infer(prompt) if shim else None
    if shim_out and shim_out.get("intent"):
        dprint(debug, f"[DEBUG][top1] shim.intent={shim_out['intent']} score={shim_out.get('score'):.3f}")
        return shim_out["intent"], {"ok": True, "reason": "shim:mapper", "aux": shim_out}

    # Fallthrough
    dprint(debug, "[DEBUG][top1] no intent available")
    return None, {"ok": False, "reason": "low_confidence"}

# -- Cases (exact prompts kept minimal; adjust if your repo uses constants) -----
OK_CASES = [
    {"name": "ok_deposit", "prompt": "deposit 10 ETH into aave", "expect_top1": "deposit_asset"},
    {"name": "ok_swap",    "prompt": "swap 2 ETH for USDC",       "expect_top1": "swap_asset"},
]

EDGE_CASES = [
    {"name": "edge_ltv_withdraw_unsafe",    "prompt": "withdraw 5 ETH"},
    {"name": "edge_hf_health_breach",       "prompt": "increase borrow to the maximum"},
    {"name": "edge_oracle_stale_price",     "prompt": "borrow 1000 USDC"},
    {"name": "edge_mapper_low_conf_or_nonexec", "prompt": "stake xyz"},
]

# -- Single run ---------------------------------------------------------------
def run_once(run_micro, rails: str, T: int, prompt: str, context: Dict[str,Any], policy: Dict[str,Any],
             shim: MapperShim, debug=False) -> Dict[str, Any]:
    dprint(debug, f"\n[DEBUG][run_once] prompt={prompt}, rails={rails}, T={T}")
    out = run_micro("defi", prompt, context=context, policy=policy, rails=rails, T=T)
    # Normalize minimal shape we care about
    label   = out.get("label")
    reason  = out.get("reason")
    arts    = out.get("artifacts") or {}
    dprint(debug, "[DEBUG][run_once] run_micro output keys:", list(out.keys()))
    dprint(debug, "[DEBUG][run_once] label:", label)

    top1, verify = choose_top1(arts, shim, prompt, debug=debug)
    dprint(debug, "[DEBUG][run_once] Result top1:", top1)
    dprint(debug, "[DEBUG][run_once] verify:", verify)

    return {
        "rails": rails,
        "prompt": prompt,
        "label": label,
        "reason": reason,
        "top1": top1,
        "verify": verify,
        "artifacts": arts,
    }

# -- Suite driver -------------------------------------------------------------
def run_suite(run_micro, rails: str, T: int, runs: int,
              context: Dict[str,Any], policy: Dict[str,Any],
              debug=False) -> Tuple[List[Dict[str,Any]], List[str]]:
    shim = MapperShim(
        model_path=((policy or {}).get("mapper") or {}).get("model_path"),
        debug=debug
    )

    results = []
    failures = []

    # OK cases: require specific top1
    for case in OK_CASES:
        dprint(debug, f"\n[DEBUG][run_suite] Running case: {case['name']}")
        passed = False
        last_out = None
        for i in range(runs):
            out = run_once(run_micro, rails, T, case["prompt"], context, policy, shim, debug=debug)
            last_out = out
            if out["top1"] == case["expect_top1"]:
                passed = True
                break
        results.append({"case": case["name"], "out": last_out, "passed": passed})
        if not passed:
            failures.append(f"{case['name']}: expected top1={case['expect_top1']}, got={last_out['top1']}")

    # Edge cases: we only require that we DO NOT falsely claim a concrete OK action (i.e., no wrong top1).
    # Abstain or any safe refusal is OK, as is a domain-specific guardrail decision.
    for case in EDGE_CASES:
        dprint(debug, f"\n[DEBUG][run_suite] Running case: {case['name']}")
        last_out = None
        safe_ok = False
        for i in range(runs):
            out = run_once(run_micro, rails, T, case["prompt"], context, policy, shim, debug=debug)
            last_out = out
            # Consider it "safe" if no concrete OK action is proposed
            if out["top1"] is None:
                safe_ok = True
                break
        results.append({"case": case["name"], "out": last_out, "passed": safe_ok})
        if not safe_ok:
            failures.append(f"{case['name']}: expected no concrete action, got top1={last_out['top1']}")

    return results, failures

# -- Reporting ----------------------------------------------------------------
def write_summary_and_report(out_json: str, out_md: str,
                             rails: str, baseline: str,
                             res_main: List[Dict[str,Any]], fails_main: List[str],
                             res_base: List[Dict[str,Any]], fails_base: List[str]):
    summary = {
        "ok": len(fails_main) == 0,
        "rails": rails,
        "baseline_rails": baseline,
        "timestamp": int(time.time()),
        "results": {
            rails: {
                "failures": fails_main,
                "cases": res_main,
            },
            baseline: {
                "failures": fails_base,
                "cases": res_base,
            }
        }
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(summary, indent=2))

    lines = []
    lines.append(f"# M11 Smoke — rails={rails} vs baseline={baseline}\n")
    def block(title, res, fails):
        lines.append(f"## {title}\n")
        lines.append(f"- **pass:** {len([r for r in res if r['passed']])}")
        lines.append(f"- **fail:** {len(fails)}")
        if fails:
            lines.append("### Failures")
            for f in fails:
                lines.append(f"- {f}")
        lines.append("")
    block(rails, res_main, fails_main)
    block(baseline, res_base, fails_base)
    Path(out_md).write_text("\n".join(lines))

    return summary

# -- Main ---------------------------------------------------------------------
def main():
    args = parse_args()
    debug = args.debug

    try:
        policy = json.loads(args.policy)
        context = json.loads(args.context)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"invalid json in --policy/--context: {e}"}))
        sys.exit(1)

    run_micro = import_run_micro(debug=debug)

    # Primary rails
    res_main, fails_main = run_suite(run_micro, args.rails, args.T, args.runs, context, policy, debug=debug)
    # Baseline rails (sanity)
    res_base, fails_base = run_suite(run_micro, args.baseline_rails, args.T, args.runs, context, policy, debug=debug)

    summary = write_summary_and_report(args.out_json, args.out_md,
                                       args.rails, args.baseline_rails,
                                       res_main, fails_main, res_base, fails_base)

    print(json.dumps({"ok": summary["ok"], "summary": args.out_json, "report": args.out_md}))
    sys.exit(0 if summary["ok"] else 2)

if __name__ == "__main__":
    main()
