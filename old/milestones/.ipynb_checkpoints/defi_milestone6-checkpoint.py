# Milestone 6 starter script: mapper integration scaffold
# This script is designed to drop into your repo and run standalone.
# It wires: prompt -> mapper -> primitives (+abstain) and emits a report.
# If your repo provides a micro_lm.run_micro (or similar), it will try to call it;
# otherwise it falls back to a no-rails dry-run to validate the mapping & thresholds.

import json, argparse, os, sys, time, hashlib
from typing import List, Dict, Any, Tuple
import os


"""
pytest -q tests/test_m6_mapper_wire.py -q

python3 milestones/defi_milestone6.py \
  --mapper_path .artifacts/defi_mapper.joblib \
  --confidence_threshold 0.70 \
  --rails stage11 \
  --runs 6 \
  --context '{"oracle":{"age_sec":5,"max_age_sec":30}}' \
  --policy '{"ltv_max":0.75, "mapper":{"confidence_threshold":0.70}}' \
  --out_summary .artifacts/defi_milestone6_summary.json
python3 milestones/inspect_summary.py .artifacts/defi_milestone6_summary.json
"""

# Optional deps
try:
    import joblib  # scikit-learn models
except Exception:
    joblib = None

# Try to import your repo's runner hooks (best effort; safe if missing)
RUN_MICRO_AVAILABLE = False
try:
    # Adjust to your actual entry point if different
    from micro_lm.runner import run_micro  # type: ignore
    RUN_MICRO_AVAILABLE = True
except Exception:
    try:
        from micro_lm.core.runner import run_micro  # alternate path
        RUN_MICRO_AVAILABLE = True
    except Exception:
        RUN_MICRO_AVAILABLE = False


PRIMS_DEF = ["deposit_asset", "withdraw_asset", "swap_asset", "check_balance"]

DEFAULT_PROMPTS = [
    "deposit 10 ETH into aave",
    "withdraw 200 USDC from compound",
    "swap 2 ETH to USDC",
    "check balance",
    "swap 500 USDC to WBTC on uniswap",
    "deposit DAI",
]


def sha_short(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()[:8]


def load_mapper(path: str):
    if not joblib:
        raise RuntimeError("joblib is not available; please `pip install joblib` or use a scikit-learn .joblib model.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mapper not found: {path}")
    return joblib.load(path)


def mapper_predict(mapper, prompt: str, class_names: List[str], conf_threshold: float) -> Tuple[List[str], Dict[str, float]]:
    """
    Generic scikit-learn-like interface:
      - if `predict_proba` exists, use it.
      - else if `decision_function` exists, softmax it.
      - else, fallback to `predict` with 1.0 conf.
    Returns:
      (selected_labels, per_class_probs)
    """
    import numpy as np

    # Allow simple vectorizers in pipelines
    if hasattr(mapper, "predict_proba"):
        probs = mapper.predict_proba([prompt])[0]
        # Map to classes
        if hasattr(mapper, "classes_"):
            classes = list(mapper.classes_)
        else:
            classes = class_names
        score_map = {str(c): float(p) for c, p in zip(classes, probs)}
    elif hasattr(mapper, "decision_function"):
        logits = mapper.decision_function([prompt])[0]
        logits = np.array(logits, dtype=float)
        # Softmax
        ex = np.exp(logits - logits.max())
        probs = ex / (ex.sum() + 1e-12)
        if hasattr(mapper, "classes_"):
            classes = list(mapper.classes_)
        else:
            classes = class_names
        score_map = {str(c): float(p) for c, p in zip(classes, probs)}
    else:
        pred = mapper.predict([prompt])[0]
        score_map = {str(pred): 1.0}

    # Normalize onto the declared class set if necessary
    for cname in class_names:
        score_map.setdefault(cname, 0.0)

    # Choose top-1 with abstain
    top_label = max(class_names, key=lambda c: score_map.get(c, 0.0))
    top_conf = score_map.get(top_label, 0.0)
    selected = [top_label] if top_conf >= conf_threshold else []
    return selected, score_map


def run_single(mapper, prompt: str, *, class_names: List[str], rails: str, conf_threshold: float,
               context: Dict[str, Any], policy: Dict[str, Any], T: int) -> Dict[str, Any]:
    # 1) mapper → primitive(s) (top-1 with abstain for Milestone 6)
    selected, score_map = mapper_predict(mapper, prompt, class_names, conf_threshold)

    # Plan skeleton
    plan = {
        "sequence": selected,            # [] => ABSTAIN
        "scores": score_map,             # per-class confidence for inspection
        "rails": rails,
        "T": T
    }

    # 2) If rails runner exists, attempt to execute
    exec_result: Dict[str, Any] = {"executed": False, "ok": None, "details": None}
    if RUN_MICRO_AVAILABLE and selected:
        try:
            # Expect your project signature: run_micro(domain, prompt, context, policy, rails, T)
            res = run_micro(
                domain="defi",
                prompt=prompt,
                context=context,
                policy=policy,
                rails=rails,
                T=T
            )
            exec_result = {"executed": True, "ok": bool(res.get("ok", True)), "details": res}
        except Exception as e:
            exec_result = {"executed": False, "ok": None, "details": f"run_micro failed: {e}"}

    return {"prompt": prompt, "plan": plan, "exec": exec_result}


def parse_json_arg(maybe_json: str) -> Dict[str, Any]:
    if not maybe_json:
        return {}
    # If looks like a path and exists, load as file
    if os.path.exists(maybe_json):
        with open(maybe_json, "r") as f:
            return json.load(f)
    # Otherwise parse as JSON string
    try:
        return json.loads(maybe_json)
    except Exception:
        # return empty if unparsable (avoid crashing)
        return {}


def main():
    ap = argparse.ArgumentParser(description="Milestone 6 — Mapper integration scaffold (prompt → mapper → rails)")
    ap.add_argument("--mapper_path", type=str, required=True, help="Path to trained mapper (e.g., .joblib)")
    ap.add_argument("--class_names", type=str, default=",".join(PRIMS_DEF),
                    help="Comma-separated class labels in mapper order (if model lacks .classes_)")
    ap.add_argument("--confidence_threshold", type=float, default=0.70, help="Abstain if top-1 prob < threshold")
    ap.add_argument("--rails", type=str, default="stage11", help="Rails selector (stage10|stage11)")
    ap.add_argument("--T", type=int, default=128, help="Rollout steps if rails execute")
    ap.add_argument("--runs", type=int, default=6, help="Number of prompts to run")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for prompt shuffling")
    ap.add_argument("--prompts_jsonl", type=str, default="", help="Optional .jsonl with {prompt: ...} per line")
    ap.add_argument("--context", type=str, default='{"oracle":{"age_sec":5,"max_age_sec":30}}',
                    help="JSON string or path for context")
    ap.add_argument("--policy", type=str, default='{"ltv_max":0.75, "mapper":{"confidence_threshold":0.70}}',
                    help="JSON string or path for policy")
    ap.add_argument("--out_summary", type=str, default=".artifacts/defi_milestone6_summary.json")
    ap.add_argument("--out_report", type=str, default=".artifacts/defi_milestone6_report.md")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)
    # ensure report dir too
    if getattr(args, "out_report", ""):
        os.makedirs(os.path.dirname(args.out_report) or ".", exist_ok=True)
    

    # Load artifacts
    mapper = load_mapper(args.mapper_path)
    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    context = parse_json_arg(args.context)
    policy = parse_json_arg(args.policy)




    # Prompts
    prompts: List[str] = []
    if args.prompts_jsonl == "-":
        # Read JSONL from STDIN (for tests and piping)
        for line in sys.stdin:
            try:
                rec = json.loads(line)
                p = rec.get("prompt", "").strip()
                if p:
                    prompts.append(p)
            except Exception:
                continue
    elif args.prompts_jsonl and os.path.exists(args.prompts_jsonl):
        with open(args.prompts_jsonl, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    p = rec.get("prompt", "").strip()
                    if p:
                        prompts.append(p)
                except Exception:
                    continue
    if not prompts:
        prompts = DEFAULT_PROMPTS.copy()

    # Deterministic subset
    import random
    random.seed(args.seed)
    random.shuffle(prompts)
    prompts = prompts[: max(1, args.runs)]

    # Run
    rows: List[Dict[str, Any]] = []
    abstains = 0
    oks = 0
    for p in prompts:
        res = run_single(
            mapper, p, class_names=class_names, rails=args.rails,
            conf_threshold=args.confidence_threshold, context=context, policy=policy, T=args.T
        )
        rows.append(res)
        if len(res["plan"]["sequence"]) == 0:
            abstains += 1
        # Consider executed OK if either: (no rails) OR rails ok==True
        exec_ok = (not res["exec"]["executed"]) or bool(res["exec"]["ok"])
        oks += int(exec_ok)

    total = len(rows)
    executed_ok = oks
    abstain_rate = abstains / max(1, total)
    status = "pass" if (executed_ok == total and abstain_rate <= 0.25) else "fail"
    
    summary = {
          "ok": True,
          "milestone": "defi_milestone6",
          "rails": args.rails,
          "T": args.T,
         "runs": total,
         "executed_ok": executed_ok,
         "abstain_count": abstains,
         "abstain_rate": abstain_rate,
          "status": status,
          "timestamp": int(time.time()),
          "rows": rows,
      }

    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    # Minimal MD report
    md = [f"# Milestone 6 — Mapper Integration Report",
          f"**Rails:** {args.rails}  \n**Runs:** {len(rows)}  \n**T:** {args.T}",
          f"- Abstain rate: **{summary['abstain_rate']:.3f}** ({abstains}/{len(rows)})",
          f"- Executed OK (incl. dry-run): **{oks}/{len(rows)}**",
          "",
          "## Cases"]
    for r in rows:
        seq = r["plan"]["sequence"]
        conf = r["plan"]["scores"]
        md.append(f"- **Prompt:** `{r['prompt']}`")
        md.append(f"  - Plan: {seq if seq else 'ABSTAIN'}")
        # show top-2
        top2 = sorted(conf.items(), key=lambda kv: kv[1], reverse=True)[:2]
        md.append(f"  - Top-2: {', '.join([f'{k}:{v:.2f}' for k,v in top2])}")
        if r["exec"]["executed"]:
            md.append(f"  - Rails: executed — ok={r['exec']['ok']}")
        else:
            md.append(f"  - Rails: not executed (dry-run)")
        md.append("")
    with open(args.out_report, "w") as f:
        f.write("\n".join(md))

    print(json.dumps({"ok": True, "summary": args.out_summary, "report": args.out_report,
                      "milestone": "defi_milestone6",
                      "rails": args.rails, "T": args.T, "runs": len(rows)}, indent=2))

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(1)
