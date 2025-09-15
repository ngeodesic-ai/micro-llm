#!/usr/bin/env python3
# bench_driver.py — distribution benchmark over JSONL suites
import argparse, json, sys, time, pathlib, statistics as stats
from collections import Counter, defaultdict
from typing import Dict, Any

from micro_lm.pipelines.runner import run_micro  # same API as milestones use  ✅

"""
python3 benchmarks/defi/bench_driver.py \
  --suite benchmarks/suites/defi_dist_v1.jsonl \
  --rails stage11 \
  --runs 3 \
  --context '{"oracle":{"age_sec":5,"max_age_sec":30}}' \
  --policy  '{"ltv_max":0.75,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}'
cp .artifacts/defi_bench_dist.json defi_bench_dist.json

python3 benchmarks/defi/bench_driver.py \
  --suite benchmarks/suites/defi_dist_v1.jsonl \
  --rails stage11 --runs 5 \
  --context '{"oracle":{"age_sec":5,"max_age_sec":30}}' \
  --policy  '{"ltv_max":0.75,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}' \
  --out .artifacts/defi_dist_v1_r5.json

for s in defi_edges_{ltv,hf,oracle}.jsonl; do
  python3 benchmarks/defi/bench_driver.py --suite benchmarks/suites/$s \
    --rails stage11 --runs 3 \
    --context '{"oracle":{"age_sec":5,"max_age_sec":30}}' \
    --policy  '{"ltv_max":0.75,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}' \
    --out .artifacts/${s%.jsonl}_r3.json
done


for thr in 0.6 0.7 0.8; do
  for m in 0.85 0.90 0.95; do
    python3 benchmarks/defi/bench_driver.py \
      --suite benchmarks/suites/defi_dist_v2.jsonl \
      --rails stage11 --runs 3 \
      --context '{"oracle":{"age_sec":5,"max_age_sec":30}}' \
      --policy "{\"ltv_max\":0.75, \"near_margin\":$m, \"mapper\":{\"model_path\":\".artifacts/defi_mapper.joblib\",\"confidence_threshold\":$thr}}" \
      --out .artifacts/dist_v2_thr${thr}_m${m}.json
  done
done


"""

ARTIF = pathlib.Path(".artifacts")
ARTIF.mkdir(parents=True, exist_ok=True)

def load_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            yield json.loads(line)

def one_run(item: Dict[str, Any], rails: str, T: int, base_context: Dict[str, Any], base_policy: Dict[str, Any]):
    prompt = item["prompt"]
    context = {**(base_context or {}), **(item.get("context") or {})}
    policy  = {**(base_policy  or {}), **(item.get("policy")  or {})}
    res = run_micro(domain="defi", prompt=prompt, context=context, policy=policy, rails=rails, T=T)

    seq = (res.get("plan") or {}).get("sequence") or []
    top1 = seq[0] if seq else None
    verify = res.get("verify") or {}
    flags  = res.get("flags")  or {}
    return {
        "prompt": prompt,
        "kind": item.get("kind"),                 # 'exec' | 'nonexec' | 'blocked'
        "label": item.get("label"),               # optional expected primitive for exec
        "top1": top1,
        "verify_ok": bool(verify.get("ok")),
        "reason": (verify.get("reason") or "").lower(),
        "flags": flags,
        "aux": {
            "prior": (res.get("aux") or {}).get("prior"),
            "mapper_confidence": (res.get("aux") or {}).get("mapper_confidence"),
        }
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="JSONL file of items")
    ap.add_argument("--rails", default="stage11")
    ap.add_argument("--T", type=int, default=180)
    ap.add_argument("--runs", type=int, default=1, help="repetitions per item (default 1)")
    ap.add_argument("--context", default=None, help='base context JSON or file')
    ap.add_argument("--policy",  default=None, help='base policy JSON or file')
    ap.add_argument("--out", default=None,
    help="Optional path for JSON summary output (default: .artifacts/defi_bench_dist.json)")

    args = ap.parse_args()

    def _load(x):
        if not x: return {}
        x = x.strip()
        if x.startswith("{"): return json.loads(x)
        return json.loads(pathlib.Path(x).read_text())

    base_context = _load(args.context)
    base_policy  = _load(args.policy)

    started = time.time()
    rows = []
    for item in load_jsonl(args.suite):
        # Repeat if you want stability checks on noisy items
        outs = [one_run(item, args.rails, args.T, base_context, base_policy) for _ in range(args.runs)]
        # Record first + stability signals
        top1_list = [o["top1"] for o in outs]
        stable = len(set(top1_list)) == 1
        row = outs[0]
        # infer kind when missing
        if not row.get("kind"):
            if row["reason"] == "abstain_non_exec":
                row["kind"] = "nonexec"
            elif not row["verify_ok"]:
                row["kind"] = "blocked"
            else:
                row["kind"] = "exec"
        row["top1_list"] = top1_list
        row["stable_top1"] = stable
        rows.append(row)

    # ----- Metrics -----
    buckets = defaultdict(list)
    for r in rows:
        buckets[r.get("kind","")].append(r)

    def rate(x): 
        return round(100.0 * x, 2)

    # Exec metrics
    exec_rows = buckets["exec"]
    exec_acc = sum(1 for r in exec_rows if r["top1"] == r.get("label")) / max(1, len(exec_rows))
    exec_verify_pass = sum(1 for r in exec_rows if r["verify_ok"]) / max(1, len(exec_rows))

    # Non-exec metrics
    ne_rows = buckets["nonexec"]
    abstain = sum(1 for r in ne_rows if r["top1"] is None and (r["flags"].get("abstain_non_exec") or not r["verify_ok"]))
    ne_abstain_rate = abstain / max(1, len(ne_rows))
    ne_halluc = sum(1 for r in ne_rows if r["top1"] is not None) / max(1, len(ne_rows))

    # Blocked metrics (policy/risk)
    bl_rows = buckets["blocked"]
    blocked_ok = sum(1 for r in bl_rows if not r["verify_ok"]) / max(1, len(bl_rows))
    blocked_reason_hf  = sum(1 for r in bl_rows if "hf"  in r["reason"]) / max(1, len(bl_rows))
    blocked_reason_ltv = sum(1 for r in bl_rows if "ltv" in r["reason"]) / max(1, len(bl_rows))

    # Stability (optional)
    stab_pool = [r for r in rows if r["top1"] is not None]
    stab_rate = sum(1 for r in stab_pool if r["stable_top1"]) / max(1, len(stab_pool))

    summary = {
        "suite": args.suite,
        "rails": args.rails,
        "T": args.T,
        "runs": args.runs,
        "counts": {k: len(v) for k,v in buckets.items()},
        "metrics": {
            "exec_top1_acc": rate(exec_acc),
            "exec_verify_pass_rate": rate(exec_verify_pass),
            "nonexec_abstain_rate": rate(ne_abstain_rate),
            "nonexec_hallucination_rate": rate(ne_halluc),
            "blocked_verify_block_rate": rate(blocked_ok),
            "blocked_reason_hf_presence": rate(blocked_reason_hf),
            "blocked_reason_ltv_presence": rate(blocked_reason_ltv),
            "stability_rate_on_exec": rate(stab_rate),
        }
    }

    # Write artifacts
    out_json = pathlib.Path(args.out) if args.out else ARTIF / "defi_bench_dist.json"
    out_csv  = out_json.with_suffix(".csv")
    out_md   = out_json.with_suffix(".md")

    out_json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    # CSV (wide but easy to grep)
    import csv
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "kind","prompt","label","top1","verify_ok","reason","stable_top1","top1_list"
        ])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    # MD report
    lines = []
    lines.append(f"# Distribution Benchmark\n")
    lines.append(f"- suite: `{args.suite}`  •  rails: `{args.rails}`  •  T={args.T}  •  runs={args.runs}")
    lines.append(f"- counts: {summary['counts']}")
    for k,v in summary["metrics"].items():
        lines.append(f"- {k}: **{v}%**")
    lines.append("\n## Buckets\n")
    for kind, grp in buckets.items():
        lines.append(f"### {kind} ({len(grp)})")
        for r in grp[:10]:  # sample first 10
            lines.append(f"- `{r['prompt']}` → top1=`{r['top1']}` verify_ok={r['verify_ok']} reason=`{r['reason']}`")
        lines.append("")
    out_md.write_text("\n".join(lines))

    print(json.dumps({
        "ok": True, 
        "json": str(out_json), 
        "csv": str(out_csv),
        "report": str(out_md),
        "elapsed_sec": round(time.time() - started, 2)
    }))
    return 0

if __name__ == "__main__":
    sys.exit(main())
