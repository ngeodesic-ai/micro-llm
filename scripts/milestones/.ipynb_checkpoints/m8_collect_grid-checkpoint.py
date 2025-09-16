#!/usr/bin/env python3
import json
import glob
import pandas as pd
import pathlib
import re


"""
# example grid (tweak as you like)
for thr in 0.60 0.70 0.80; do
  for m in 0.85 0.90 0.95; do
    python3 benchmarks/defi/bench_driver.py \
      --suite benchmarks/suites/defi_dist_v2.jsonl \
      --rails stage11 \
      --runs 3 \
      --context '{"oracle":{"age_sec":5,"max_age_sec":30}}' \
      --policy "{\"ltv_max\":0.75, \"near_margin\":$m, \"mapper\":{\"model_path\":\".artifacts/defi_mapper_embed.joblib\",\"confidence_threshold\":$thr}}" \
      --out .artifacts/m8/dist_v2_thr${thr}_m${m}.json
  done
done
"""

def pick(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default

def main():
    rows = []
    for fp in glob.glob(".artifacts/dist_v2_thr*_m*.json"):
        stem = pathlib.Path(fp).stem
        match = re.match(r"dist_v2_thr([0-9.]+)_m([0-9.]+)", stem)
        if not match:
            print(f"[m8_collect_grid] Skipping {fp}: filename does not match pattern")
            continue
        thr_str, nm_str = match.groups()
        thr = float(thr_str)
        nm = float(nm_str)

        try:
            with open(fp) as f:
                J = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[m8_collect_grid] Failed to parse JSON in {fp}: {e}")
            continue

        # if file is a list of per-prompt records, skip (no aggregate metrics here)
        if isinstance(J, list):
            print(f"[m8_collect_grid] Skipping {fp}: contains list data")
            continue

        # unwrap if someone put everything under "summary"
        root = J.get("summary", J) if isinstance(J, dict) else None
        if not isinstance(root, dict):
            print(f"[m8_collect_grid] No valid root in {fp}")
            continue

        metrics = pick(root, "metrics", default={}) or {}
        counts = pick(root, "counts", default={}) or {}

        overall = pick(metrics, "exec_top1_acc", "overall_acc", "overall_accuracy")
        abstain = pick(metrics, "nonexec_abstain_rate", "abstain_rate", "abstain")
        halluc_rate = pick(metrics, "nonexec_hallucination_rate")
        num_nonexec = counts.get("nonexec", 0) if isinstance(counts, dict) else 0

        # compute guard_escapes
        guard_escapes = int((halluc_rate or 0) * num_nonexec / 100) if isinstance(halluc_rate, (int, float)) else 0

        # compute coverage as guarded nonexec rate (1 - hallucination rate)
        coverage = 1 - (halluc_rate or 0) / 100 if isinstance(halluc_rate, (int, float)) else None

        # coerce numerics if present as strings
        def f(x):
            try: return float(x)
            except: return None

        row = {
            "thr": thr,
            "near_margin": nm,
            "overall_acc": f(overall),
            "abstain_rate": f(abstain),
            "coverage": f(coverage),
            "guard_escapes": guard_escapes,
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values([c for c in ["thr","near_margin"] if c in df.columns])
    pathlib.Path(".artifacts").mkdir(exist_ok=True)
    out = ".artifacts/m8_grid.csv"
    df.to_csv(out, index=False)
    print(f"[m8_collect_grid] wrote {out} with {len(df)} rows")

if __name__ == "__main__":
    main()