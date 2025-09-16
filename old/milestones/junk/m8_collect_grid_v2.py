#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 8 — Grid Collector (v2)

Scans for JSON outputs produced by the grid driver, extracts metrics and policy knobs,
and writes:
  - a consolidated CSV
  - a Markdown report with basic summaries
  - optional plots (accuracy vs abstain; heatmaps if grid is rectangular)

Usage:
    python3 milestones/m8_collect_grid_v2.py \
        --glob ".artifacts/dist_v2_thr*_m*.json" \
        --out_csv .artifacts/m8_grid.csv \
        --out_md  .artifacts/REPORT_M8.md \
        --out_png .artifacts/m8_accuracy_vs_abstain.png

for thr in 0.60 0.70 0.80; do
  for m in 0.85 0.90 0.95; do
    python3 milestones/m8_collect_grid_v2.py \
        --glob ".artifacts/m8/dist_v2_thr${thr}_m${m}.json"  \
        --out_csv .artifacts/m8/m8_grid_thr${thr}_m${m}.csv \
        --out_md  .artifacts/m8/REPORT_M8.md \
        --out_png .artifacts/m8/m8_accuracy_vs_abstain_thr${thr}_m${m}.png
  done
done
        
"""
import argparse, json, os, re, glob, math
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _find(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d: return d[k]
    return default

def _deep_get(d: Any, path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict): return default
        cur = cur.get(p, None)
        if cur is None: return default
    return cur

def parse_one(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        J = json.load(f)

    rec: Dict[str, Any] = {"src": os.path.basename(path)}

    # Try common locations for metrics
    M = _find(J, ["metrics","summary","geodesic","Geodesic","eval","results"], default=None)
    if isinstance(M, dict):
        # Promote common scalar metrics if present
        for k in ["overall_acc","accuracy","accuracy_exact","acc","f1","precision","recall",
                  "abstain_rate","abstain","omission_rate","hallucination_rate",
                  "exec_verify_pass_rate","nonexec_abstain_rate","nonexec_hallucination_rate"]:
            v = _find(M, [k], default=None)
            if v is not None and isinstance(v, (int,float)):
                rec[k] = float(v)
        # If accuracy values live under geodesic key but named differently
        for k_alias, std in [
            ("accuracy_exact", "accuracy"),
            ("omission_rate","abstain_rate"),
        ]:
            if (std not in rec) and (k_alias in M) and isinstance(M[k_alias], (int,float)):
                rec[std] = float(M[k_alias])

    # Sometimes the file itself is just a flat dict of metrics
    for k in ["overall_acc","accuracy","accuracy_exact","acc","f1","precision","recall",
              "abstain_rate","abstain","omission_rate","hallucination_rate",
              "exec_verify_pass_rate","nonexec_abstain_rate","nonexec_hallucination_rate"]:
        if k not in rec and k in J and isinstance(J[k], (int,float)):
            rec[k] = float(J[k])

    # Policy knobs
    thr = _deep_get(J, ["policy","mapper","confidence_threshold"], None)
    nm  = _deep_get(J, ["policy","near_margin"], None)
    ltv = _deep_get(J, ["policy","ltv_max"], None)
    if thr is None:
        # try to regex from filename: dist_v2_thr0.70_m0.95.json
        m = re.search(r"thr([0-9.]+)", os.path.basename(path))
        if m: thr = float(m.group(1))
    if nm is None:
        m = re.search(r"_m([0-9.]+)", os.path.basename(path))
        if m: nm = float(m.group(1))
    if thr is not None: rec["thr"] = float(thr)
    if nm  is not None: rec["near_margin"] = float(nm)
    if ltv is not None: rec["ltv_max"] = float(ltv)

    # Derive unified fields
    if "accuracy" not in rec:
        for a in ["overall_acc","accuracy_exact","acc"]:
            if a in rec: rec["accuracy"] = float(rec[a]); break

    if "abstain_rate" not in rec and "abstain" in rec:
        rec["abstain_rate"] = float(rec["abstain"])
    if "abstain" not in rec and "abstain_rate" in rec:
        rec["abstain"] = float(rec["abstain_rate"])

    return rec

def pick_columns(df: pd.DataFrame) -> List[str]:
    base = ["thr","near_margin","accuracy","f1","precision","recall",
            "abstain","hallucination_rate","exec_verify_pass_rate",
            "nonexec_abstain_rate","nonexec_hallucination_rate","src"]
    return [c for c in base if c in df.columns]

def choose_operating_point(df: pd.DataFrame, abstain_cap: float = 0.10) -> Tuple[int, Dict[str,float], str]:
    # Filter candidates
    cand = df.copy()
    if "abstain" in cand:
        cand = cand[cand["abstain"] <= abstain_cap]
        reason = f"Max accuracy with abstain ≤ {abstain_cap:.2f}"
    else:
        reason = "Max accuracy (no abstain column present)"
    if cand.empty:
        cand = df.copy()
        reason = "Max accuracy (no rows under abstain constraint)"

    # Sort by: accuracy desc, (optional) lower abstain, higher near_margin, lower thr
    sort_cols, ascending = [], []
    if "accuracy" in cand: sort_cols.append("accuracy"); ascending.append(False)
    if "abstain"  in cand: sort_cols.append("abstain");  ascending.append(True)
    if "near_margin" in cand: sort_cols.append("near_margin"); ascending.append(False)
    if "thr" in cand: sort_cols.append("thr"); ascending.append(True)

    if sort_cols:
        cand = cand.sort_values(by=sort_cols, ascending=ascending)
    idx = int(cand.index[0])
    return idx, df.loc[idx].to_dict(), reason

def plot_accuracy_vs_abstain(df: pd.DataFrame, out_png: str):
    if "accuracy" not in df: return
    x = df["abstain"] if "abstain" in df else np.zeros(len(df))
    y = df["accuracy"]
    plt.figure(figsize=(7,5))
    plt.scatter(x, y, s=60)
    # annotate with (thr, m)
    labels = []
    for _, r in df.iterrows():
        parts = []
        if "thr" in r and not pd.isna(r["thr"]): parts.append(f"thr={r['thr']:.2f}")
        if "near_margin" in r and not pd.isna(r["near_margin"]): parts.append(f"m={r['near_margin']:.2f}")
        labels.append(", ".join(parts) if parts else os.path.basename(str(r.get("src",""))))
    for xi, yi, txt in zip(x, y, labels):
        plt.annotate(txt, (xi, yi), fontsize=8, xytext=(4,4), textcoords="offset points")
    plt.xlabel("Abstain rate" if "abstain" in df else "Abstain (n/a)"); plt.ylabel("Accuracy")
    plt.title("Milestone 8: Accuracy vs Abstain (thr,m annotated)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def maybe_heatmap(df: pd.DataFrame, out_png: str, metric: str = "accuracy"):
    # Build pivot if we have a rectangular grid
    if not {"thr","near_margin",metric}.issubset(df.columns): return
    piv = df.pivot_table(index="thr", columns="near_margin", values=metric, aggfunc="mean")
    if piv.shape[0] < 2 or piv.shape[1] < 2: return
    plt.figure(figsize=(6,5))
    im = plt.imshow(piv.values, aspect="auto", origin="lower")
    plt.xticks(range(piv.shape[1]), [f"{c:.2f}" for c in piv.columns], rotation=45, ha="right")
    plt.yticks(range(piv.shape[0]), [f"{r:.2f}" for r in piv.index])
    plt.xlabel("near_margin"); plt.ylabel("thr"); plt.title(f"Grid heatmap: {metric}")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout(); base, ext = os.path.splitext(out_png)
    plt.savefig(f"{base}_heatmap_{metric}.png", dpi=160); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, required=True, help="Glob pattern for grid JSONs")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_md", type=str, required=True)
    ap.add_argument("--out_png", type=str, required=True)
    ap.add_argument("--abstain_cap", type=float, default=0.10)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"No JSON files matched: {args.glob}")

    rows: List[Dict[str,Any]] = []
    for p in paths:
        try:
            rows.append(parse_one(p))
        except Exception as e:
            rows.append({"src": os.path.basename(p), "parse_error": str(e)})

    df = pd.DataFrame(rows)
    # Derived fallbacks
    if "abstain" not in df and "omission_rate" in df:
        df["abstain"] = df["omission_rate"]
    # Keep useful cols
    cols = pick_columns(df)
    df = df[cols].copy()
    df.to_csv(args.out_csv, index=False)

    # Choose operating point
    try:
        idx, op, reason = choose_operating_point(df, abstain_cap=args.abstain_cap)
    except Exception:
        idx, op, reason = -1, {}, "No selection"

    # Plots
    try:
        plot_accuracy_vs_abstain(df, args.out_png)
        maybe_heatmap(df, args.out_png, metric="accuracy")
        if "abstain" in df.columns:
            maybe_heatmap(df, args.out_png, metric="abstain")
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")

    # Markdown report
    lines = [
        "# Milestone 8 — Grid Summary (v2)",
        "",
        f"- Files scanned: **{len(paths)}**",
        f"- Selection rule: **{reason}**",
        f"- Output CSV: `{args.out_csv}`",
        f"- Plots: `{args.out_png}` (+ heatmaps if grid rectangular)",
        ""
    ]
    if op:
        pretty = {k: v for k, v in op.items() if k in ("thr","near_margin","accuracy","abstain","f1","precision","recall","hallucination_rate","src")}
        lines.append("## Chosen Operating Point")
        lines.append("```")
        lines.append(str(pretty))
        lines.append("```")
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines))

    print(json.dumps({"ok": True, "out_csv": args.out_csv, "out_md": args.out_md, "out_png": args.out_png, "rows": len(df)}, indent=2))

if __name__ == "__main__":
    main()
