#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 8 — Collector for row-level CSVs (v1)
Parses per-prompt CSV outputs (one file per (thr,m)), computes metrics, and aggregates
into a summary CSV + plots.


for thr in 0.60 0.70 0.80; do
  for m in 0.85 0.90 0.95; do
    POLICY=$(echo "$POLICY_BASE" | sed "s/MARGIN/$m/g; s/THR/$thr/g")
    python3 benchmarks/defi/bench_driver.py \
      --suite "$SUITE" \
      --rails "$RAILS" \
      --runs $RUNS \
      --context "$CTX" \
      --policy "$POLICY" \
      --out_json ".artifacts/dist_v2_thr${thr}_m${m}.json" \
      --out_csv  ".artifacts/dist_v2_thr${thr}_m${m}.csv"
  done
done

for thr in 0.60 0.70 0.80; do
  for m in 0.85 0.90 0.95; do
    python3 milestones/m8_collect_grid_from_rows.py \
      --input .artifacts/dist_v2_thr${thr}_m${m}.json \
      --out_csv .artifacts/dist_v2_thr${thr}_m${m}.csv
  done
done

python3 milestones/m8_collect_grid_from_rows.py \
  --glob ".artifacts/dist_v2_thr*_m*.csv" \
  --out_csv .artifacts/m8_grid_from_rows.csv \
  --out_md  .artifacts/REPORT_M8_from_rows.md \
  --out_png .artifacts/m8_execacc_vs_abstain.png \
  --abstain_cap 0.10


"""
import argparse, os, re, glob, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


FNAME_RE = re.compile(r"thr(?P<thr>[0-9.]+)_m(?P<m>[0-9.]+)")

def _clean_num(s: str) -> str:
    s = str(s).strip()
    # strip trailing non-numeric artifacts
    while s and not (s[-1].isdigit() or s[-1] == "."):
        s = s[:-1]
    if s.endswith("."):
        s = s[:-1]
    return s

def parse_thr_m(name: str):
    m = FNAME_RE.search(name)
    if not m:
        return None, None
    thr_raw = _clean_num(m.group("thr"))
    m_raw   = _clean_num(m.group("m"))
    try:
        return float(thr_raw), float(m_raw)
    except Exception:
        return None, None
def to_bool(x):
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if x is None: return False
    s = str(x).strip().lower()
    if s in ("true","t","1","yes","y"): return True
    if s in ("false","f","0","no","n",""): return False
    # numeric fallback
    try:
        return float(s) != 0.0
    except:
        return False

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase columns
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    # Ensure required columns exist
    for need in ["prompt","kind","label","top1","verify_ok","stable_top1"]:
        if need not in df.columns:
            df[need] = np.nan
    # Normalize top1 empty/none
    def norm_top1(x):
        if x is None: return ""
        s = str(x).strip()
        if s.lower() in ("none","null","nan"): return ""
        return s
    df["top1"] = df["top1"].map(norm_top1)
    df["label"] = df["label"].fillna("")
    # Bools
    df["verify_ok"] = df["verify_ok"].map(to_bool)
    df["stable_top1"] = df["stable_top1"].map(to_bool)
    df["kind"] = df["kind"].astype(str).str.lower().str.strip()
    return df

def metrics_for(df: pd.DataFrame) -> dict:
    n = len(df)
    # Partition
    exec_df = df[df["kind"]=="exec"]
    nonexec_df = df[df["kind"]=="nonexec"]
    blocked_df = df[df["kind"]=="blocked"] if "blocked" in df["kind"].unique() else df[df["kind"]=="block"]
    # Core
    out = {}
    # Exec metrics
    if len(exec_df):
        if exec_df['label'].replace('', float('nan')).isna().all():
            # Fallback: use verify_ok as correctness proxy when labels are absent
            out['exec_accuracy_exact'] = float(exec_df['verify_ok'].mean())
        else:
            out['exec_accuracy_exact'] = float((exec_df['top1'] == exec_df['label']).mean())
        out["exec_verify_pass_rate"] = float(exec_df["verify_ok"].mean())
        out["exec_stability_rate"] = float(exec_df["stable_top1"].mean())
        out["exec_abstain_rate"] = float((exec_df["top1"] == "").mean())
    else:
        out["exec_accuracy_exact"] = np.nan
        out["exec_verify_pass_rate"] = np.nan
        out["exec_stability_rate"] = np.nan
        out["exec_abstain_rate"] = np.nan
    # Non-exec metrics
    if len(nonexec_df):
        out["nonexec_abstain_rate"] = float((nonexec_df["top1"] == "").mean())
        out["nonexec_hallucination_rate"] = float((nonexec_df["top1"] != "").mean())
    else:
        out["nonexec_abstain_rate"] = np.nan
        out["nonexec_hallucination_rate"] = np.nan
    # Blocked metrics (optional)
    if len(blocked_df):
        # Blocked means the guard should have prevented execution; success if verify_ok==False OR top1==""
        out["blocked_verify_block_rate"] = float(((~blocked_df["verify_ok"]) | (blocked_df["top1"]=="")).mean())
    else:
        out["blocked_verify_block_rate"] = np.nan
    # Overall proxy
    abstain_rate = float((df["top1"] == "").mean())
    out["abstain_rate"] = abstain_rate
    # "Overall accuracy" as: exec correct + nonexec abstains over all rows (if both exist)
    overall_hits = 0.0
    denom = float(n) if n else 1.0
    if len(exec_df):
        overall_hits += float((exec_df["top1"] == exec_df["label"]).sum())
    if len(nonexec_df):
        overall_hits += float((nonexec_df["top1"] == "").sum())
    out["overall_acc"] = overall_hits / denom
    return out

def choose_operating_point(df: pd.DataFrame, abstain_cap=0.10):
    cand = df.copy()
    if "abstain_rate" in cand:
        cand = cand[cand["abstain_rate"] <= abstain_cap]
        reason = f"Max exec_accuracy_exact with abstain ≤ {abstain_cap:.2f}"
    else:
        reason = "Max exec_accuracy_exact (no abstain_rate present)"
    if cand.empty:
        cand = df.copy()
        reason = "Max exec_accuracy_exact (no rows under abstain constraint)"
    sort_cols, ascending = ["exec_accuracy_exact","abstain_rate","near_margin","thr"], [False, True, False, True]
    keep = [c for c in sort_cols if c in cand.columns]
    asc  = [ascending[sort_cols.index(c)] for c in keep]
    cand = cand.sort_values(by=keep, ascending=asc)
    idx = int(cand.index[0])
    return idx, cand.loc[idx].to_dict(), reason

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, required=True, help="Glob of row-level CSVs, e.g., .artifacts/dist_v2_thr*.csv")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_md", type=str, required=True)
    ap.add_argument("--out_png", type=str, required=True)
    ap.add_argument("--abstain_cap", type=float, default=0.10)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"No CSVs matched: {args.glob}")

    rows = []
    for p in paths:
        df = pd.read_csv(p)
        df = normalize_cols(df)
        thr, m = parse_thr_m(os.path.basename(p))
        mets = metrics_for(df)
        rows.append({
            "src": os.path.basename(p),
            "thr": thr, "near_margin": m,
            **mets
        })
    summary = pd.DataFrame(rows)

    # Save CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    summary.to_csv(args.out_csv, index=False)

    # Plot Accuracy vs Abstain (exec accuracy)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,5))
        x = summary["abstain_rate"]
        y = summary["exec_accuracy_exact"]
        plt.scatter(x, y, s=60)
        for _, r in summary.iterrows():
            lbl = f"thr={r['thr']:.2f}, m={r['near_margin']:.2f}"
            plt.annotate(lbl, (r["abstain_rate"], r["exec_accuracy_exact"]), fontsize=8, xytext=(4,4), textcoords="offset points")
        plt.xlabel("Abstain rate")
        plt.ylabel("Exec accuracy")
        plt.title("M8: Exec accuracy vs Abstain (thr,m annotated)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=160)
        plt.close()
    except Exception as e:
        print("[WARN] plotting failed:", e)

    # Pick operating point
    try:
        idx, op, reason = choose_operating_point(summary, abstain_cap=args.abstain_cap)
    except Exception:
        idx, op, reason = -1, {}, "No selection"

    # Markdown report
    md = []
    md += ["# Milestone 8 — Grid Summary (from row-level CSVs)",
           "",
           f"- Files scanned: **{len(paths)}**",
           f"- Selection rule: **{reason}**",
           f"- Output CSV: `{args.out_csv}`",
           f"- Plot: `{args.out_png}`",
           ""]
    if op:
        pretty = {k: op.get(k) for k in ["thr","near_margin","exec_accuracy_exact","abstain_rate",
                                         "nonexec_abstain_rate","nonexec_hallucination_rate",
                                         "exec_verify_pass_rate","exec_stability_rate","blocked_verify_block_rate"] if k in op}
        md += ["## Chosen Operating Point", "```", json.dumps(pretty, indent=2), "```"]
    with open(args.out_md, "w") as f:
        f.write("\n".join(md))

    print(json.dumps({"ok": True, "rows": len(summary), "out_csv": args.out_csv, "out_md": args.out_md, "out_png": args.out_png}, indent=2))

if __name__ == "__main__":
    main()
