import os, sys, re, random, pandas as pd

"""
python3 milestones/m8_sample_nonexec.py .artifacts/m8_grid_from_rows.csv
"""

GRID = ".artifacts/m8_grid_from_rows.csv"
DEFAULT_ROWS = None         # optional override like ".artifacts/dist_v2_thr0.60_m0.95.csv"
SAMPLE_N = 20
SEED = 42

# 1) If an explicit CSV path is provided as an arg, use it; else pick OP from the grid
rows_csv = DEFAULT_ROWS
if len(sys.argv) > 1:
    rows_csv = sys.argv[1]

if not rows_csv:
    df = pd.read_csv(GRID)
    safe = df[df["nonexec_hallucination_rate"] == 0.0].copy()
    if safe.empty:
        print("No Tier-1 safe configs in grid (nonexec_hallucination_rate == 0). "
              "Pass a rows CSV explicitly: python3 this.py .artifacts/dist_v2_thrX_mY.csv")
        sys.exit(1)
    safe = safe.sort_values(
        by=["abstain_rate","exec_accuracy_exact","near_margin","thr"],
        ascending=[True, False, False, True]
    )
    rows_csv = safe.iloc[0]["src"]
    if not os.path.exists(rows_csv):
        # Try .artifacts prefix if the CSV is listed without it
        cand = os.path.join(".artifacts", os.path.basename(rows_csv))
        rows_csv = cand if os.path.exists(cand) else rows_csv

print(f"# Using rows CSV: {rows_csv}")
rows = pd.read_csv(rows_csv)

# Expect common columns from your row-level exports:
#   kind, prompt, top1, score, abstain (bool), label (optional), verify_ok (optional)
need = ["kind","prompt","top1","score","abstain"]
missing = [c for c in need if c not in rows.columns]
if missing:
    print(f"Missing expected columns in {rows_csv}: {missing}")
    print("Columns present:", list(rows.columns))
    sys.exit(2)

# 2) Filter to non-exec prompts (your rows typically have kind in {'exec','nonexec','blocked'})
nonexec = rows[rows["kind"].str.lower()=="nonexec"].copy()
if nonexec.empty:
    print("No non-exec rows found in this CSV. Are you pointing at the right run?")
    sys.exit(0)

# 3) Sample and pretty print
random.seed(SEED)
if len(nonexec) > SAMPLE_N:
    nonexec = nonexec.sample(SAMPLE_N, random_state=SEED)
nonexec = nonexec[["prompt","top1","score","abstain"]].reset_index(drop=True)

# Truncate long prompts for terminal readability
def trunc(s, n=80):
    s = str(s)
    return (s[:n-1] + "…") if len(s) > n else s

print("\n# Sample of non-exec prompts (random {}):".format(len(nonexec)))
print("{:<82} | {:<18} | {:>6} | {}".format("prompt", "mapper_top1", "score", "abstain"))
print("-"*82 + "-+-" + "-"*20 + "-+-" + "-"*6 + "-+-" + "-"*7)
for _, r in nonexec.iterrows():
    p = trunc(r["prompt"], 82)
    top1 = (r["top1"] if isinstance(r["top1"], str) else "")
    score = f"{float(r['score']):.3f}" if pd.notna(r["score"]) else ""
    abstain = "yes" if bool(r["abstain"]) else "no"
    print("{:<82} | {:<18} | {:>6} | {}".format(p, top1, score, abstain))