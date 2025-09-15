# Prompt -> Primitive Mapper: Scale Bench to 2–5k

## A) Retrain the mapper (same as M7)
### 0) Paths
```bash
export LAB_MINI="tests/fixtures/defi_mapper_labeled_mini.csv"   # or your curated mini CSV
export MAPPER_OUT=".artifacts/defi_mapper_embed.joblib"
mkdir -p .artifacts
```

### 1) Train a tiny, calibrated mapper (SBERT → LogisticRegression)
```bash
PYTHONPATH=. python3 milestones/train_mapper_embed.py \
  --labels_csv "$LAB_MINI" \
  --out_path  "$MAPPER_OUT" \
  --sbert sentence-transformers/all-mpnet-base-v2 \
  --C 8 --max_iter 2000 --calibrate
```
* It’s the same invocation shown in the training script’s header docstring, including the calibration switch and SBERT backbone .
* The script enforces prompt,label columns and does tiny-data-friendly calibration automatically (isotonic/sigmoid or skip when too few samples)

## B) Regenerate m7_rows.csv (threshold sweep)
```bash
# 2) Inputs for the sweep
export PROMPTS_JSONL="tests/fixtures/defi_mapper_stress_prompts.jsonl"
export LABELS_CSV="tests/fixtures/defi_mapper_stress_labeled.csv" # optional but recommended

# 3) Sweep thresholds, pick OP, and emit rows
python3 milestones/defi_milestone7.py \
  --mapper_path "$MAPPER_OUT" \
  --prompts_jsonl "$PROMPTS_JSONL" \
  --labels_csv    "$LABELS_CSV" \
  --thresholds "0.20,0.25,0.30,0.35,0.40" \
  --max_abstain_rate 0.20 \
  --min_overall_acc 0.85 \
  --choose_by utility \
  --rows_csv .artifacts/m7_rows.csv \
  --out_summary .artifacts/m7_summary.json \
  --out_csv     .artifacts/m7_metrics.csv
```
* defi_milestone7.py is an evaluation-only threshold sweeper; it loads the trained mapper, scores prompts, sweeps thresholds, writes a summary/CSV, and crucially dumps the per-prompt audit trail to --rows_csv at the chosen threshold .
* The script computes abstain, coverage, accuracy (if labels provided), then selects an operating point by your rule; all of these metrics are spelled out in the code and report section

Handy one-liners to glance at outputs (these exact examples are in the M7 header docstring):
```bash
column -s, -t < .artifacts/m7_metrics.csv | sed -n '1,12p'
awk -F, 'NR==1 || ($5=="False" && $2!=$3)' .artifacts/m7_rows.csv | sed -n '1,20p'
jq '.chosen' .artifacts/m7_summary.json
```

## C) Curate-then-scale recipe (documented steps to reuse)

Use this checklist whenever you add/rename primitives (e.g., moving from 4 → 7+ classes):

* **Define the class set (taxonomy)**
    * Lock names (e.g., deposit_asset, withdraw_asset, borrow_asset, repay_loan, swap_asset, add_collateral, remove_collateral).
    * Keep names stable across training/eval; pass via --class_names if you need to override defaults (defaults shown in M7)

* **Curate a tiny, balanced labeled CSV (the “mini” set)**
    * Columns: prompt,label.
    * 20–50 clean exemplars per class.    * 
    * Include synonym coverage (e.g., “top up / add” → deposit_asset), and hard negatives only if you intend a nonexec class—otherwise handle non-exec via the rails (front-gate).
    * Keep phrasing unambiguous; avoid overlapping intents in the mini set.

* **Train a simple calibrated mapper**
    * Run the command in Section A with --calibrate. The training script will pick an appropriate calibration method or skip if per-class counts are too low .

* **Prepare a small eval set mirroring your current taxonomy**
    * JSONL of prompts (optionally paired CSV labels).
    * Include a few “edge synonyms” that historically confused the mapper.

* **Run the M7 sweep**
    * Use defi_milestone7.py as in Section B.
    * If two classes remain confusable, add a small per-class thresholds JSON and pass --per_class_thresholds (the script supports this override) .

* **Inspect m7_rows.csv**
    * It lists prompt,gold_label,predicted,confidence,abstain,threshold for each example at the chosen threshold (easy failure drill-down) .
    * Fix obvious label or taxonomy issues; add 2–3 exemplars for frequent misses; repeat the sweep if necessary.

* **Freeze the OP + artifacts**
    * Save: mapper .joblib, chosen threshold(s), optional per-class overrides, plus the metrics and rows CSV.

* **Only then scale (Milestone 8)**
    * Move to the big suite + grid over confidence_threshold × near_margin as the detailed plan suggests (and keep the “0% hallucination” rails invariant)



