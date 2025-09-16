# Prompt -> Primitive Mapper: small scale (< 200) (Milestone 7)

### 1) Paths
```bash
export LAB_MINI="tests/fixtures/defi_mapper_labeled_mini.csv"  
export MAPPER_OUT=".artifacts/defi_mapper_embed.joblib"
mkdir -p .artifacts
```

### 2) Curate a tiny, clean training set

* CSV: prompt,label
    * 20–50 examples/class, balanced
    * Include synonyms you know are production-relevant (“top up”, “add” → deposit_asset)
    * Keep non-execs out (we handle those with front-gate/rails), unless you explicitly add a nonexec class later

### 3) Train a simple calibrated mapper
```bash
PYTHONPATH=. python3 milestones/train_mapper_embed.py \
  --labels_csv "$LAB_MINI" \
  --out_path  "$MAPPER_OUT" \
  --sbert sentence-transformers/all-mpnet-base-v2 \
  --C 8 --max_iter 2000 --calibrate
```
* It’s the same invocation shown in the training script’s header docstring, including the calibration switch and SBERT backbone .
* The script enforces prompt,label columns and does tiny-data-friendly calibration automatically (isotonic/sigmoid or skip when too few samples)

### 4) Threshold sweep (eval-only)
```bash
export PROMPTS_JSONL="tests/fixtures/defi_mapper_stress_prompts.jsonl"
export LABELS_CSV="tests/fixtures/defi_mapper_stress_labeled.csv" # optional but recommended

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
* Selection rule: minimize abstain under the cap, then maximize overall accuracy (and/or utility).
*Operating point (OP) for your run: thr=0.30 (0.925 overall, 0% abstain, full coverage).
* Artifacts to freeze: `mapper.joblib`, `m7_summary.json`, `m7_metrics.csv`, and `m7_rows.csv`.
* Handy one-liners to glance at outputs (these exact examples are in the M7 header docstring):
```bash
column -s, -t < .artifacts/m7_metrics.csv | sed -n '1,12p'
awk -F, 'NR==1 || ($5=="False" && $2!=$3)' .artifacts/m7_rows.csv | sed -n '1,20p'
jq '.chosen' .artifacts/m7_summary.json
```

### 5) Per-class tweaks (optional, only where needed)
If confusions cluster (e.g., borrow vs deposit), create:
```bash
{
  "borrow_asset": 0.62,
  "deposit_asset": 0.58
}
```
…and pass --per_class_thresholds into defi_milestone7.py. This keeps global coverage while shaving specific mistakes.

### 6) Template Makefile targets (so it’s one-command next time)

```bash
ARTIFACTS := .artifacts
MAPPER := $(ARTIFACTS)/defi_mapper_embed.joblib
LAB := tests/fixtures/defi_mapper_labeled_mini.csv
PROMPTS := tests/fixtures/defi_mapper_stress_prompts.jsonl
LABELS := tests/fixtures/defi_mapper_stress_labeled.csv

.PHONY: mapper-train
mapper-train:
\tPYTHONPATH=. python3 milestones/train_mapper_embed.py \
\t  --labels_csv $(LAB) --out_path $(MAPPER) \
\t  --sbert sentence-transformers/all-mpnet-base-v2 \
\t  --C 8 --max_iter 2000 --calibrate

.PHONY: m7-sweep
m7-sweep:
\tpython3 milestones/defi_milestone7.py \
\t  --mapper_path $(MAPPER) \
\t  --prompts_jsonl $(PROMPTS) \
\t  --labels_csv $(LABELS) \
\t  --thresholds "0.20,0.25,0.30,0.35,0.40" \
\t  --max_abstain_rate 0.20 \
\t  --min_overall_acc 0.85 \
\t  --choose_by utility \
\t  --rows_csv $(ARTIFACTS)/m7_rows.csv \
\t  --out_summary $(ARTIFACTS)/m7_summary.json \
\t  --out_csv $(ARTIFACTS)/m7_metrics.csv
```