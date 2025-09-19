# Micro-LM SBERT — Benchmarks & Training Guide

This guide explains how to **train** the Tier-1 mapper and how to run the three **MVP benchmarks**:
- **Stage 8 (Mapper bench, ~5k prompts)** — validates the prompt→primitive mapper.

---

## 0) Prereqs & data layout

- Python 3.10+ and your project’s deps installed.
- Datasets (example layout):
  - `tests/fixtures/defi/defi_mapper_labeled_large.csv`  ← **training** (balanced ~1k)
  - `tests/fixtures/defi/defi_mapper_5k_prompts.jsonl`  ← **eval prompts**
  - `tests/fixtures/defi/defi_mapper_labeled_5k.csv`    ← **eval labels**
  - `tests/fixtures/defi/m8_per_class_thresholds.json`  ← (optional) per-class thresholds
- Artifacts will be written under `.artifacts/…`

> **No data leakage:** train on the small labeled CSV; evaluate only on the held-out 5k set.

---

## 1) Train SBERT on labeled CSV (Tier-1)

This produces the `.joblib` mapper artifact used by the SBERT backend.

```bash
PYTHONPATH=. python3 src/micro_lm/domains/defi/benches/train_mapper_embed.py   --labels_csv tests/fixtures/defi/defi_mapper_labeled_large.csv   --out_path .artifacts/defi_mapper.joblib   --sbert sentence-transformers/all-MiniLM-L6-v2   --C 8 --max_iter 2000 --calibrate
```

- Adjust the path if your trainer lives elsewhere (e.g., at repo root).
- The output path is referenced by the benches as `--model_path .artifacts/defi_mapper.joblib`.

---

## 2) Stage 8 — Mapper Bench (5k scale)

**Goal:** measure accuracy, coverage, and abstain on the held-out 5k set.

### SBERT @ scale
```bash
python3 src/micro_lm/domains/defi/benches/mapper_bench.py   --backend sbert   --model_path .artifacts/defi_mapper.joblib   --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl   --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv   --thresholds 0.5,0.55,0.6,0.65,0.7   --min_overall_acc 0.85   --out_dir .artifacts/defi/mapper_bench
```

### (Optional) Wordmap baseline
```bash
python3 src/micro_lm/domains/defi/benches/mapper_bench.py   --backend wordmap   --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl   --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv   --thresholds 0.2,0.25,0.3,0.35,0.4   --min_overall_acc 0.70   --out_dir .artifacts/defi/mapper_bench
```

**Outputs (all under `.artifacts/defi/mapper_bench/`):**
- `summary.json` — includes the **chosen** threshold
- `metrics.csv` — per-threshold stats (accuracy, coverage, abstain)
- `rows_thr_*.csv` — per-example predictions per threshold
- `report.md` — human-readable summary

> Tip: If your chosen threshold is stable (e.g., **0.5**), you can pin it in a config file:
> ```json
> .artifacts/defi/mapper_config.json
> { "backend":"sbert", "model_path":".artifacts/defi_mapper.joblib", "threshold":0.5 }
> ```

### (Optional) Per-class thresholds (classic M8 runner)
If you need per-class overrides, use your original milestone runner:
```bash
python3 defi_milestone8.py   --mapper_path .artifacts/defi_mapper.joblib   --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl   --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv   --thresholds "0.5,0.55,0.6,0.65,0.7"   --choose_by utility   --per_class_thresholds tests/fixtures/defi/m8_per_class_thresholds.json   --rows_csv .artifacts/m8_rows.csv   --out_summary .artifacts/m8_sum.json   --out_csv .artifacts/m8_metrics.csv
```

---

## TL;DR flow

1. **Train mapper (SBERT)** → `.artifacts/defi_mapper.joblib`  
2. **Run Stage 8 (5k)** → pin threshold (e.g., 0.5)  

