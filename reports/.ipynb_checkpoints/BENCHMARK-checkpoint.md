# DeFi Benches — Mapper vs Auditor

This folder contains standalone benches for the **mapper** (prompt → primitive) and the **auditor** (policy/verifier).  
They let you validate each component in isolation, without running the full `run_micro` pipeline.

---

## 1. Mapper Bench

**Purpose:** Evaluate the trained `.joblib` mapper on a labeled prompt set.  
Reports accuracy, abstain rate, hallucinations, omissions, and per-class metrics.

**Usage:**

```bash
# Run mapper-only benchmark
python3 src/micro_lm/domains/defi/benches/mapper_bench.py \
  --backend sbert \
  --model_path .artifacts/defi_mapper.joblib \
  --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl \
  --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv \
  --thresholds 0.5,0.55,0.6,0.65,0.7 \
  --min_overall_acc 0.75 \
  --out_dir .artifacts/defi/mapper_bench
```

**Inspect results:**

```bash
column -s, -t < .artifacts/defi/mapper_bench/metrics.csv
jq '.chosen' .artifacts/defi/mapper_bench/summary.json
```

---

## 2. Auditor Bench

**Purpose:** Run the **verifiers** (HF, LTV, oracle staleness) on mapper outputs.  
Checks that unsafe actions are correctly rejected or abstained.

**Usage:**

```bash
PYTHONWARNINGS="ignore::FutureWarning" \
python3 src/micro_lm/domains/defi/benches/audit_bench_metrics.py \
  --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl \
  --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv \
  --sbert sentence-transformers/all-MiniLM-L6-v2 \
  --n_max 4 --tau_span 0.50 --tau_rel 0.60 --tau_abs 0.93 \
  --out_dir .artifacts/defi/audit_bench \
  --competitive_eval
```

**Inspect results:**

```bash
jq '.metrics' .artifacts/defi/audit_bench/metrics_audit.json
head .artifacts/defi/audit_bench/rows_audit.csv
```

---

## 3. When to Use Each

- Use **Mapper Bench** during **M8** (scale-up to 5k prompts) to tune thresholds and confirm accuracy vs abstain.  
- Use **Auditor Bench** during **M9** to validate that policy guards (HF, LTV, oracle) block unsafe scenarios.  
- From **M10** onward, both components are integrated under Stage-11 rails (see `defi_milestone10.py`).

---

✅ This separation makes debugging easier:  
- If accuracy is low → check Mapper.  
- If unsafe paths are leaking through → check Auditor.  
