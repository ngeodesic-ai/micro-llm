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
python3 src/micro_lm/domains/defi/benches/mapper_bench.py   --labels_csv tests/fixtures/defi/defi_mapper_labeled_5k.csv   --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl   --model_path .artifacts/defi_mapper.joblib   --thresholds "0.2,0.3,0.35,0.4,0.7"   --out_csv .artifacts/mapper_metrics.csv   --out_summary .artifacts/mapper_summary.json
```

**Inspect results:**

```bash
column -s, -t < .artifacts/mapper_metrics.csv
jq '.chosen' .artifacts/mapper_summary.json
```

---

## 2. Auditor Bench

**Purpose:** Run the **verifiers** (HF, LTV, oracle staleness) on mapper outputs.  
Checks that unsafe actions are correctly rejected or abstained.

**Usage:**

```bash
python3 src/micro_lm/domains/defi/benches/audit_bench_metrics.py   --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl   --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv   --sbert sentence-transformers/all-MiniLM-L6-v2   --n_max 4   --tau_span 0.50 --tau_rel 0.60 --tau_abs 0.70   --out_summary .artifacts/audit_summary.json   --out_csv .artifacts/audit_metrics.csv
```

**Inspect results:**

```bash
jq '.metrics' .artifacts/audit_summary.json
head .artifacts/audit_metrics.csv
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
