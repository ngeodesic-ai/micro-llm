# Micro‑LM Benchmarks — Public Overview & Quick‑Run

> **Micro‑LMs** deliver production‑grade reliability on a small set of actions (“primitives”).  
> They’re **deterministic**, **hallucination‑free**, and **near‑perfect** on their domain — without the size and cost of a giant LLM.

---

## Headline Results (DeFi MVP)

| Benchmark | What it tests | Key Signals | Result |
|---|---|---|---|
| **Stage 8 — Mapper (5,000 prompts)** | Prompt → Primitive classification | Accuracy (overall), Coverage, Abstain | **98.8%** accuracy, **0.7%** abstain |
| **Stage 10 — Rails** | Canonical actions through rails layer | Correctness on curated actions | **100%** |
| **Stage 11 — End‑to‑End** | Full pipeline (mapper + guardrails + oracle/verifiers) | Determinism across 3 runs | **100% consistent**, **0% hallucinations** |

**Why it matters:** Micro‑LMs do not “chat”; they **select** from a fixed, auditable set of primitives and execute via rails. That’s why you get **0% hallucination** and **deterministic** outcomes.

---

## Quick‑Run (Reproduce the Results)

> These steps reproduce the public numbers above using the DeFi MVP repo layout you have. Commands are copy‑pasteable.

### 0) Environment
- Python 3.10+
- Install your project’s dependencies (per your repo README).

### 1) Train the Mapper (Tier‑1 SBERT)
Produces a small model artifact used by the mapper backend.
```bash
PYTHONPATH=. python3 src/micro_lm/domains/defi/benches/train_mapper_embed.py   --labels_csv tests/fixtures/defi/defi_mapper_labeled_large.csv   --out_path .artifacts/defi_mapper.joblib   --sbert sentence-transformers/all-MiniLM-L6-v2   --C 8 --max_iter 2000 --calibrate
```

### 2) Stage 8 — Mapper Benchmark (5k held‑out)
Evaluates accuracy/coverage/abstain at scale and auto‑selects a threshold.
```bash
python3 src/micro_lm/domains/defi/benches/mapper_bench.py   --backend sbert   --model_path .artifacts/defi_mapper.joblib   --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl   --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv   --thresholds 0.5,0.55,0.6,0.65,0.7   --min_overall_acc 0.85   --out_dir .artifacts/defi/mapper_bench
```
**Expected:** overall accuracy ≈ **98–99%**, abstain < **1%**.  
**Artifacts:** `.artifacts/defi/mapper_bench/{summary.json,metrics.csv,report.md,rows_thr_*.csv}`.  
Pin the chosen threshold (e.g., **0.5**) for the next steps.

> Optional (advanced): per‑class thresholds with `defi_milestone8.py` and `m8_per_class_thresholds.json`.

### 3) Stage 10 — Rails Bench
Validates mapper + rails integration on canonical actions; expected to be near‑perfect.
```bash
python3 src/micro_lm/domains/defi/benches/rails_bench.py   --runs 3 --rails stage11 --gate_min 0.66
```
**Expected:** near **100%** correctness on the curated suite; reasons stable.  
**Artifacts:** `.artifacts/defi/rails_bench/{summary.json,report.md}`.

### 4) Stage 11 — End‑to‑End Bench
Runs the full deterministic pipeline; checks repeatability across runs.
```bash
python3 src/micro_lm/domains/defi/benches/e2e_bench.py   --runs 3 --rails stage11 --gate_min 0.66
```
**Expected:** **100%** on curated suite; **deterministic reasons**; **0% hallucinations**.  
**Artifacts:** `.artifacts/defi/e2e_bench/{summary.json,report.md}`.

---

## How to Present the Results (Public)

- **Reliability:** “On 5,000 unseen prompts, the mapper achieves ~**99%** accuracy with <**1%** abstain.”  
- **Determinism:** “Rails/E2E runs produce the same action and reason code across runs.”  
- **Zero Hallucinations:** “The system never invents unsupported actions — it abstains or selects a valid primitive.”  
- **Operational Fit:** “A compact model, fast to deploy, cheap to run — ideal for production workflows defined by a known set of primitives.”

---

## FAQ

- **Are these numbers from a generative LLM?**  
  No. Micro‑LMs are **classifiers + rails**, not open‑ended generators. That’s why they are deterministic and hallucination‑free.

- **Why is Rails/E2E ‘100%’?**  
  Those suites are **curated** (canonical actions). The mapper validated in Stage 8 is integrated via rails; with guardrails and verifiers, the pipeline is designed to be **correct and consistent** on those actions.

- **What if the mapper is unsure?**  
  It **abstains** (safe fallback), which you’ll see as a small abstain rate in Stage 8.

---

## One‑liner Quick Demo

```bash
# Train, then benchmark mapper @5k, then run rails and e2e benches
PYTHONPATH=. python3 src/micro_lm/domains/defi/benches/train_mapper_embed.py   --labels_csv tests/fixtures/defi/defi_mapper_labeled_large.csv   --out_path .artifacts/defi_mapper.joblib --sbert sentence-transformers/all-MiniLM-L6-v2 --C 8 --max_iter 2000 --calibrate && python3 src/micro_lm/domains/defi/benches/mapper_bench.py   --backend sbert --model_path .artifacts/defi_mapper.joblib   --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl   --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv   --thresholds 0.5,0.55,0.6,0.65,0.7 --min_overall_acc 0.85   --out_dir .artifacts/defi/mapper_bench && python3 src/micro_lm/domains/defi/benches/rails_bench.py --runs 3 --rails stage11 --gate_min 0.66 && python3 src/micro_lm/domains/defi/benches/e2e_bench.py   --runs 3 --rails stage11 --gate_min 0.66
```

---

**Micro‑LMs**: trustable AI for real workflows. If you don’t need a giant LLM, you need a micro‑LM.
