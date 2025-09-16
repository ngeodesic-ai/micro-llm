# Milestone 9 — Verifier & Guard Benchmarks

**Status:** ✅ Passing (as of `.artifacts/defi_milestone9_summary.json`)

---

## Overview

Milestone 9 (M9) introduces **verifiers and safety guards** into the DeFi micro-LLM pipeline.  
Where Milestone 8 focused on mapper benchmarks, M9 adds enforcement of **risk-aware execution rules**:

- Loan-to-Value (LTV) thresholds  
- Health Factor (HF) minimums  
- Oracle freshness  
- Abstain handling for low-confidence or unsupported prompts  

This milestone is critical for making Tier-1 safe and deterministic — it ensures the pipeline will **reject unsafe or ambiguous actions**, instead of hallucinating.

---

## Key Features

- **Edge Case Suite:** Tests unsafe withdraws, HF breaches, oracle staleness, and low-confidence prompts.  
- **Verifier Integration:** Calls `verify_with_mapper` to evaluate execution plans.  
- **Reason Tokens:** Expanded reason matching to allow flexible checks (e.g., `ltv` **or** `oracle/stale`).  
- **Consistency:** Runs multiple repeats per edge case to ensure stability of failure reasons.  
- **Artifacts:**  
  - JSON summary: `.artifacts/defi_milestone9_summary.json`  
  - Markdown report: `.artifacts/defi_milestone9_report.md`  

---

## Usage

Example run:

```bash
python3 milestones/defi_milestone9.py \
  --rails stage11 \
  --runs 3 --T 180 \
  --policy '{"ltv_max":0.75,"hf_min":1.0,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}' \
  --context '{"oracle":{"age_sec":5,"max_age_sec":30}}'
```

Inspect results:

```bash
python3 milestones/inspect_summary.py .artifacts/defi_milestone9_summary.json
```

---

## Expected Behavior

- **Deposit / Swap:** ✅ Allowed and verified.  
- **Unsafe Withdraw (LTV breach):** ❌ Rejected with reason containing `ltv`.  
- **HF Breach (borrow too high):** ❌ Rejected with reason containing `hf`.  
- **Oracle Staleness:** ❌ Rejected with reason containing `oracle` or `stale`.  
- **Low-confidence / unknown primitive:** ❌ Abstains with `low_conf` or `abstain_non_exec`.  

---

## Alignment with Tier-1 Roadmap

- **Milestone 7/8:** Mapper benchmarks established mapping quality.  
- **Milestone 9:** Adds verifiers + guards, hardening the system against unsafe or ambiguous actions.  
- **Milestone 10:** Builds on this by consolidating sandboxing and full pipeline integration.  
- **Milestone 11:** Uses these verifiers in the consolidated Stage-11 benchmark.

---

## Deliverable

- **Script:** `milestones/defi_milestone9.py`  
- **Artifacts:**  
  - JSON summary: `.artifacts/defi_milestone9_summary.json`  
  - Markdown report: `.artifacts/defi_milestone9_report.md`  

This milestone ensures the **safety guarantees** of the Tier-1 MVP: unsafe actions are deterministically rejected or abstained, with clear reason codes.
