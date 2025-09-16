# Milestone 11 — Stage-11 Integration & Verification

**Status:** ✅ Passing (as of `.artifacts/defi_milestone11_summary.json`)

---

## Overview

Milestone 11 (M11) is the **culmination of Tier-1** in the DeFi micro-LLM roadmap.  
It integrates the full **Stage-11 NGF rails** (Warp → Detect → Denoise) with DeFi verifiers, ensuring deterministic reasoning and safe execution.

M11 builds directly on:

- **M9:** Edge case verifier & guards (LTV, HF, oracle, abstain).  
- **M10:** End-to-end consolidation of Tier-1 pipeline.  

The new contribution in M11 is the **baseline vs Stage-11 comparison** plus the **deterministic verification path**, locking in suppression of hallucinations and consistent abstain behavior.

---

## Key Features

- **Dual Rails:** Run both Stage-10 baseline and Stage-11 rails for apples-to-apples comparison.  
- **Verifier Integration:** Health-factor (HF) and loan-to-value (LTV) checks, oracle freshness enforcement.  
- **Abstain Handling:** Explicit abstain flags when confidence or oracle freshness is insufficient.  
- **Failure Reason Normalization:** Consistent mapping of failure reasons to expected categories (`ltv`, `hf`, `oracle_stale`, `abstain_non_exec`).  
- **Summary + Report:** JSON summary (`.artifacts/defi_milestone11_summary.json`) and human-readable report (`.artifacts/defi_milestone11_report.md`).  

---

## Usage

Example run:

```bash
python3 milestones/defi_milestone11.py \
  --rails stage11 --baseline_rails stage10 \
  --runs 5 --T 180 \
  --policy '{"ltv_max":0.75,"hf_min":1.0,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}' \
  --context '{"oracle":{"age_sec":5,"max_age_sec":30}}'
```

Inspect results:

```bash
python3 milestones/inspect_summary.py .artifacts/defi_milestone11_summary.json
```

---

## Expected Behavior

- **Deposit / Swap:** ✅ Verifies successfully under Stage-11 rails.  
- **Unsafe Withdraw (LTV breach):** ❌ Correctly rejected with `ltv` reason.  
- **Health Breach (HF < min):** ❌ Correctly rejected with `hf` reason.  
- **Oracle Staleness:** ❌ Correctly rejected with `oracle_stale`.  
- **Low-confidence or unknown primitive:** ❌ Abstains with `abstain_non_exec`.  

---

## Alignment with NGF Doctrine

- **Stage-10 (Baseline):** Matched filter + dual thresholds.  
- **Stage-11 (Integration):**  
  - **Warp:** PCA(3) funnel fit, single-well enforcement.  
  - **Detect:** Matched filtering with permuted null calibration.  
  - **Denoise:** EMA + median smoothing, phantom guard, jitter averaging.  

These modules embody the **Warp → Detect → Denoise** doctrine formalized in the patents and article.

---

## Deliverable

- **Script:** `milestones/defi_milestone11.py`  
- **Artifacts:**  
  - JSON summary: `.artifacts/defi_milestone11_summary.json`  
  - Markdown report: `.artifacts/defi_milestone11_report.md`  

This milestone locks in **Tier-1 MVP determinism** and provides the foundation for Tier-2 sidecar integration.
