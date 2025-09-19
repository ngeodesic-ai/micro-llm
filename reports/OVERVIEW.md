# micro-lm — Tier-1 Refactor Overview

## Purpose
**micro-lm** provides a lightweight, deterministic reasoning stack for domain-specific language models. Tier-1 establishes the *minimal viable package* by consolidating the mapper, verifier, and rails into a single harness. It is designed to be domain-agnostic with adapters for **DeFi** and **ARC**, and sets the foundation for Tier-2 (latent + WDD pipeline).

---

## Architecture

Tier-1 organizes into four core modules:

1. **Adapters**
   - `SimpleContextAdapter`: Normalizes raw JSON or dict-like context into a stable schema.  
   - Domain adapters (DeFi, ARC) sit here and provide context hydration.

2. **Mapper**
   - `JoblibMapper`: Wraps a SBERT or wordmap encoder (loaded from `.artifacts/defi_mapper.joblib`).  
   - Deterministic top-1 classification over the primitive set (`deposit_asset`, `swap_asset`, etc.).  
   - Configurable via `JoblibMapperConfig` (model path + confidence threshold).

3. **Verifier**
   - Policy-driven checks on the candidate action.  
   - DeFi: LTV ceiling, HF floor, oracle staleness.  
   - ARC: schema compliance, input/output trace sanity.  
   - Returns pass/abstain with reason.

4. **Planner**
   - `RulePlanner`: Lightweight template-based planner that translates `(intent, text, context)` into a structured plan.  
   - Deterministic expansion of actions (e.g. `swap_asset` → `{from, to, amount}`).

---

## Rails

Tier-1 runs on **Stage-11 NGF rails**:

```
Warp → Detect → Denoise
```

- Warp: canonical tokenization and normalization.  
- Detect: matched-filter hooks for critical spans.  
- Denoise: suppresses false activations, enforces abstain when uncertain.  

The rails produce deterministic traces with bounded depth (`T`).

---

## Harness

The **deterministic harness** glues everything together:

- **MapperShim**: fallback when the rails abstain, ensuring a safe top-1 when confidence ≥ threshold.  
- **Verifier**: applies domain-specific policy checks.  
- **Planner**: generates structured action plans.  
- **Debug Gate (`--debug`)**: surfaces all intermediate artifacts (mapper raw scores, verifier reasons, planner expansion).

Example:

```python
from micro_lm.core.runner import run_micro

out = run_micro(
    domain="defi",
    prompt="deposit 10 ETH into aave",
    context={"oracle": {"age_sec": 5}},
    policy={"ltv_max": 0.75, "hf_min": 1.0,
            "mapper": {"confidence_threshold": 0.7}},
    rails="stage11",
    T=180
)
```

Output:

```json
{
  "ok": true,
  "label": "deposit_asset",
  "score": 0.71,
  "reason": "shim:accept:stage-4",
  "artifacts": {
    "mapper": {"score": 0.71, "reason": "heuristic:deposit"},
    "verify": {"ok": true, "reason": "shim:accept:stage-4"},
    "schema": {"v": 1, "keys": ["mapper", "verify"]}
  }
}
```

---

## Domains

- **DeFi**: Primitives include `deposit_asset`, `swap_asset`, `withdraw_asset`, `borrow_asset`, `repay_asset`, `stake_asset`, `unstake_asset`, `claim_rewards`. Verifier enforces safety guards (LTV, HF, oracle).  
- **ARC**: Puzzle transformation primitives (e.g. flip, rotate, color-map). Verifier ensures plan matches schema and trace length.  

---

## Status

- ✅ Mapper refactored under `JoblibMapper` with deterministic shim.  
- ✅ Verifier consolidated into policy-driven checks.  
- ✅ RulePlanner integrated.  
- ✅ Stage-11 rails locked in.  
- ✅ Deterministic harness with `--debug` gate.  

**Tier-1 is complete and stable.**  
Next step: **Tier-2** (latents, PCA + WDD pipeline) builds on this solid base without revisiting the foundation.
