# micro-lm

NGF-powered micro-LMs: lightweight, domain-specific reasoning sidecars.  
This repo is a research testbed: first for **ARC** (visual reasoning), then for a **DeFi** PoC — both built on top of the `ngeodesic` Python package.

---

## Why this repo?

- **ARC micro-LM (today):** a compact, NGF-style classifier that detects and orders latent “primitives” on synthetic ARC-like traces. It demonstrates the **Adapter → Detect** path and stable metrics.
- **DeFi micro-LM (next):** same skeleton, different adapter — turn market features into latent traces and reuse the exact parser/denoiser stack.

> NGF’s repeatable pipeline: **Adapter → Warp → Detect → Denoise → Execute → Verify**. Here we focus on Adapter→Detect (+optional Denoise) for a small, reliable sidecar you can pair with a larger LLM.

---

## Tier-1 LLMs vs. micro-lm (DeFi Execution)

| Dimension | Tier-1 LLMs (ChatGPT, Claude, Meta, Perplexity, etc.) | **micro-lm Tier-1 (DeFi)** |
|-----------|-------------------------------------------------------|-----------------------------|
| **Domain accuracy** | Broad coverage, but DeFi primitives are not a training focus. Accuracy drifts under phrasing changes. | Mapper trained on 1k–5k DeFi prompts. Benchmarked accuracy > 98% on swaps/deposits; abstains correctly when uncertain. |
| **Determinism** | Outputs vary run-to-run (sampling drift). Even `temperature=0` doesn’t guarantee identical results. | Stage-11 NGF rails (Warp → Detect → Denoise) yield reproducible traces. Perturbation tests confirm stable decisions. |
| **Safety / Policy enforcement** | Can be prompted with “stay under LTV 0.75,” but no hard guarantees — may still propose unsafe actions. | Built-in verifiers: Loan-to-Value (LTV), Health Factor (HF), Oracle freshness. Unsafe paths always block or abstain. |
| **Abstain behavior** | Rarely abstains — tends to “make something up” even when uncertain. | Explicit abstain mode: non-exec prompts (balance checks, nonsense) → abstain with clear reason (`abstain_non_exec`). |
| **Auditability** | Opaque; no structured rationale. | Every run produces machine-readable artifacts: mapper score, abstain reason, verifier tags, plan trace. Auditable for compliance. |
| **Efficiency / Cost** | 10s–100s of billions of params; inference is slow/expensive. | SBERT (~22M params) + lightweight classifier. Fast, cheap, deployable in CI. |
| **Regulatory / Compliance fit** | Hard to certify (stochastic, unexplainable). | Deterministic + auditable by design. Built for domains where regulators demand safety. |

---

### **Summary**
- **Tier-1 LLMs = generalists**: broad knowledge, flexible language, but *stochastic and unsafe* for mission-critical DeFi execution.  
- **micro-lm Tier-1 = specialist**: slim, deterministic, auditable, and *more accurate where it matters* (DeFi primitives, policy enforcement, reproducibility).


## Foundation: `ngeodesic` (NGF Stage-10/11)

- **Stage-10 (Parser):** matched-filter parsing with dual thresholds (absolute vs null; relative vs best channel), then ordering by peak time.
- **Stage-11 (Denoise):** stabilization via hybrid EMA+median smoothing, confidence gates, seed-jitter averaging — the Warp→Detect→Denoise doctrine to suppress phantoms.

These are provided by the `ngeodesic` package and reused here without modification.

---

## What’s included

- **ARC sandbox**: synthetic adapters that produce per-primitive traces; evaluation scripts that compute accuracy, precision/recall/F1, and NGF-native rates (hallucination/omission).
- **DeFi stubs (WIP)**: adapters for market features → latent traces; same parser/denoiser stack; same metrics.

### ARC micro PoC — run a reasoning primitive
micro-arc --prompt "flip the grid horizontally" --rails stage11

---

## DeFi Micro-LM: Tiered Plan of Attack

This repo hosts experiments in **micro-scale language models** with **domain-specific reasoning**. Our current focus is the DeFi domain, but the architecture generalizes to other verticals. Each tier represents an increasing level of capability and integration. 

---

### **Tier-0: Baseline Deterministic Rails (✔ Secured)**  
- **Stock matched filter + parser** pipeline.  
- Supports core DeFi primitives with deterministic abstain paths.  
- Sandbox verified and benchmarked with stable execution.

**Status:** ✅ Complete — foundation secured.

### **Tier-1: Micro-LM on Synthetic Latents (In Progress)**  
- Replace hashmap lookups with a **trained micro-LM encoder**.  
- Train against **2–5k synthetic latent prompts**.  
- Benchmark with full Stage-11 runner on DeFi suites.

**Status:** ✅ Complete — MVP secured.

### **Tier-2: Incorporate WDD with Synthetic Latents (Operational)**  
- Add **Warp → Detect → Denoise (WDD)** pipeline.  
- Stress test signal separation + denoising with synthetic latents.

**Status:** 🚧 Proven — WDD LLM benchmarks confirm deterministic reasoning on synthetic latents.

### **Tier-3: Real Latents (End Goal)**  
- Swap synthetic latents for **true model latents**.  
- Validate WDD under real-world latent distributions.

**Status:** 🔮 Planning stage — future work, not required for MVP.

---

**Roadmap Summary:**  
- Tiers 0 + 1 provide a safe, working MVP with deterministic rails and micro-LM reasoning on synthetic latents.
- Tier 2 expands the scope of what micro-LM can do
- Tier 3 remains a the end goal: sidecar integration for real latents, to be explored later.

## Quickstart

``` python
from micro_lm.core.runner import run_micro

# Example prompt
prompt = "deposit 10 ETH into aave"

# Minimal policy & context
policy = {
    "mapper": {
        "model_path": ".artifacts/defi_mapper.joblib",
        "confidence_threshold": 0.5,
    }
}
context = {}

# Run through micro-lm pipeline
out = run_micro(
    domain="defi",
    prompt=prompt,
    context=context,
    policy=policy,
    rails="stage11",
    T=180,
    backend="wordmap",   # or "sbert"
)

print(out)
```

------------------------------------------------------------------------

### ✅ Example Output

``` python
{
  'ok': True,
  'label': 'deposit_asset',
  'score': 0.71,
  'reason': 'shim:accept:stage-4',
  'artifacts': {
    'mapper': {
      'score': 0.71,
      'reason': 'heuristic:deposit',
      'aux': {'reason': 'heuristic:deposit'}
    },
    'verify': {
      'ok': True,
      'reason': 'shim:accept:stage-4'
    },
    'schema': {
      'v': 1,
      'keys': ['mapper', 'verify']
    }
  }
}
```

------------------------------------------------------------------------

### 🔎 Output Breakdown

#### Top-level fields

-   **`ok: True`** → Overall run succeeded, action allowed.\
-   **`label: 'deposit_asset'`** → Canonical intent chosen.\
-   **`score: 0.71`** → Mapper's confidence.\
-   **`reason: 'shim:accept:stage-4'`** → Accepted by Stage-4 rails
    shim.

### Artifacts

-   **`mapper`**
    -   Raw mapper result.\
    -   Score + heuristic reason.
-   **`verify`**
    -   Rails/audit check result.\
    -   `ok=True` → passed safety/policy.
-   **`schema`**
    -   Metadata about which artifact keys exist.

------------------------------------------------------------------------

## 🧩 Interpretation

This tells us:\
1. Prompt looked like a **deposit**.\
2. Mapper classified with \~71% confidence.\
3. Audit/rails verifier confirmed no violations.\
4. Final decision → **allow**, with `deposit_asset` as the action.


## Install

```bash
# 1) Install the NGF core
python3 -m pip install -U ngeodesic

# 2) (optional) install this repo in editable mode
git clone https://github.com/ngeodesic-ai/micro-lm.git
cd micro-lm
python3 -m pip install -e .
```