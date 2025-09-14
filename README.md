# micro-llm

NGF-powered micro-LLMs: lightweight, domain-specific reasoning sidecars.  
This repo is a research testbed: first for **ARC** (visual reasoning), then for a **DeFi** PoC — both built on top of the `ngeodesic` Python package.

---

## Why this repo?

- **ARC micro-LLM (today):** a compact, NGF-style classifier that detects and orders latent “primitives” on synthetic ARC-like traces. It demonstrates the **Adapter → Detect** path and stable metrics.
- **DeFi micro-LLM (next):** same skeleton, different adapter — turn market features into latent traces and reuse the exact parser/denoiser stack.

> NGF’s repeatable pipeline: **Adapter → Warp → Detect → Denoise → Execute → Verify**. Here we focus on Adapter→Detect (+optional Denoise) for a small, reliable sidecar you can pair with a larger LLM.

---

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

#### Example output:
{
  "domain": "arc",
  "rails": "stage11",
  "plan": {
    "sequence": ["flip_h"]
  },
  "verify": {"ok": true, "reason": ""}
}

---

## DeFi Micro-LLM: Tiered Plan of Attack

This repo hosts experiments in **micro-scale language models** with **domain-specific reasoning**. Our current focus is the DeFi domain, but the architecture generalizes to other verticals. Each tier represents an increasing level of capability and integration. 

---

### **Tier-0: Baseline Deterministic Rails (✔ Secured)**  
- **Stock matched filter + parser** pipeline.  
- Supports core DeFi primitives with deterministic abstain paths.  
- Sandbox verified and benchmarked with stable execution.

**Status:** ✅ Complete — foundation secured.

### **Tier-1: Micro-LLM on Synthetic Latents (In Progress)**  
- Replace hashmap lookups with a **trained micro-LLM encoder**.  
- Train against **2–5k synthetic latent prompts**.  
- Benchmark with full Stage-11 runner on DeFi suites.

**Status:** 👷 In progress — LLM benchmarks confirm deterministic reasoning on synthetic latents (MVP)

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
- Tiers 0 + 1 provide a safe, working MVP with deterministic rails and micro-LLM reasoning on synthetic latents.
- Tier 2 expands the scope of what micro-LLM can do
- Tier 3 remains a the end goal: sidecar integration for real latents, to be explored later.


## Install

```bash
# 1) Install the NGF core
python3 -m pip install -U ngeodesic

# 2) (optional) install this repo in editable mode
git clone https://github.com/ngeodesic-ai/micro-llm.git
cd micro-llm
python3 -m pip install -e .
```