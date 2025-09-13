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

---

## DeFi Micro-LLM: Tiered Plan of Attack

This project follows a three-tier strategy aligned with the NGF Stage-10/11 doctrine  
(**Warp → Detect → Denoise → Verify**). Each tier represents an increasing level of capability and integration.

---

### Tier 0 — Baseline Deterministic Rails (✔ Secured)
- Stage-10 rails with matched filter + dual thresholds.  
- Simple rule-based mapper produces residual traces.  
- Basic verifiers (ARC grid ops, DeFi invariants).  
- End-to-end CLI runs and benchmark harness are working.  

**Status:** ✅ Complete — foundation secured.

---

### Tier 1 — Micro-LLM on Synthetic Latents (Operational)
- Hybrid mapper + prior injection for prompt→feature mapping.  
- Stage-11 warp + detect + denoise rails applied to synthetic latent traces.  
- Benchmarked successfully on ARC-like synthetic latents (deterministic reasoning).  
- Micro-LLM exists: parser + denoiser operating on synthetic wells.  

**Status:** ✅ Proven — benchmarks confirm deterministic reasoning on synthetic latents.

---

### Tier 2 — Sidecar Integration with Real Latents (Aspirational)
- Integrate with an external LLM (e.g. GPT-2) to extract **pooled latents** from DeFi prompts.  
- Replace synthetic latent generator with live embeddings.  
- Run rails (warp → detect → denoise) on real latents to classify/sequence primitives.  
- Bragging-point tier — showcases novel architecture but currently high-risk and unproven.  

**Status:** 🚧 Dream stage — future work, not required for MVP.

---

**Roadmap Summary:**  
- Tiers 0 + 1 provide a safe, working MVP with deterministic rails and micro-LLM reasoning on synthetic latents.  
- Tier 2 remains a stretch goal: sidecar integration for real latents, to be explored later.

## ARC micro PoC — run a reasoning primitive
micro-arc --prompt "flip the grid horizontally" --rails stage11

### Example output:
{
  "domain": "arc",
  "rails": "stage11",
  "plan": {
    "sequence": ["flip_h"]
  },
  "verify": {"ok": true, "reason": ""}
}


## Install

```bash
# 1) Install the NGF core
python3 -m pip install -U ngeodesic

# 2) (optional) install this repo in editable mode
git clone https://github.com/ngeodesic-ai/micro-llm.git
cd micro-llm
python3 -m pip install -e .
```