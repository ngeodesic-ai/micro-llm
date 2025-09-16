# Positioning Note: NGF Core vs. micro-lm Adapters

## Why Two Repos?

To keep the project scientifically rigorous **and** practically flexible, we are deliberately
splitting the work into two layers:

---

## 1. **ngeodesic v0.1.0** — Hardened Core Science

- **What it is:**  
  A minimal, stable Python package that implements the **Noetic Geodesic Framework (NGF)** —  
  Stage-10 (geodesic parser) and Stage-11 (Warp → Detect → Denoise).

- **Why:**  
  - Encapsulates the core math, algorithms, and signal-processing doctrine.  
  - Matches the patent filings (Stage-10【484】, Stage-11【483】, Appendices A+B【481】【482】).  
  - Used in the NGF article draft【480】.  
  - Provides deterministic rails: parser, warp, detect, denoise.  

- **Role:**  
  - Scientific **reference implementation**.  
  - Evolves slowly, versioned carefully (v0.1.0 = hardened reference).  
  - Cited in papers, patents, and benchmarks.  

---

## 2. **micro-lm** — Applied Adapters & Domains

- **What it is:**  
  A flexible integration layer that builds **domain-specific pipelines** (e.g. DeFi, ARC).  
  Each pipeline maps *prompt → primitive → rails → verifier*.

- **Why:**  
  - Keeps R&D agile (milestones M1–M11【485】【486】【487】).  
  - Domain-agnostic structure supports multiple tracks (DeFi, ARC, future domains).  
  - Provides demos and benchmarks at three breakpoints:  
    - Mapper-only (prompt → primitive)  
    - Rails-only (primitive → latent, via ngeodesic)  
    - End-to-end (prompt → primitive → rails → verifier)  

- **Role:**  
  - Rapid iteration, stakeholder demos, policy experiments.  
  - Calls into **ngeodesic** for deterministic rails; never re-implements them.  
  - Ships docs, configs, and artifacts for reproducibility.  

---

## Why This Split Matters

- **Stability vs. Flexibility**  
  - *ngeodesic* is stable science — patents + reference code.  
  - *micro-lm* is agile application — milestones, demos, fast iteration.  

- **Stakeholder clarity**  
  - NGF = “core science, hardened.”  
  - micro-lm = “applied adapters, domain pipelines.”  

- **Future-proof**  
  - Tier-2 (sidecar + real latents) will plug in at the *micro-lm* layer,  
    without destabilizing NGF core.

---

## TL;DR

- **ngeodesic v0.1.0** → hardened rails (Stage-10/11 NGF).  
- **micro-lm** → domain adapters (DeFi, ARC, …) calling those rails.  

Together: **science + application**, cleanly separated.
