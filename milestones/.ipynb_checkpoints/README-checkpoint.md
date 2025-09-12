# DeFi Milestone Plan (Condensed to 5)

This document condenses the NGF 12-step process into **5 parsable milestones** for DeFi, aligned with the **Warp → Detect → Denoise** doctrine described in the patents and article.  

Each milestone is **script-driven** (e.g., `defi_milestoneX.py`), validated, then folded into the `micro_llm` package with benchmarks and tests.

---

## Milestone 0 — Baseline ✅ (done)
- **Adapters**: simple rule + stub mapper (ARC + DeFi).  
- **Residual traces + priors**.  
- **Stage-10 rails** (matched filter + dual thresholds).  
- **Basic verifiers** (ARC grid ops, DeFi invariants).  
- **End-to-end CLI runs working**.  
➡️ Baseline ARC prompts working, components folded into pkg

---

## Milestone 1 — Hybrid Mapper + Prior Injection ✅ (done)
- Integrate **trained mapper** (scikit-learn joblib stub).  
- **Confidence scoring** + abstain wiring.  
- Adapter residuals boosted with **priors**.  
- Stage-10 rails respect priors when ordering.  
➡️ Produces a stronger front-end mapping from **prompt → features → traces**.

---

## Milestone 2 — Stage-11 Warp + Detect ✅ (done)
- **Warp**: funnel fit, PCA(3), curvature metrics.  
- **Detect**: matched filtering with calibrated nulls (permutation / circular shifts).  
- **Output**: stable well identification + ordered primitives.  
➡️ First NGF-native reasoning path (without denoiser).

---

## Milestone 3 — Stage-11 Denoise + Safety Guards ✅ (done)
- **Hybrid EMA+median** smoothing.  
- **Confidence gates** + noise floor rejection.  
- **Phantom-guard probes**.  
- **Monte Carlo jitter averaging**.  
- **Logging**: SNR, phantom index, hallucination/omission rates.  
➡️ Yields deterministic suppression of phantom wells.

---

## Milestone 4 — Consolidation + Benchmarks
- Fold into **micro_llm package**.  
- ARC + DeFi **benchmark suites** (latent ARC, toy DeFi scenarios).  
- **Test harness** for hallucination/abstain metrics.  
- Push to GitHub, **tag release**.  
➡️ Demonstrates end-to-end deterministic micro-LLM reasoning.

---

## ✅ Checklist

| Milestone | Status |
|-----------|--------|
| 0 — Baseline | ✅ Done |
| 1 — Hybrid Mapper + Prior Injection | ✅ Done  |
| 2 — Stage-11 Warp + Detect | ✅ Done |
| 3 — Stage-11 Denoise + Safety Guards |  ✅ Done |
| 4 — Consolidation + Benchmarks | ⬜ Pending |
