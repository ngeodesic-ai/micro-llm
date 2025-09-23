# Stage 8 Test & Benchmark Checklist (Tier-2 Close-Out)

## 1. Core Harness Tests

- **Smoke tests**: run `run_micro` with simple DeFi + ARC prompts, ensure no crashes.  
- **Policy integration**: confirm LTV/HF/oracle checks still block unsafe actions.  
- **Profile mode**: verify `.artifacts/{domain}/profile/*` writes `profile.json` and `topline.json`.  

## 2. Audit Tooling

- **PCA priors**: round-trip load/apply tests (`test_pca_priors.py`) pass end-to-end.  
- **WDD audit**: confirm abstain behavior on ambiguous prompts.  
- **Combined audit path**: verify `policy={"audit":{"backend":"wdd","profile":True}}` runs without regressions.  

## 3. Determinism & Stability

- **Repeat runs (`--runs 5`)**: all outputs identical → determinism check.  
- **Perturbation robustness (`--perturb --perturb_k 3`)**: decisions stable across prompt/number jitter.  
- **Random seeds**: changing seed does not flip verdicts.  

## 4. Benchmark Suites

- **DeFi baseline**: re-run Milestone 10/11 style pipelines with Stage-2 code, compare accuracy vs Tier-1.  
- **ARC baseline**: run primitive set (flip/rotate/tile) and confirm expected abstains vs passes.  
- **Cross-domain consistency**: DeFi + ARC both supported under the same `run_micro` harness.  

## 5. Accuracy Preservation

- **Deposit/swap**: pass as `ok=True`.  
- **Withdraw (high LTV)**: blocked.  
- **Borrow (low HF)**: blocked.  
- **Oracle stale**: blocked.  
- **Non-exec prompts**: abstain only.  
- **ARC ops**: correct primitive mapping with <1% hallucination.  

## 6. Artifacts & Reports

- **JSON summaries**: `.artifacts/stage8_summary.json` created.  
- **Markdown reports**: `.artifacts/stage8_report.md` with human-readable breakdown.  
- **Status flag**: `ok=true/false` printed to console.  
