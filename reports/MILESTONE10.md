# Milestone 10 — Tier 1 Consolidation (Stage-11 Rails)

Milestone 10 pulls together everything from **Tier 1** into a single **end-to-end pipeline**:

- **Mapper** — maps free-form DeFi prompts into candidate actions.
- **Policy Guards** — enforce risk limits (e.g. `ltv_max`, `hf_min`).
- **Oracle Context** — validates freshness (`age_sec`, `max_age_sec`).
- **Stage-11 Rails** — Warp → Detect → Denoise (phantom suppression, deterministic funnels).
- **Verifier** — ensures actions match invariants (e.g. LTV/HF checks, abstain on non-exec).
- **Stability Harness** — repeated runs + perturbations to prove determinism.

This is the consolidation milestone: **Tier 1 is proven operational as a coherent system**.

---

## Purpose

- Verify **safe exec** paths (`deposit_asset`, `swap_asset`).
- Verify **blocked** paths (`withdraw` high-LTV, `borrow` low-HF).
- Verify **abstain** on non-exec prompts (e.g. balance checks).
- Check **determinism** across multiple runs.
- Check **robustness** under perturbations (prompt wording / numeric jitter).

---

## Usage

Run clean consolidation (no perturbations):

```bash
python3 milestones/defi_milestone10.py \
  --rails stage11 --runs 5 \
  --policy '{"ltv_max":0.75,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}' \
  --context '{"oracle":{"age_sec":5,"max_age_sec":30}}'
```

Run with perturbation tests:

```bash
python3 milestones/defi_milestone10.py \
  --rails stage11 --runs 5 --perturb --perturb_k 3 \
  --policy '{"ltv_max":0.75,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}' \
  --context '{"oracle":{"age_sec":5,"max_age_sec":30}}'
```

Inspect the results:

```bash
python3 milestones/inspect_summary.py .artifacts/defi_milestone10_summary.json
```

---

## Freeze knobs in a small config file:
```bash
cat > configs/m10_defaults.json <<'JSON'
{"rails":"stage11","runs":5,"T":180,
 "policy":{"ltv_max":0.75,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}},
 "context":{"oracle":{"age_sec":5,"max_age_sec":30}},
 "perturb": true, "perturb_k": 3}
JSON
```

## Outputs

- **Summary JSON:** `.artifacts/defi_milestone10_summary.json`  
  (machine-readable, includes per-scenario results, perturb variants, pass/fail).
- **Markdown Report:** `.artifacts/defi_milestone10_report.md`  
  (human-readable, scenario-by-scenario breakdown).
- **Status Flag:** printed `ok=true/false` on console.

Sample inspector output:

```
Milestone: defi_milestone10
Status:    pass
Rails:     stage11
T:         180
runs:      5

- deposit_eth: ok=True
  [run]
    prompt:   deposit 10 ETH into aave
    top1:     deposit_asset

- withdraw_high_ltv: ok=True
  [run]
    prompt:   withdraw 5 ETH
    top1:     None
```

---

## Interpretation

- ✅ **ok=True** → behavior matched expectations.
- `top1=deposit_asset` / `swap_asset` → safe exec paths succeed.
- `top1=None` + `ok=True` → blocked or abstain scenarios succeeded.
- Perturbation variants must not flip decisions (all stay `ok=True`).

---

## Next Steps

- **Milestone 11:** broaden coverage (more assets, amounts, phrasing), parameter sweeps, oracle edge-cases.
- **Milestone 12:** Tier 1 close-out with consolidated benchmark + release artifacts.

---

## References

- Tiered Plan (Tier 0 → Tier 2): [defi_tiers_plan.pdf](../defi_tiers_plan.pdf)
- Stage-11 Rails Benchmark: [stage11_benchmark_latest.py](../stage11_benchmark_latest.py)
- NGF Doctrine (Warp → Detect → Denoise): [micro_llm_12steps.pdf](../micro_llm_12steps.pdf)
