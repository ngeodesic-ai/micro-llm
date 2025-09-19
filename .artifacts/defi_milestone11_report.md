# Milestone 11 — Consolidated Tier-1 Benchmark (DeFi)

- Status: ❌ fail
- Rails: `stage11`  •  Baseline: `stage10`  •  T=180  •  runs=3

## Metrics — Stage-11

- accuracy: **0.0%**
- precision: **0.0%**  •  recall: **0.0%**  •  F1: **0.0%**
- hallucination: **0.00%**  •  omission: **100.00%**
- abstain: **100.00%**

## Metrics — Baseline

- accuracy: **0.0%**
- precision: **0.0%**  •  recall: **0.0%**  •  F1: **0.0%**
- hallucination: **0.00%**  •  omission: **100.00%**
- abstain: **100.00%**

## Failures
- ok_deposit: expected top1=deposit_asset, got=None
- ok_swap: expected top1=swap_asset, got=None

## Scenarios (brief)
### Main — `stage11`
- **ok_deposit** [exec_ok] → decision: `reject` • top1: `None` • verify.ok: `False` • reason: `abstain_non_exec`
- **ok_swap** [exec_ok] → decision: `reject` • top1: `None` • verify.ok: `False` • reason: `abstain_non_exec`

### Baseline — `stage10`
- **ok_deposit** [exec_ok] → decision: `reject` • top1: `None` • verify.ok: `False` • reason: `abstain_non_exec`
- **ok_swap** [exec_ok] → decision: `reject` • top1: `None` • verify.ok: `False` • reason: `abstain_non_exec`
