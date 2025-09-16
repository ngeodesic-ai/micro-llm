# Milestone 11 — Consolidated Tier‑1 Benchmark (DeFi)

- Status: ✅ pass
- Rails: `stage11`  •  Baseline: `stage10`  •  T=180  •  runs=5

## Metrics — Stage‑11

- accuracy: **100.0%**
- precision: **100.0%**  •  recall: **100.0%**  •  F1: **100.0%**
- hallucination: **0.00%**  •  omission: **0.00%**
- abstain: **50.00%**

## Metrics — Baseline

- accuracy: **100.0%**
- precision: **100.0%**  •  recall: **100.0%**  •  F1: **100.0%**
- hallucination: **0.00%**  •  omission: **0.00%**
- abstain: **50.00%**

## Scenarios (brief)
### Main — `stage11`
- **edge_ltv_withdraw_unsafe** [edge] → decision: `reject` • top1: `None` • verify.ok: `False` • reason: `abstain_non_exec`
- **edge_hf_health_breach** [edge] → decision: `reject` • top1: `None` • verify.ok: `False` • reason: `abstain_non_exec`
- **edge_oracle_stale_price** [edge] → decision: `reject` • top1: `None` • verify.ok: `False` • reason: `oracle_stale`
- **edge_mapper_low_conf_or_nonexec** [edge] → decision: `reject` • top1: `None` • verify.ok: `False` • reason: `abstain_non_exec`
- **ok_deposit** [exec_ok] → decision: `approve` • top1: `deposit_asset` • verify.ok: `True` • reason: ``
- **ok_swap** [exec_ok] → decision: `approve` • top1: `swap_asset` • verify.ok: `True` • reason: ``

### Baseline — `stage10`
- **edge_ltv_withdraw_unsafe** [edge] → decision: `reject` • top1: `None` • verify.ok: `False` • reason: `abstain_non_exec`
- **edge_hf_health_breach** [edge] → decision: `reject` • top1: `None` • verify.ok: `False` • reason: `abstain_non_exec`
- **edge_oracle_stale_price** [edge] → decision: `reject` • top1: `None` • verify.ok: `False` • reason: `oracle_stale`
- **edge_mapper_low_conf_or_nonexec** [edge] → decision: `reject` • top1: `None` • verify.ok: `False` • reason: `abstain_non_exec`
- **ok_deposit** [exec_ok] → decision: `approve` • top1: `deposit_asset` • verify.ok: `True` • reason: ``
- **ok_swap** [exec_ok] → decision: `approve` • top1: `swap_asset` • verify.ok: `True` • reason: ``
