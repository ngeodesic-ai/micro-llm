# Milestone 9 — Verifier/Guards Benchmark (Tier-1)

- Status: ✅ pass
- Rails: `stage11`  •  T=180  •  runs=3

## Metrics

- Edge coverage: **4/4** = 100.0%
- False approvals: **0** (target **0**)
- Exec-ok accuracy: **2/2** = 100.0% (target ≥ 90%)

## Scenarios
### edge_ltv_withdraw_unsafe — OK
- type: `edge`
- prompt: `withdraw 5 ETH`
- stable_reason: `True`  •  reasons: ['ltv', 'ltv', 'ltv']
- top1: `None`  •  verify.ok: `False`  •  reason: `ltv`

### edge_hf_health_breach — OK
- type: `edge`
- prompt: `increase borrow to the maximum`
- stable_reason: `True`  •  reasons: ['hf', 'hf', 'hf']
- top1: `None`  •  verify.ok: `False`  •  reason: `hf`

### edge_oracle_stale_price — OK
- type: `edge`
- prompt: `borrow 1000 USDC`
- stable_reason: `True`  •  reasons: ['hf', 'hf', 'hf']
- top1: `None`  •  verify.ok: `False`  •  reason: `hf`

### edge_mapper_low_conf_or_nonexec — OK
- type: `edge`
- prompt: `stake xyz`
- stable_reason: `True`  •  reasons: ['abstain_non_exec', 'abstain_non_exec', 'abstain_non_exec']
- top1: `None`  •  verify.ok: `False`  •  reason: `abstain_non_exec`

### ok_deposit — OK
- type: `exec_ok`
- prompt: `deposit 10 ETH into aave`
- top1: `deposit_asset`  •  verify.ok: `True`  •  reason: ``

### ok_swap — OK
- type: `exec_ok`
- prompt: `swap 2 ETH for USDC`
- top1: `swap_asset`  •  verify.ok: `True`  •  reason: ``
