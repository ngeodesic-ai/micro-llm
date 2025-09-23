# Stage‑8 Tier‑2 Benchmark — Report

- Status: ❌ fail
- Rails: `stage11`  •  T=180  •  runs=1

## Metrics

- DeFi edges: 0/4 caught  •  false approvals: 4
- DeFi exec accuracy: 1.000
- ARC accuracy: 0.000  (n=0)

## Failures

- edge_ltv_withdraw_unsafe: expected verify.ok=False, got v={'ok': True, 'reason': 'shim:accept:stage-4'}
- edge_hf_health_breach: expected verify.ok=False, got v={'ok': True, 'reason': 'shim:accept:stage-4'}
- edge_oracle_stale: expected verify.ok=False, got v={'ok': True, 'reason': 'shim:accept:stage-4'}
- edge_low_conf_nonexec: expected verify.ok=False, got v={'ok': True, 'reason': 'shim:accept:stage-4'}
