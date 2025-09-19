# M11 Smoke — rails=stage11 vs baseline=stage10

## stage11

- **pass:** 2
- **fail:** 4
### Failures
- edge_ltv_withdraw_unsafe: expected no concrete action, got top1=withdraw_asset
- edge_hf_health_breach: expected no concrete action, got top1=borrow_asset
- edge_oracle_stale_price: expected no concrete action, got top1=borrow_asset
- edge_mapper_low_conf_or_nonexec: expected no concrete action, got top1=stake_asset

## stage10

- **pass:** 2
- **fail:** 4
### Failures
- edge_ltv_withdraw_unsafe: expected no concrete action, got top1=withdraw_asset
- edge_hf_health_breach: expected no concrete action, got top1=borrow_asset
- edge_oracle_stale_price: expected no concrete action, got top1=borrow_asset
- edge_mapper_low_conf_or_nonexec: expected no concrete action, got top1=stake_asset
