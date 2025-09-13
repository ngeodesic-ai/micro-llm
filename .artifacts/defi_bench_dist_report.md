# Distribution Benchmark

- suite: `benchmarks/suites/defi_dist_v1.jsonl`  •  rails: `stage11`  •  T=180  •  runs=3
- counts: {'exec': 4, 'nonexec': 2, 'blocked': 2}
- exec_top1_acc: **50.0%**
- exec_verify_pass_rate: **50.0%**
- nonexec_abstain_rate: **100.0%**
- nonexec_hallucination_rate: **0.0%**
- blocked_verify_block_rate: **100.0%**
- blocked_reason_hf_presence: **50.0%**
- blocked_reason_ltv_presence: **50.0%**
- stability_rate_on_exec: **100.0%**

## Buckets

### exec (4)
- `deposit 10 ETH into aave` → top1=`deposit_asset` verify_ok=True reason=``
- `swap 2 ETH for USDC` → top1=`swap_asset` verify_ok=True reason=``
- `add 1 ETH collateral` → top1=`None` verify_ok=False reason=`abstain_non_exec`
- `repay 200 USDC` → top1=`None` verify_ok=False reason=`abstain_non_exec`

### nonexec (2)
- `check balance` → top1=`None` verify_ok=False reason=`abstain_non_exec`
- `what are current gas fees?` → top1=`None` verify_ok=False reason=`abstain_non_exec`

### blocked (2)
- `withdraw 5 ETH` → top1=`None` verify_ok=False reason=`ltv`
- `borrow 1000 USDC` → top1=`None` verify_ok=False reason=`hf`
