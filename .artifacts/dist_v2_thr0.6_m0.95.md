# Distribution Benchmark

- suite: `benchmarks/suites/defi_dist_v2.jsonl`  •  rails: `stage11`  •  T=180  •  runs=3
- counts: {'exec': 10, 'blocked': 10, 'nonexec': 6}
- exec_top1_acc: **50.0%**
- exec_verify_pass_rate: **50.0%**
- nonexec_abstain_rate: **100.0%**
- nonexec_hallucination_rate: **0.0%**
- blocked_verify_block_rate: **90.0%**
- blocked_reason_hf_presence: **40.0%**
- blocked_reason_ltv_presence: **50.0%**
- stability_rate_on_exec: **100.0%**

## Buckets

### exec (10)
- `deposit 10 ETH into aave` → top1=`deposit_asset` verify_ok=True reason=``
- `deposit 2500 USDC into aave` → top1=`deposit_asset` verify_ok=True reason=``
- `swap 2 ETH for USDC` → top1=`swap_asset` verify_ok=True reason=``
- `swap 1000 USDC to ETH` → top1=`swap_asset` verify_ok=True reason=``
- `repay 200 USDC loan` → top1=`None` verify_ok=False reason=`hf`
- `repay 0.1 ETH debt` → top1=`None` verify_ok=False reason=`abstain_non_exec`
- `add 1 ETH as collateral` → top1=`None` verify_ok=False reason=`abstain_non_exec`
- `add 500 USDC as collateral` → top1=`borrow_asset` verify_ok=True reason=``
- `remove 0.05 ETH of collateral` → top1=`None` verify_ok=False reason=`abstain_non_exec`
- `remove 100 USDC collateral` → top1=`None` verify_ok=False reason=`abstain_non_exec`

### blocked (10)
- `withdraw 5 ETH` → top1=`None` verify_ok=False reason=`ltv`
- `withdraw 3 ETH` → top1=`None` verify_ok=False reason=`ltv`
- `borrow 1000 USDC` → top1=`None` verify_ok=False reason=`hf`
- `borrow 2500 USDC` → top1=`None` verify_ok=False reason=`hf`
- `borrow 2 ETH` → top1=`None` verify_ok=False reason=`hf`
- `withdraw 1500 USDC` → top1=`None` verify_ok=False reason=`ltv`
- `withdraw 1 ETH` → top1=`None` verify_ok=False reason=`ltv`
- `borrow 500 USDC` → top1=`borrow_asset` verify_ok=True reason=``
- `withdraw 0.5 ETH` → top1=`None` verify_ok=False reason=`ltv`
- `borrow 300 USDC` → top1=`None` verify_ok=False reason=`hf`

### nonexec (6)
- `check balance` → top1=`None` verify_ok=False reason=`abstain_non_exec`
- `show my open positions` → top1=`None` verify_ok=False reason=`abstain_non_exec`
- `list my loans` → top1=`None` verify_ok=False reason=`hf`
- `what is my health factor` → top1=`None` verify_ok=False reason=`abstain_non_exec`
- `explain LTV` → top1=`None` verify_ok=False reason=`abstain_non_exec`
- `help` → top1=`None` verify_ok=False reason=`abstain_non_exec`
