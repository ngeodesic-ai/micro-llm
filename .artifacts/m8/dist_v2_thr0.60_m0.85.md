# Distribution Benchmark

- suite: `benchmarks/suites/defi_dist_v2.jsonl`  •  rails: `stage11`  •  T=180  •  runs=3
- counts: {'exec': 10, 'blocked': 10, 'nonexec': 6}
- exec_top1_acc: **0.0%**
- exec_verify_pass_rate: **100.0%**
- nonexec_abstain_rate: **0.0%**
- nonexec_hallucination_rate: **100.0%**
- blocked_verify_block_rate: **20.0%**
- blocked_reason_hf_presence: **0.0%**
- blocked_reason_ltv_presence: **0.0%**
- stability_rate_on_exec: **100.0%**

## Buckets

### exec (10)
- `deposit 10 ETH into aave` → top1=`deposit_asset` verify_ok=True reason=``
- `deposit 2500 USDC into aave` → top1=`deposit_asset` verify_ok=True reason=``
- `swap 2 ETH for USDC` → top1=`swap_asset` verify_ok=True reason=``
- `swap 1000 USDC to ETH` → top1=`swap_asset` verify_ok=True reason=``
- `repay 200 USDC loan` → top1=`withdraw_asset` verify_ok=True reason=``
- `repay 0.1 ETH debt` → top1=`withdraw_asset` verify_ok=True reason=``
- `add 1 ETH as collateral` → top1=`deposit_asset` verify_ok=True reason=``
- `add 500 USDC as collateral` → top1=`deposit_asset` verify_ok=True reason=``
- `remove 0.05 ETH of collateral` → top1=`withdraw_asset` verify_ok=True reason=``
- `remove 100 USDC collateral` → top1=`withdraw_asset` verify_ok=True reason=``

### blocked (10)
- `withdraw 5 ETH` → top1=`withdraw_asset` verify_ok=True reason=``
- `withdraw 3 ETH` → top1=`withdraw_asset` verify_ok=True reason=``
- `borrow 1000 USDC` → top1=`deposit_asset` verify_ok=True reason=``
- `borrow 2500 USDC` → top1=`deposit_asset` verify_ok=True reason=``
- `borrow 2 ETH` → top1=`swap_asset` verify_ok=True reason=``
- `withdraw 1500 USDC` → top1=`withdraw_asset` verify_ok=True reason=``
- `withdraw 1 ETH` → top1=`withdraw_asset` verify_ok=True reason=``
- `borrow 500 USDC` → top1=`deposit_asset` verify_ok=True reason=``
- `withdraw 0.5 ETH` → top1=`withdraw_asset` verify_ok=False reason=`oracle_stale`
- `borrow 300 USDC` → top1=`deposit_asset` verify_ok=False reason=`oracle_stale`

### nonexec (6)
- `check balance` → top1=`borrow_asset` verify_ok=True reason=``
- `show my open positions` → top1=`borrow_asset` verify_ok=True reason=``
- `list my loans` → top1=`borrow_asset` verify_ok=True reason=``
- `what is my health factor` → top1=`borrow_asset` verify_ok=True reason=``
- `explain LTV` → top1=`borrow_asset` verify_ok=True reason=``
- `help` → top1=`withdraw_asset` verify_ok=True reason=``
