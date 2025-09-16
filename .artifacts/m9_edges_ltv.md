# Distribution Benchmark

- suite: `benchmarks/suites/defi_edges_ltv_m9.jsonl`  •  rails: `stage11`  •  T=180  •  runs=1
- counts: {'exec': 50, 'nonexec': 0, 'blocked': 0}
- exec_top1_acc: **0.0%**
- exec_verify_pass_rate: **100.0%**
- nonexec_abstain_rate: **0.0%**
- nonexec_hallucination_rate: **0.0%**
- blocked_verify_block_rate: **0.0%**
- blocked_reason_hf_presence: **0.0%**
- blocked_reason_ltv_presence: **0.0%**
- stability_rate_on_exec: **100.0%**

## Buckets

### exec (50)
- `take a max LTV loan on aave (ethereum) — push it to the limit and borrow as much DAI as possible` → top1=`borrow_asset` verify_ok=True reason=``
- `open debt on compound ethereum at 90% LTV in MATIC using WBTC collateral` → top1=`borrow_asset` verify_ok=True reason=``
- `borrow 20000 USDC from aave on avalanche using just 1000 OP as collateral` → top1=`borrow_asset` verify_ok=True reason=``
- `open debt on maker optimism at 90% LTV in AAVE using ARB collateral` → top1=`borrow_asset` verify_ok=True reason=``
- `borrow 20000 ARB from aave on avalanche using just 1000 WETH as collateral` → top1=`borrow_asset` verify_ok=True reason=``
- `draw ETH on aave avalanche with minimal collateral — target 85% LTV` → top1=`borrow_asset` verify_ok=True reason=``
- `max out LTV and borrow WBTC against tiny SOL on maker (polygon)` → top1=`borrow_asset` verify_ok=True reason=``
- `take a max LTV loan on maker (optimism) — push it to the limit and borrow as much AAVE as possible` → top1=`borrow_asset` verify_ok=True reason=``
- `take a max LTV loan on compound (optimism) — push it to the limit and borrow as much WETH as possible` → top1=`borrow_asset` verify_ok=True reason=``
- `borrow 20000 AAVE from aave on ethereum using just 1000 BTC as collateral` → top1=`borrow_asset` verify_ok=True reason=``

### nonexec (0)

### blocked (0)
