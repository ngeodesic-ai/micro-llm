# Distribution Benchmark

- suite: `benchmarks/suites/defi_edges_hf.jsonl`  •  rails: `stage11`  •  T=180  •  runs=3
- counts: {'exec': 2, 'blocked': 3, 'nonexec': 1}
- exec_top1_acc: **0.0%**
- exec_verify_pass_rate: **100.0%**
- nonexec_abstain_rate: **100.0%**
- nonexec_hallucination_rate: **0.0%**
- blocked_verify_block_rate: **100.0%**
- blocked_reason_hf_presence: **66.67%**
- blocked_reason_ltv_presence: **33.33%**
- stability_rate_on_exec: **100.0%**

## Buckets

### exec (2)
- `deposit 10 ETH into aave` → top1=`deposit_asset` verify_ok=True reason=``
- `swap 2 ETH for USDC` → top1=`swap_asset` verify_ok=True reason=``

### blocked (3)
- `borrow 1000 USDC` → top1=`None` verify_ok=False reason=`hf`
- `borrow 100 USDC` → top1=`None` verify_ok=False reason=`hf`
- `withdraw 5 ETH` → top1=`None` verify_ok=False reason=`ltv`

### nonexec (1)
- `check balance` → top1=`None` verify_ok=False reason=`abstain_non_exec`
