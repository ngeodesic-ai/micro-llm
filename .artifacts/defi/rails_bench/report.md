# DeFi Rails Bench — Report

Runs: 1
Total cases: 3
ok: 2
ok_acc: 0.667

## Cases
- prompt: `deposit 10 ETH into aave` → pred=`` reason=`shim:accept:stage-4` ok=True
- prompt: `swap 2 ETH to USDC on uniswap v3` → pred=`` reason=`shim:accept:stage-4` ok=True
- prompt: `borrow 500 USDC with ETH collateral at 70% ltv` → pred=`` reason=`shim:accept:stage-11` ok=False