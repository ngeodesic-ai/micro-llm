# Milestone 10 Report

- Status: ✅ pass
- Rails: `stage11`  •  T=180  •  runs=1  •  perturb=True (k=1)

## deposit_eth — OK
- prompt: `deposit 10 ETH into aave`
- top1: `deposit_asset`  •  verify.ok: `True`
## swap_eth_usdc — OK
- prompt: `swap 2 ETH for USDC`
- top1: `swap_asset`  •  verify.ok: `True`
## withdraw_high_ltv — OK
- prompt: `withdraw 5 ETH`
- top1: `None`  •  verify.ok: `False`
## borrow_low_hf — OK
- prompt: `borrow 1000 USDC`
- top1: `None`  •  verify.ok: `False`
## nonexec_abstain — OK
- prompt: `check balance`
- top1: `None`  •  verify.ok: `False`
## deposit_eth_perturb — OK
- prompt: `add 10.1 ETH into aave`
- top1: `deposit_asset`  •  verify.ok: `True`
- perturb variants: 1  •  fails: 0
## swap_eth_usdc_perturb — OK
- prompt: `exchange 2.02 ETH for USDC`
- top1: `swap_asset`  •  verify.ok: `True`
- perturb variants: 1  •  fails: 0
## withdraw_high_ltv_perturb — OK
- prompt: `withdraw 4.95 ETH`
- top1: `None`  •  verify.ok: `False`
- perturb variants: 1  •  fails: 0
## borrow_low_hf_perturb — OK
- prompt: `borrow 990 USDC`
- top1: `None`  •  verify.ok: `False`
- perturb variants: 1  •  fails: 0
## nonexec_abstain_perturb — OK
- prompt: `check balance`
- top1: `None`  •  verify.ok: `False`
- perturb variants: 1  •  fails: 0