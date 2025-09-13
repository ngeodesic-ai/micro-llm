# Milestone 5 Report

- Status: ✅ pass
- Rails: `stage11`  •  T=180  •  runs=5

## deposit_eth — OK
- prompt: `deposit 10 ETH into aave`
- top1: `deposit_asset`  •  verify.ok: `True`  •  reason: ``
  - top1_list: ['deposit_asset', 'deposit_asset', 'deposit_asset', 'deposit_asset', 'deposit_asset']  •  stable: True  •  top1(first): deposit_asset

## swap_eth_usdc — OK
- prompt: `swap 2 ETH for USDC`
- top1: `swap_asset`  •  verify.ok: `True`  •  reason: ``
  - top1_list: ['swap_asset', 'swap_asset', 'swap_asset', 'swap_asset', 'swap_asset']  •  stable: True  •  top1(first): swap_asset

## withdraw_high_ltv — OK
- prompt: `withdraw 5 ETH`
- top1: `None`  •  verify.ok: `False`  •  reason: `ltv`
  - top1_list: [None, None, None, None, None]  •  stable: True  •  top1(first): None

## borrow_low_hf — OK
- prompt: `borrow 1000 USDC`
- top1: `None`  •  verify.ok: `False`  •  reason: `hf`
  - top1_list: [None, None, None, None, None]  •  stable: True  •  top1(first): None

## nonexec_abstain — OK
- prompt: `check balance`
- top1: `None`  •  verify.ok: `False`  •  reason: `abstain_non_exec`
  - top1_list: [None, None, None, None, None]  •  stable: True  •  top1(first): None

## deposit_eth_denoised — OK
- prompt: `deposit 10 ETH into aave`
- top1: `deposit_asset`  •  verify.ok: `True`  •  reason: ``

## swap_eth_usdc_denoised — OK
- prompt: `swap 2 ETH for USDC`
- top1: `swap_asset`  •  verify.ok: `True`  •  reason: ``
