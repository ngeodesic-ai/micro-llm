Rails:     stage11
Runs/Case: 5
OK:        False
F1 (micro/macro): 0.500 / 0.333

- prompt:   deposit 10 ETH into aave
  expected: deposit_asset
  stable:   True
  top1:     deposit_asset
  top1_list: ['deposit_asset', 'deposit_asset', 'deposit_asset', 'deposit_asset', 'deposit_asset']

- prompt:   swap 2 ETH for USDC
  expected: swap_asset
  stable:   True
  top1:     swap_asset
  top1_list: ['swap_asset', 'swap_asset', 'swap_asset', 'swap_asset', 'swap_asset']

- prompt:   withdraw 5 ETH
  expected: withdraw_asset
  stable:   True
  top1:     None
  top1_list: [None, None, None, None, None]

- prompt:   borrow 1000 USDC
  expected: borrow_asset
  stable:   True
  top1:     None
  top1_list: [None, None, None, None, None]

- prompt:   repay 500 USDC
  expected: repay_loan
  stable:   True
  top1:     swap_asset
  top1_list: ['swap_asset', 'swap_asset', 'swap_asset', 'swap_asset', 'swap_asset']

- prompt:   check balance
  expected: None
  stable:   True
  top1:     None
  top1_list: [None, None, None, None, None]
