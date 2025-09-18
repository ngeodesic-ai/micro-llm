# Auditor Bench Report

## Metrics
```json
{
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0,
  "hallucination_rate_on_junk": 0.05,
  "abstain_rate_on_junk": 0.95,
  "n_test": 10,
  "n_junk": 20
}
```

## Errors (first 10)
- id=0 gold=deposit_asset pred= :: deposit 100 USDC
- id=1 gold=deposit_asset pred=repay_asset :: put in 1 ETH
- id=2 gold=withdraw_asset pred= :: take out 5 DAI
- id=3 gold=withdraw_asset pred=repay_asset :: take out 50 WBTC
- id=4 gold=borrow_asset pred=repay_asset :: obtain 50 USDC
- id=5 gold=borrow_asset pred= :: take loan of 5 WBTC
- id=6 gold=repay_asset pred= :: repay 50 USDC
- id=7 gold=repay_asset pred= :: settle 2 ETH debt
- id=8 gold=swap_asset pred= :: exchange 10 WBTC for USDC
- id=9 gold=swap_asset pred= :: swap 1 WBTC to ETH

## Junk that should abstain (first 10 that failed)
- id=9 pred=withdraw_asset :: who won the football match