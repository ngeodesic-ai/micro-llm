#!/usr/bin/env python3
"""
gen_defi_prompts.py — synthesize labeled DeFi prompts for mapper/auditor benches.

Writes:
  - <out_dir>/defi_prompts_<N>.jsonl   (fields: {"prompt": ...})
  - <out_dir>/defi_labels_<N>.csv      (fields: index,label)

Usage:
  python3 gen_defi_prompts.py --out_dir .artifacts/defi/synth --per_class 15 --seed 42

This produces 8 classes × 15 = 120 prompts by default.
"""
from __future__ import annotations
import argparse, json, random, csv
from pathlib import Path

PRIMS = [
    "deposit_asset",
    "withdraw_asset",
    "swap_asset",
    "borrow_asset",
    "repay_asset",
    "stake_asset",
    "unstake_asset",
    "claim_rewards",
]

VERBS = {
    "deposit_asset":  ["deposit", "supply", "provide", "add liquidity to", "top up"],
    "withdraw_asset": ["withdraw", "redeem", "remove liquidity from", "unstake and take out"],
    "swap_asset":     ["swap", "convert", "trade", "exchange"],
    "borrow_asset":   ["borrow", "draw", "open a loan for"],
    "repay_asset":    ["repay", "pay back", "close out the loan for"],
    "stake_asset":    ["stake", "lock", "bond"],
    "unstake_asset":  ["unstake", "unlock", "unbond"],
    "claim_rewards":  ["claim rewards from", "harvest", "collect rewards on"],
}

PROTOCOLS = ["aave", "compound", "uniswap", "sushiswap", "balancer", "curve", "yearn", "stargate"]
NETWORKS = ["ethereum", "arbitrum", "base", "optimism", "polygon", "avalanche", "solana"]
TOKENS = ["ETH","WETH","WBTC","USDC","USDT","DAI","OP","ARB","MATIC","SOL","AVAX","LINK","AAVE"]
MODES = ["— safe mode", "— asap", "— minimize gas", "— use normal gas", "— low slippage", ""]

TEMPLATES = {
    "single_asset_on_protocol": "{verb} {amt} {tok} {prep} {proto} on {net} {mode}",
    "single_asset_generic": "{verb} {amt} {tok} {mode}",
    "lp_style": "{verb} {amt} {tok} into {proto} on {net} {mode}",
    "swap_style": "{verb} {amt} {tok} to {tok2} on {proto} {mode}",
    "rewards_style": "{verb} {proto} on {net} {mode}",
}

def _amt():
    import random
    return f"{random.uniform(0.02, 5000):.4f}".rstrip("0").rstrip(".")

def make_prompt(label: str, rng: random.Random) -> str:
    tok = rng.choice(TOKENS)
    tok2 = rng.choice([t for t in TOKENS if t != tok])
    proto = rng.choice(PROTOCOLS)
    net = rng.choice(NETWORKS)
    mode = rng.choice(MODES)
    verb = rng.choice(VERBS[label])
    amt = _amt()

    if label == "swap_asset":
        tmpl = TEMPLATES["swap_style"]
        return tmpl.format(verb=verb, amt=amt, tok=tok, tok2=tok2, proto=proto, mode=mode, net=net, prep="into")
    elif label in ("claim_rewards",):
        tmpl = TEMPLATES["rewards_style"]
        return tmpl.format(verb=verb, proto=proto, net=net, mode=mode, amt=amt, tok=tok, tok2=tok2, prep="into")
    else:
        tmpl = rng.choice([TEMPLATES["single_asset_on_protocol"], TEMPLATES["lp_style"], TEMPLATES["single_asset_generic"]])
        prep = "to" if label in ("deposit_asset","stake_asset","repay_asset") else "from"
        return tmpl.format(verb=verb, amt=amt, tok=tok, proto=proto, net=net, mode=mode, prep=prep, tok2=tok2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=".artifacts/defi/synth")
    ap.add_argument("--per_class", type=int, default=15, help="samples per primitive (8 classes total)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = []
    labels = []
    for prim in PRIMS:
        for _ in range(args.per_class):
            p = make_prompt(prim, rng)
            prompts.append({"prompt": p})
            labels.append(prim)

    N = len(prompts)
    # small shuffle to mix classes
    idx = list(range(N))
    rng.shuffle(idx)
    prompts = [prompts[i] for i in idx]
    labels  = [labels[i]  for i in idx]

    jsonl_fp = out_dir / f"defi_prompts_{N}.jsonl"
    labels_fp = out_dir / f"defi_labels_{N}.csv"

    with jsonl_fp.open("w") as f:
        for row in prompts:
            f.write(json.dumps(row) + "\n")

    with labels_fp.open("w", newline="") as f:
        W = csv.writer(f)
        W.writerow(["index","label"])
        for i, lab in enumerate(labels):
            W.writerow([i, lab])

    print(f"Wrote {jsonl_fp}")
    print(f"Wrote {labels_fp}")

if __name__ == "__main__":
    main()
