#!/usr/bin/env python3
import json, random, argparse

"""
mkdir -p tools benchmarks/suites
python3 milestones/m8_make_defi_suite.py \
  --out benchmarks/suites/defi_dist_v3.jsonl \
  --per_class 250 \
  --nonexec 750 \
  --seed 13

wc -l benchmarks/suites/defi_dist_v3.jsonl   # expect ~2500
head -3 benchmarks/suites/defi_dist_v3.jsonl
"""

PRIMS = {
  "deposit_asset": [
    "deposit {amt} {token} into {proto}",
    "top up {amt} {token} on {proto}",
    "add {amt} {token} to {proto}",
  ],
  "withdraw_asset": [
    "withdraw {amt} {token} from {proto}",
    "remove {amt} {token} from {proto}",
    "pull out {amt} {token} on {proto}",
  ],
  "borrow_asset": [
    "borrow {amt} {token} on {proto}",
    "take a loan of {amt} {token} from {proto}",
  ],
  "repay_loan": [
    "repay {amt} {token} on {proto}",
    "pay back {amt} {token} on {proto}",
  ],
  "swap_asset": [
    "swap {amt} {base} for {quote}",
    "convert {amt} {base} to {quote}",
  ],
  "add_collateral": [
    "add {amt} {token} as collateral on {proto}",
    "post {amt} {token} collateral to {proto}",
  ],
  "remove_collateral": [
    "remove {amt} {token} collateral on {proto}",
    "unpost {amt} {token} collateral from {proto}",
  ],
}

NONEXEC_TEMPLATES = [
  "what's my balance on {proto}?",
  "show my health factor",
  "how do I add collateral?",
  "what are the current borrow rates for {token}?",
  "help",
  "explain LTV on {proto}",
  "what assets can I use as collateral?",
]

TOKENS = ["USDC", "USDT", "DAI", "ETH", "WBTC"]
PROTOS = ["aave", "compound", "spark", "morpho"]
AMTS   = ["50","100","250","500","1000","2500","1","2","5","10"]

def one_exec(prim, rng):
  tpl = rng.choice(PRIMS[prim])
  token = rng.choice(TOKENS)
  proto = rng.choice(PROTOS)
  base  = rng.choice([t for t in TOKENS if t != token])
  quote = rng.choice([t for t in TOKENS if t != base])
  amt   = rng.choice(AMTS)
  prompt = tpl.format(amt=amt, token=token, proto=proto, base=base, quote=quote)
  return {"prompt": prompt, "kind": "exec", "expect": {"top1": prim}}

def one_nonexec(rng):
  tpl = rng.choice(NONEXEC_TEMPLATES)
  token = rng.choice(TOKENS); proto = rng.choice(PROTOS)
  return {"prompt": tpl.format(token=token, proto=proto), "kind": "nonexec", "expect": {"top1": ""}}

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="benchmarks/suites/defi_dist_v3.jsonl")
  ap.add_argument("--seed", type=int, default=13)
  ap.add_argument("--per_class", type=int, default=300, help="exec prompts per primitive")
  ap.add_argument("--nonexec", type=int, default=600, help="non-exec prompts")
  a = ap.parse_args()

  rng = random.Random(a.seed)
  rows = []
  for prim in sorted(PRIMS):           # balanced execs across primitives
    for _ in range(a.per_class):
      rows.append(one_exec(prim, rng))
  for _ in range(a.nonexec):           # pooled non-execs
    rows.append(one_nonexec(rng))

  rng.shuffle(rows)
  with open(a.out, "w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")
  print(f"Wrote {len(rows)} rows to {a.out}")
