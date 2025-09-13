import re
PRIMS = ["deposit_asset","withdraw_asset","borrow_asset","repay_loan","swap_asset","add_collateral","remove_collateral"]

_PATTERNS = {
  "deposit_asset":      [r"\b(deposit|supply|add)\b.*\b(collateral)?\b"],
  "withdraw_asset":     [r"\b(withdraw|redeem|remove)\b(?!.*collateral)"],
  "borrow_asset":       [r"\b(borrow|draw|take\s+loan)\b"],
  "repay_loan":         [r"\b(repay|pay\s*back|return)\b.*\b(loan)?\b"],
  "swap_asset":         [r"\b(swap|trade|convert)\b", r"(sell\s+\w+.*buy\s+\w+)", r"(→|->)"],
  "add_collateral":     [r"\b(add|increase)\b.*\bcollateral\b"],
  "remove_collateral":  [r"\b(remove|decrease|take\s+out)\b.*\bcollateral\b"],
}

def rule_prior(text: str, boost: float = 1.0):
    t = text.lower()
    prior = {k: 0.0 for k in PRIMS}
    for k, pats in _PATTERNS.items():
        if any(re.search(p, t) for p in pats):
            prior[k] = boost
    # weak hint for swaps when both tokens appear
    if re.search(r"\beth\b.*\busdc\b|\busdc\b.*\beth\b", t):
        prior["swap_asset"] = max(prior["swap_asset"], 0.5*boost)
    s = sum(prior.values())
    if s > 0:
        for k in prior: prior[k] /= s
    return prior
