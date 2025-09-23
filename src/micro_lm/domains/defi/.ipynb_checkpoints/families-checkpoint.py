# Minimal DeFi family router for Tier-2 strict tests.
# Replace with real WDD envelopes later.
import re
from typing import Dict, Any, List

_DEPOSIT = re.compile(r"\b(deposit|supply|provide|add liquidity|top up|stake|move)\b", re.I)
_SWAP    = re.compile(r"\b(swap|convert|trade|exchange)\b", re.I)

def route_families(prompt: str, policy: Dict[str, Any]) -> Dict[str, Any]:
    keep: List[str] = []
    order: List[str] = []
    if _DEPOSIT.search(prompt): keep.append("deposit_asset"); order.append("deposit_asset")
    if _SWAP.search(prompt):
        keep.append("swap_asset")
        if "swap_asset" not in order: order.insert(0, "swap_asset")
    # dedupe order
    seen=set(); order=[x for x in order if not (x in seen or seen.add(x))]
    keep=list(dict.fromkeys(keep))
    return {"keep": keep, "order": order, "route": ["defi"], "reason": "family_stub"}
