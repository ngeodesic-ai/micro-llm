from __future__ import annotations
from typing import Any, Dict, Tuple


class WordMapMapper:
    """
    Extremely simple keyword/phrase matcher for Tier-0 fallback.
    Per-domain vocab lives with adapters later; here we use a tiny default.
    """

    def __init__(self, *, domain: str, policy: Dict[str, Any]):
        self.domain = domain
        self.policy = policy
        self.vocab = {
            "defi": {
                "deposit": "deposit_asset",
                "withdraw": "withdraw_asset",
                "borrow": "borrow_asset",
                "repay": "repay_debt",
                "swap": "swap_assets",
            },
            "arc": {
                "count": "count_objects",
                "extend": "extend_pattern",
                "flip": "flip_tile",
            },
        }.get(domain, {})

    def map_prompt(self, prompt: str) -> Tuple[str, float, Dict[str, Any]]:
        p = prompt.lower()
        for k, v in self.vocab.items():
            if k in p:
                return v, 0.66, {"reason": f"word:{k}"}
        return "abstain", 0.0, {"reason": "no_match"}
