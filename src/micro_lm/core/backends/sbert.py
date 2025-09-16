from __future__ import annotations
from typing import Any, Dict, Tuple

class SBertMapper:
    """
    Stub for Tier-1 encoder. Replace with actual model loading in Stage-2/3.
    """
    def __init__(self, *, domain: str, policy: Dict[str, Any]):
        self.domain = domain
        self.policy = policy
        # In Stage-2 we may read a joblib from configs or artifacts.

    def map_prompt(self, prompt: str) -> Tuple[str, float, Dict[str, Any]]:
        # Minimal heuristic so tests pass without heavy deps
        p = prompt.lower().strip()
        if not p:
            return "abstain", 0.0, {"reason": "empty"}
        if "swap" in p:
            return "swap_assets", 0.72, {"reason": "heuristic:swap"}
        if "deposit" in p:
            return "deposit_asset", 0.71, {"reason": "heuristic:deposit"}
        return "abstain", 0.49, {"reason": "uncertain"}
