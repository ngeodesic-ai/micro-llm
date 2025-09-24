"""
# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Copyright 2025 Ian C. Moore (Provisional Patents #63/864,726, #63/865,437, #63/871,647 and #63/872,334)
# Email: ngeodesic@gmail.com
# Part of Noetic Geodesic Framework (NGF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import re

PRIMS = [
    "deposit_asset","withdraw_asset","swap_asset","borrow_asset",
    "repay_asset","stake_asset","unstake_asset","claim_rewards"
]

KEYS = {
    "deposit_asset":  [r"\bdeposit\b", r"\badd\b", r"\bsupply\b"],
    "withdraw_asset": [r"\bwithdraw\b", r"\bremove\b", r"\bpull\b"],
    "swap_asset":     [r"\bswap\b", r"\bconvert\b", r"\btrade\b"],
    "borrow_asset":   [r"\bborrow\b", r"\bdraw\b"],
    "repay_asset":    [r"\brepay\b", r"\bpay\s*back\b"],
    "stake_asset":    [r"\bstake\b", r"\brestake\b"],
    "unstake_asset":  [r"\bunstake\b", r"\bunlock\b"],
    "claim_rewards":  [r"\bclaim\b", r"\bcollect\b"]
}

class WordmapBackend:
    name = "wordmap"
    def __init__(self, confidence_threshold: float = 0.35, **kwargs) -> None:
        # Accept and ignore extra kwargs (e.g., model_path) for a uniform loader API
        self.th = float(confidence_threshold)

    def predict(self, prompts: List[str]) -> List[Tuple[Optional[str], float]]:
        out: List[Tuple[Optional[str], float]] = []
        for p in prompts:
            p_l = p.lower()
            scores = {}
            for prim, pats in KEYS.items():
                s=0.0
                for pat in pats:
                    if re.search(pat, p_l):
                        s += 0.5
                scores[prim] = min(s, 1.0)
            best = max(scores.items(), key=lambda kv: kv[1])
            label, conf = best
            out.append((label if conf >= self.th else None, conf))
        return out
