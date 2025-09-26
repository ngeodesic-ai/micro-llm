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
