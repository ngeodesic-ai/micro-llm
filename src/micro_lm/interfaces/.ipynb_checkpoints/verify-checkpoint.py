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

from typing import Protocol, Dict, Any, List, TypedDict, Optional

class VerifyResult(TypedDict, total=False):
    ok: bool
    reason: str
    tags: List[str]
    aux: Dict[str, Any]

class Verifier(Protocol):
    def __call__(
        self,
        prompt: str,
        plan: Dict[str, Any],
        *,
        context: Dict[str, Any],
        policy: Dict[str, Any]
    ) -> VerifyResult: ...

# example “no-ops are safe” verifier you can swap out
def safe_default_verifier(prompt, plan, *, context, policy) -> VerifyResult:
    # approve only if plan contains exactly one safe, known primitive
    seq = (plan or {}).get("sequence") or []
    known = {"deposit_asset", "swap_asset", "withdraw_asset", "borrow_asset", "repay_asset", "stake_asset", "unstake_asset", "claim_rewards"}
    if len(seq) == 1 and (seq[0] or {}).get("op") in known:
        return {"ok": True, "reason": "single_known_primitive"}
    return {"ok": False, "reason": "low_confidence_or_empty", "tags": ["abstain"]}
