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

from typing import Dict, Any
from .schema import DEFI_LABELS, REASONS
from .guards import check_oracle, check_ltv, check_hf

def verify_action_local(*, label: str, context: dict, policy: dict) -> Dict[str, Any]:
    # Only handle our known labels locally; otherwise abstain to shim/rails.
    if label not in DEFI_LABELS:
        return {"ok": False, "reason": REASONS["abstain"]}

    # Oracle freshness
    o = check_oracle(context)
    if not o["ok"]:
        return {"ok": False, "reason": REASONS["oracle"]}

    # Basic policy checks (apply only to exec-like labels)
    if label in {"withdraw_asset", "borrow_asset"}:
        ltvc = check_ltv(policy, context)
        if not ltvc["ok"]:
            return {"ok": False, "reason": REASONS["ltv"]}

    if label in {"borrow_asset"}:
        hfc = check_hf(policy, context)
        if not hfc["ok"]:
            return {"ok": False, "reason": REASONS["hf"]}

    # If we get here, it’s locally “safe”; the rails shim can still refine.
    return {"ok": True, "reason": REASONS["ok"]}
