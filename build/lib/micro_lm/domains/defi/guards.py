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

def check_oracle(context: dict) -> Dict[str, Any]:
    oracle = (context or {}).get("oracle", {})
    age = float(oracle.get("age_sec", 0))
    max_age = float(oracle.get("max_age_sec", 30))
    if age > max_age:
        return {"ok": False, "reason": "oracle_stale"}
    return {"ok": True, "reason": "fresh"}

def check_ltv(policy: dict, ctx: dict) -> Dict[str, Any]:
    # Minimal stub: accept unless explicitly failing; ctx can carry ltv value
    ltv_max = float((policy or {}).get("ltv_max", 0.75))
    ltv = float((ctx or {}).get("ltv", 0.0))
    if ltv > ltv_max:
        return {"ok": False, "reason": "ltv"}
    return {"ok": True, "reason": "ok"}

def check_hf(policy: dict, ctx: dict) -> Dict[str, Any]:
    hf_min = float((policy or {}).get("hf_min", 1.0))
    hf = float((ctx or {}).get("hf", 1.0))
    if hf < hf_min:
        return {"ok": False, "reason": "hf"}
    return {"ok": True, "reason": "ok"}
