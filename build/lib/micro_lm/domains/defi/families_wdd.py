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

from typing import List, Dict, Any
from micro_lm.core.audit import FamilySpec

_NAME_TO_IDXS = {
    "swap_like": [0,1,2],
    "deposit_like": [3,4],
    "withdraw_like": [5],
    "stake_like": [6],
    "unstake_like": [7],
    "borrow_like": [8,9],
    "repay_like": [10],
    "claim_rewards_like": [11],
}
def _safe(idxs, K): return [i for i in idxs if 0 <= i < K]

def defi_family_registry(K: int,
                         defaults: Dict[str, Any] = None,
                         overrides: Dict[str, Dict[str, Any]] = None) -> List[FamilySpec]:
    d = {"template_width": 64, "z_abs": 0.55, "keep_frac": 0.70}
    if defaults: d.update(defaults)
    overrides = overrides or {}
    fams: List[FamilySpec] = []
    for name, idxs in _NAME_TO_IDXS.items():
        spec = dict(d); spec.update(overrides.get(name, {}))
        fams.append(FamilySpec(
            name=name,
            idxs=_safe(idxs, K),
            z_abs=float(spec["z_abs"]),
            keep_frac=float(spec["keep_frac"]),
            template_width=int(spec["template_width"]),
        ))
    return fams
