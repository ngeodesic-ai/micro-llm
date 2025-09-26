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
from micro_lm.core.interfaces.context_adapter import ContextAdapter, Context

class SimpleContextAdapter(ContextAdapter):
    def normalize(self, context: Dict[str, Any]) -> Context:
        oracle = (context or {}).get("oracle", {})
        account = (context or {}).get("account", {})
        market = (context or {}).get("market", {})
        return Context(
            raw=context or {},
            oracle_age_sec=oracle.get("age_sec"),
            oracle_max_age_sec=oracle.get("max_age_sec"),
            account_balances=(account.get("balances") or {}),
            venues=(market.get("venues") or []),
        )
