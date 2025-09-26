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

from typing import Any, Dict, List
from micro_lm.core.interfaces.exec_planner import ExecPlanner, Plan, PlanStep

class RulePlanner(ExecPlanner):
    def plan(self, *, intent: str, text: str, context: Dict[str, Any]) -> Plan:
        steps: List[PlanStep] = []
        if intent == "deposit_asset":
            steps = [PlanStep("parse_amount_asset_venue", {"text": text}),
                     PlanStep("call:aave.deposit", {"source": "account"})]
            rationale = "Deposit intent → parse then call aave.deposit"
        elif intent == "swap_asset":
            steps = [PlanStep("parse_pair_amount", {"text": text}),
                     PlanStep("call:uniswap.swap", {"slippage_bps": 30})]
            rationale = "Swap intent → parse pair then call uniswap.swap"
        else:
            steps = [PlanStep("noop", {"note": "unsupported or abstain"})]
            rationale = "No concrete plan for this intent"
        return Plan(steps=steps, rationale=rationale)
