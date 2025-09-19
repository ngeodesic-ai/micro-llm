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
