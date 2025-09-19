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
