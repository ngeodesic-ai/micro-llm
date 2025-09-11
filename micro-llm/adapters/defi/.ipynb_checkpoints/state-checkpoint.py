from typing import Dict, Any

def normalize_defi_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "balances": ctx.get("balances", {}),
        "debt": ctx.get("debt", {}),
        "prices": ctx.get("prices", {}),
        "rates": ctx.get("rates", {}),
        "liquidity": ctx.get("liquidity", {}),
        "risk": ctx.get("risk", {"hf": 1.2}),
        "oracle": ctx.get("oracle", {"age_sec": 0, "max_age_sec": 30}),
    }
