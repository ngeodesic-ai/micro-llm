# micro_llm/verify/defi_verify.py
from __future__ import annotations
from typing import Dict, Any, List

def _get(d: Dict[str, Any], *path, default=None):
    x = d
    for p in path:
        if not isinstance(x, dict): return default
        x = x.get(p)
        if x is None: return default
    return x

def _ltv(collateral_value: float, debt_value: float) -> float:
    return 0.0 if collateral_value <= 0 else debt_value / collateral_value

def _hf(collateral_value: float, debt_value: float, liq_threshold: float) -> float:
    # HF = (collateral_value * LT) / debt_value; if no debt -> inf
    return float("inf") if debt_value <= 0 else (collateral_value * liq_threshold) / debt_value

def defi_verify(plan: Dict[str, List[str]], ctx: Dict[str, Any]) -> Dict[str, Any]:
    seq = plan.get("sequence", [])
    if not seq:
        return {"ok": False, "reason": "abstain_non_exec_or_empty"}

    # Pull policy knobs (with reasonable defaults)
    liq_thr = float(_get(ctx, "policy", "liq_threshold", default=0.85))
    ltv_max = float(_get(ctx, "policy", "ltv_max", default=0.75))

    # Minimal current portfolio snapshot
    coll_val = float(_get(ctx, "positions", "collateral_value", default=0.0))
    debt_val = float(_get(ctx, "positions", "debt_value", default=0.0))

    pre = {"ltv": _ltv(coll_val, debt_val), "hf": _hf(coll_val, debt_val, liq_thr)}

    action = seq[0]

    # Borrow worsens HF/LTV; require healthy buffer
    if action == "borrow_asset":
        if pre["hf"] < 1.10:   # buffer above 1.0
            return {"ok": False, "reason": "hf_too_low_for_borrow"}

    # Withdrawing / removing collateral tightens LTV; require margin
    if action in ("withdraw_asset", "remove_collateral"):
        if pre["ltv"] > (ltv_max * 0.90):
            return {"ok": False, "reason": "ltv_near_max_for_withdraw"}

    # Deposits, repays, add_collateral, and neutral swaps → OK in v0
    return {"ok": True, "reason": ""}