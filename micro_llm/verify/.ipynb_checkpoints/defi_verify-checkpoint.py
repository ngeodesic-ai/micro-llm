# micro_llm/verify/defi_verify.py
from __future__ import annotations
from typing import Dict, Any, List

# -------------------- DEBUG BLOCK -------------------- #
import os, sys, json

def _dbg_on() -> bool:
    # Set MICROLLM_DEBUG=1 to enable prints
    return os.getenv("MICROLLM_DEBUG", "").strip() not in ("", "0", "false", "False")

def _dprint(*args, **kwargs):
    if _dbg_on():
        print(*args, file=sys.stderr, **kwargs)

def _snap(v):
    try:
        return json.dumps(v)
    except Exception:
        return repr(v)

def _get(d: Dict[str, Any], *path, default=None):
    x = d
    for p in path:
        if not isinstance(x, dict):
            return default
        x = x.get(p)
        if x is None:
            return default
    return x
# -------------------- DEBUG BLOCK -------------------- #


def _ltv(collateral_value: float, debt_value: float) -> float:
    return 0.0 if collateral_value <= 0 else debt_value / collateral_value

def _hf(collateral_value: float, debt_value: float, liq_threshold: float) -> float:
    # HF = (collateral_value * LT) / debt_value; if no debt -> inf
    return float("inf") if debt_value <= 0 else (collateral_value * liq_threshold) / debt_value


def _canon(reason: str, strict: bool) -> str:
    """
    Collapse verbose verifier reasons to canonical tokens if strict=True.
    """
    if not strict:
        return reason
    if reason in ("", None):
        return ""
    # map verbose → tokens expected by runner/bench
    if "ltv" in reason:
        return "ltv"
    if "hf" in reason:
        return "hf"
    if "oracle" in reason:
        return "oracle"
    if "abstain_non_exec" in reason or "empty" in reason:
        return "abstain_non_exec"
    return reason


def defi_verify(plan: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Policy verifier for DeFi plans.
    Honors:
      - ctx.policy.ltv_max (float, default 0.75)
      - ctx.policy.liq_threshold (float, default 0.85)
      - ctx.policy.strict_reason_tokens (bool, default False)
      - ctx.oracle.age_sec / ctx.oracle.max_age_sec
      - ctx.risk.ltv / ctx.risk.hf (if provided by adapter)
      - ctx.positions.collateral_value / debt_value (for pre-state checks)
    """
    plan = plan or {}
    ctx = ctx or {}

    seq: List[str] = plan.get("sequence") or []
    pol: Dict[str, Any] = ctx.get("policy") or {}
    risk: Dict[str, Any] = ctx.get("risk") or {}
    oracle: Dict[str, Any] = ctx.get("oracle") or {}
    strict_tokens: bool = bool(pol.get("strict_reason_tokens", False))

    # Debug header
    _dprint(
        "[defi_verify] ▶ enter",
        " seq=", seq,
        " policy=", _snap(pol),
        " risk=", _snap(risk),
        " oracle=", _snap(oracle),
    )

    # Empty or non-exec plan (runner might already treat as abstain)
    if not seq:
        reason = _canon("abstain_non_exec_or_empty", strict_tokens)
        _dprint("[defi_verify] rule=EMPTY_PLAN →", reason)
        return {"ok": False, "reason": reason}

    # Pull policy knobs (with defaults)
    liq_thr = float(_get(ctx, "policy", "liq_threshold", default=0.85))
    ltv_max = float(_get(ctx, "policy", "ltv_max", default=0.75))

    # Minimal current snapshot for pre-state checks
    coll_val = float(_get(ctx, "positions", "collateral_value", default=0.0))
    debt_val = float(_get(ctx, "positions", "debt_value", default=0.0))
    pre = {"ltv": _ltv(coll_val, debt_val), "hf": _hf(coll_val, debt_val, liq_thr)}

    action = seq[0]

    # ---------------- Oracle freshness guard ----------------
    age = _get(ctx, "oracle", "age_sec", default=None)
    max_age = _get(ctx, "oracle", "max_age_sec", default=None)
    if isinstance(age, (int, float)) and isinstance(max_age, (int, float)):
        if age > max_age:
            reason = _canon("oracle_stale", strict_tokens)
            _dprint("[defi_verify] rule=ORACLE_FRESHNESS →", reason, f"(age={age} > max={max_age})")
            return {"ok": False, "reason": reason}

    # ---------------- Borrow guard (HF) ----------------
    if any(step == "borrow_asset" for step in seq):
        # Prefer adapter's risk.hf if provided; fall back to pre-state HF
        hf = risk.get("hf")
        _dprint("[defi_verify] check=HF", " risk.hf=", hf, " pre.hf=", pre["hf"])
        try:
            hf_val = float(hf) if hf is not None else pre["hf"]
        except Exception:
            hf_val = pre["hf"]

        # Require healthy buffer above 1.0
        if hf_val < 1.10:
            reason = _canon("hf_too_low_for_borrow", strict_tokens)
            _dprint("[defi_verify] rule=HF_GUARD →", reason, f"(hf={hf_val} < 1.10)")
            return {"ok": False, "reason": reason}

    # ---------------- Withdraw / Remove-collateral guard (LTV) ----------------
    if any(step in ("withdraw_asset", "remove_collateral") for step in seq):
        # Prefer adapter's risk.ltv if provided; fall back to pre-state LTV
        ltv = risk.get("ltv")
        _dprint("[defi_verify] check=LTV", " risk.ltv=", ltv, " pre.ltv=", pre["ltv"], " ltv_max=", ltv_max)
        try:
            ltv_val = float(ltv) if ltv is not None else pre["ltv"]
        except Exception:
            ltv_val = pre["ltv"]

        # “Near max” margin (tune as needed)
        margin = 0.90
        if ltv_val > (margin * ltv_max):
            reason = _canon("ltv_near_max_for_withdraw", strict_tokens)
            _dprint("[defi_verify] rule=LTV_GUARD →", reason,
                    f"(ltv={ltv_val} > {margin}*{ltv_max})")
            return {"ok": False, "reason": reason}

    _dprint("[defi_verify] ✔ ok=True (no guard triggered)")
    return {"ok": True, "reason": ""}




# # micro_llm/verify/defi_verify.py
# from __future__ import annotations
# from typing import Dict, Any, List

# # -------------------- DEBUG BLOCK -------------------- #
# import os, sys, json

# def _dbg_on() -> bool:
#     # Set MICROLLM_DEBUG=1 to enable prints
#     return os.getenv("MICROLLM_DEBUG", "").strip() not in ("", "0", "false", "False")

# def _dprint(*args, **kwargs):
#     if _dbg_on():
#         print(*args, file=sys.stderr, **kwargs)

# def _snap(k, v):
#     try:
#         return json.dumps(v)
#     except Exception:
#         return repr(v)


# def _get(d: Dict[str, Any], *path, default=None):
#     x = d
#     for p in path:
#         if not isinstance(x, dict): return default
#         x = x.get(p)
#         if x is None: return default
#     return x

# # -------------------- DEBUG BLOCK -------------------- #

# def _ltv(collateral_value: float, debt_value: float) -> float:
#     return 0.0 if collateral_value <= 0 else debt_value / collateral_value

# def _hf(collateral_value: float, debt_value: float, liq_threshold: float) -> float:
#     # HF = (collateral_value * LT) / debt_value; if no debt -> inf
#     return float("inf") if debt_value <= 0 else (collateral_value * liq_threshold) / debt_value

# # -------------------- DEBUG BLOCK -------------------- #
# def defi_verify(plan: dict, ctx: dict) -> dict:
#     seq = (plan or {}).get("sequence") or []
#     pol = (ctx or {}).get("policy") or {}
#     risk = (ctx or {}).get("risk") or {}
#     oracle = (ctx or {}).get("oracle") or {}
#     ltv_max = pol.get("ltv_max")

#     _dprint("[defi_verify] ▶ enter",
#             " seq=", seq,
#             " ltv_max=", ltv_max,
#             " risk=", _snap("risk", risk),
#             " oracle=", _snap("oracle", oracle))

#     if not seq:
#         reason = "abstain_non_exec_or_empty"
#         # NOTE: If runner fast-paths policy blocks, it may already have set reason elsewhere.
#         # Here we’re marking the *verifier’s* own reason.
#         _dprint("[defi_verify] rule=EMPTY_PLAN →", reason)
#         return {"ok": False, "reason": reason}
    
# # -------------------- DEBUG BLOCK -------------------- #

# # def defi_verify(plan: Dict[str, List[str]], ctx: Dict[str, Any]) -> Dict[str, Any]:
# #     seq = plan.get("sequence", [])
#     # if not seq:
#     #     return {"ok": False, "reason": "abstain_non_exec_or_empty"}

#     # Pull policy knobs (with reasonable defaults)
#     liq_thr = float(_get(ctx, "policy", "liq_threshold", default=0.85))
#     ltv_max = float(_get(ctx, "policy", "ltv_max", default=0.75))

#     # Minimal current portfolio snapshot
#     coll_val = float(_get(ctx, "positions", "collateral_value", default=0.0))
#     debt_val = float(_get(ctx, "positions", "debt_value", default=0.0))

#     pre = {"ltv": _ltv(coll_val, debt_val), "hf": _hf(coll_val, debt_val, liq_thr)}

#     action = seq[0]

# # -------------------- DEBUG BLOCK -------------------- #
#     # Example: check if plan proposes a borrow
#     if any(step == "borrow_asset" for step in seq):
#         hf = risk.get("hf")
#         _dprint("[defi_verify] check=HF", " hf=", hf)
#         try:
#             hf_val = float(hf) if hf is not None else None
#         except Exception:
#             hf_val = None
#         if hf_val is not None and hf_val < 1.10:
#             reason = "hf_too_low_for_borrow"
#             _dprint("[defi_verify] rule=HF_GUARD →", reason)
#             return {"ok": False, "reason": reason}
# # -------------------- DEBUG BLOCK -------------------- #
    
#     # Borrow worsens HF/LTV; require healthy buffer
#     if action == "borrow_asset":
#         if pre["hf"] < 1.10:   # buffer above 1.0
#             return {"ok": False, "reason": "hf_too_low_for_borrow"}

# # -------------------- DEBUG BLOCK -------------------- #
#     if any(step in ("withdraw_asset", "remove_collateral") for step in seq):
#         ltv = risk.get("ltv")
#         _dprint("[defi_verify] check=LTV", " ltv=", ltv, " ltv_max=", ltv_max)
#         try:
#             ltv_val = float(ltv) if ltv is not None else None
#             ltv_max_val = float(ltv_max) if ltv_max is not None else None
#         except Exception:
#             ltv_val, ltv_max_val = None, None

#         # “Near max” margin (adjust if needed)
#         margin = 0.90
#         if (ltv_val is not None) and (ltv_max_val is not None) and (ltv_val > margin * ltv_max_val):
#             reason = "ltv_near_max_for_withdraw"
#             _dprint("[defi_verify] rule=LTV_GUARD →", reason,
#                     " (ltv=", ltv_val, ">", f"{margin}*{ltv_max_val})")
#             return {"ok": False, "reason": reason}
# # -------------------- DEBUG BLOCK -------------------- #


#     # Withdrawing / removing collateral tightens LTV; require margin
#     if action in ("withdraw_asset", "remove_collateral"):
#         if pre["ltv"] > (ltv_max * 0.90):
#             return {"ok": False, "reason": "ltv_near_max_for_withdraw"}

# # -------------------- DEBUG BLOCK -------------------- #
#     _dprint("[defi_verify] ✔ ok=True (no guard triggered)")
#     return {"ok": True, "reason": ""}
# # -------------------- DEBUG BLOCK -------------------- #
    
#     # Deposits, repays, add_collateral, and neutral swaps → OK in v0
#     # return {"ok": True, "reason": ""}