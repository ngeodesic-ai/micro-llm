# tests/test_defi_verify.py
from micro_lm.verify.defi_verify import defi_verify

def test_borrow_abstains_when_hf_low():
    plan = {"sequence": ["borrow_asset"]}
    ctx = {"positions": {"collateral_value": 1000.0, "debt_value": 900.0},
           "policy": {"liq_threshold": 0.85}}
    out = defi_verify(plan, ctx)
    assert out["ok"] is False and "hf_too_low" in out["reason"]

def test_withdraw_abstains_when_ltv_high():
    plan = {"sequence": ["withdraw_asset"]}
    ctx = {"positions": {"collateral_value": 1000.0, "debt_value": 740.0},
           "policy": {"ltv_max": 0.75}}
    out = defi_verify(plan, ctx)
    assert out["ok"] is False and "ltv_near_max" in out["reason"]

def test_deposit_is_ok():
    plan = {"sequence": ["deposit_asset"]}
    ctx = {"positions": {"collateral_value": 1000.0, "debt_value": 700.0},
           "policy": {"ltv_max": 0.75}}
    out = defi_verify(plan, ctx)
    assert out["ok"] is True
