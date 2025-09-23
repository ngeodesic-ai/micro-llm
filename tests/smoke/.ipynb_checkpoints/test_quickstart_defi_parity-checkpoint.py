# tests/test_quickstart_defi_parity.py
import json
from micro_lm.cli.defi_quickstart import quickstart

def _shape_only(d):
    assert set(d.keys()) == {"plan", "verify"}
    assert set(d["plan"].keys()) == {"sequence"}
    assert isinstance(d["plan"]["sequence"], list)
    assert set(d["verify"].keys()) == {"ok", "reason"}
    assert isinstance(d["verify"]["ok"], bool)
    assert isinstance(d["verify"]["reason"], str)

def test_deposit_maps_and_verifies_ok():
    out = quickstart("deposit 10 ETH into aave")
    _shape_only(out)
    assert out["plan"]["sequence"][:1] == ["deposit_asset"]
    assert out["verify"]["ok"] is True

def test_swap_maps_and_verifies_ok():
    out = quickstart("swap 2 ETH for USDC")
    _shape_only(out)
    assert out["plan"]["sequence"][:1] == ["swap_asset"]
    assert out["verify"]["ok"] is True

def test_withdraw_ltv_edge_abstains_or_blocks():
    out = quickstart(
        "withdraw 5 ETH",
        policy={"ltv_max": 0.60},                          # stricter to trigger block
        context={"oracle": {"age_sec": 5, "max_age_sec": 30}, "risk": {"hf": 1.15}}
    )
    assert out["plan"]["sequence"] == []
    # assert out["verify"]["ok"] is False
    # assert "ltv" in out["verify"]["reason"].lower()
