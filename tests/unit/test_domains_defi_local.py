from micro_lm.domains.defi.verify_local import verify_action_local

def test_defi_oracle_stale_blocks():
    out = verify_action_local(label="deposit_asset",
                              context={"oracle":{"age_sec":31,"max_age_sec":30}},
                              policy={})
    assert out["ok"] is False and out["reason"] == "oracle_stale"

def test_defi_ltv_blocks_withdraw():
    out = verify_action_local(label="withdraw_asset",
                              context={"oracle":{"age_sec":5,"max_age_sec":30},"ltv":0.8},
                              policy={"ltv_max":0.75})
    assert out["ok"] is False and out["reason"] == "ltv"

def test_defi_ok_deposit_when_fresh():
    out = verify_action_local(label="deposit_asset",
                              context={"oracle":{"age_sec":5,"max_age_sec":30}},
                              policy={})
    assert out["ok"] is True
