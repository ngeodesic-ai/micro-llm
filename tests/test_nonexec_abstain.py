# tests/test_nonexec_abstain.py
from micro_lm.pipelines.runner import run_micro

def test_check_balance_abstains():
    res = run_micro(
        domain="defi",
        prompt="check balance",
        context={"risk":{"hf":1.3}, "oracle":{"age_sec":5,"max_age_sec":30}},
        policy={"ltv_max":0.75},
        rails="stage11",
        T=128
    )
    assert res["plan"]["sequence"] == []
    assert res["verify"]["ok"] is False
    assert res["verify"]["reason"] == "abstain_non_exec"
