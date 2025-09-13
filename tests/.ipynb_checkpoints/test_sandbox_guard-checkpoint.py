# tests/test_sandbox_guard.py
import pytest
from micro_llm.pipelines.runner import run_micro

def test_defi_sandbox_disables_stage11():
    """
    Ensure that when running in DeFi domain with sandbox policy
    (use_wdd=false, denoise=false), the Stage-11 machinery is bypassed.
    """

    policy = {
        "rails": {"use_wdd": False, "denoise": False},
        "ltv_max": 0.75,
        "mapper": {
            "model_path": ".artifacts/defi_mapper.joblib",
            "confidence_threshold": 0.7,
        },
    }
    context = {"oracle": {"age_sec": 5, "max_age_sec": 30}}

    res = run_micro(
        domain="defi",
        prompt="deposit 10 ETH into aave",
        context=context,
        policy=policy,
        rails="stage11",
        T=128,
    )

    # Sandbox guarantees
    assert isinstance(res, dict)
    # no Stage-11 report
    assert res.get("report", {}) == {}
    # aux schema does not contain a stage11 key
    assert "stage11" not in res.get("aux", {})

    # sanity: still produces a valid top1 plan
    plan_seq = res.get("plan", {}).get("sequence", [])
    assert plan_seq == [] or plan_seq[0] == "deposit_asset"
