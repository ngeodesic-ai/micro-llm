def test_defi_stage11_enabled_inserts_report():
    """
    Sanity check: if sandbox is OFF (use_wdd=true), Stage-11
    should produce a non-empty report or aux.stage11 key.
    This confirms both branches (sandbox vs non-sandbox) work.
    """

    policy = {
        "rails": {"use_wdd": True, "denoise": False},
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

    assert isinstance(res, dict)

    # Expect Stage-11 machinery to leave a trace somewhere:
    has_stage11_aux = "stage11" in res.get("aux", {})
    has_report_keys = bool(res.get("report"))
    assert has_stage11_aux or has_report_keys, (
        "Stage-11 enabled but no stage11 report or aux key present"
    )
