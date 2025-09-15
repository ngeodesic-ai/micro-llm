# tests/test_m6_mapper_wire.py
import json, subprocess, sys

def test_mapper_threshold(tmp_path):
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    # Train a tiny, deterministic text -> primitive classifier
    X = [
        "deposit 10 eth into aave",
        "deposit funds",
        "withdraw 200 usdc from compound",
        "please withdraw dai",
        "swap 2 eth to usdc",
        "swap usdc to wbtc",
        "check balance",
        "check my wallet balance",
    ]
    y = [
        "deposit_asset",
        "deposit_asset",
        "withdraw_asset",
        "withdraw_asset",
        "swap_asset",
        "swap_asset",
        "check_balance",
        "check_balance",
    ]
    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1,2), min_df=1),
        LogisticRegression(max_iter=4000, random_state=0, C=10.0)
    )
    clf.fit(X, y)

    mp = tmp_path / "mapper.joblib"
    joblib.dump(clf, mp)

    # Case 1: high threshold => both prompts should ABSTAIN
    prompts1 = tmp_path / "prompts1.jsonl"
    prompts1.write_text('{"prompt":"swap 2 ETH"}\n{"prompt":"check balance"}\n')
    summ1 = tmp_path / "sum1.json"
    cmd1 = [
        sys.executable, "milestones/defi_milestone6.py",
        "--mapper_path", str(mp),
        "--runs", "2",
        "--prompts_jsonl", str(prompts1),
        "--confidence_threshold", "0.95",
        "--out_summary", str(summ1),
    ]
    p1 = subprocess.run(cmd1, capture_output=True, text=True)
    assert summ1.exists(), f"summary not written\nstdout:\n{p1.stdout}\nstderr:\n{p1.stderr}"
    S1 = json.loads(summ1.read_text())
    assert S1["abstain_count"] == 2

    # Case 2: lower threshold => “deposit” should FIRE (no abstain)
    prompts2 = tmp_path / "prompts2.jsonl"
    prompts2.write_text('{"prompt":"deposit 10 ETH into aave"}\n')
    summ2 = tmp_path / "sum2.json"
    cmd2 = [
        sys.executable, "milestones/defi_milestone6.py",
        "--mapper_path", str(mp),
        "--runs", "1",
        "--prompts_jsonl", str(prompts2),
        "--confidence_threshold", "0.50",
        "--out_summary", str(summ2),
    ]
    p2 = subprocess.run(cmd2, capture_output=True, text=True)
    assert summ2.exists(), f"summary not written\nstdout:\n{p2.stdout}\nstderr:\n{p2.stderr}"
    S2 = json.loads(summ2.read_text())
    assert S2["abstain_count"] == 0
    # Also verify the mapped primitive is deposit
    seq = S2["rows"][0]["plan"]["sequence"]
    assert seq == ["deposit_asset"], f"expected deposit_asset, got {seq}"
