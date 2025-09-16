import json, subprocess, sys, pathlib

def test_m10_smoke():
    summary = pathlib.Path(".artifacts/defi_milestone10_summary.json")
    # 1 fast run, 1 perturb variant to keep it quick
    cmd = [
        sys.executable, "milestones/defi_milestone10.py",
        "--rails", "stage11", "--runs", "1", "--perturb", "--perturb_k", "1",
        "--policy", '{"ltv_max":0.75,"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7}}',
        "--context", '{"oracle":{"age_sec":5,"max_age_sec":30}}'
    ]
    subprocess.run(cmd, check=True)
    J = json.loads(summary.read_text())
    assert J["status"] == "pass", J.get("failures")
    # sanity: each base scenario present and ok
    names = {s["name"]: s["ok"] for s in J["scenarios"] if not s["name"].endswith("_perturb")}
    for k in ["deposit_eth","swap_eth_usdc","withdraw_high_ltv","borrow_low_hf","nonexec_abstain"]:
        assert names.get(k) is True
