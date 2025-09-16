# tests/test_m5_regression.py
import json, pathlib, subprocess, sys

def run_cmd(cmd: str):
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    assert proc.returncode in (0,2), proc.stderr  # 2 used if script decides to fail
    return proc

def test_m5_regression():
    proc = run_cmd(
        "python3 milestones/defi_milestone5.py "
        "--rails stage11 "
        "--runs 5 "
        "--policy '{\"ltv_max\":0.75, \"mapper\":{\"model_path\": \".artifacts/defi_mapper.joblib\", \"confidence_threshold\":0.7}}' "
        "--context '{\"oracle\":{\"age_sec\":5,\"max_age_sec\":30}}'"
    )
    last = proc.stdout.strip().splitlines()[-1]
    top = json.loads(last)
    assert top.get("ok") is True, f"Milestone5 failed top-level ok: {top}"
    summary = pathlib.Path(top["summary"])
    assert summary.exists(), "Milestone5 summary file missing"
    data = json.loads(summary.read_text())
    assert data.get("status") == "pass", f"Milestone5 status not pass: {data.get('status')}"
    # sanity: ensure our three categories exist
    names = [s["name"] for s in data["scenarios"]]
    for must in ["deposit_eth", "swap_eth_usdc", "nonexec_abstain"]:
        assert must in names, f"missing scenario {must}"
