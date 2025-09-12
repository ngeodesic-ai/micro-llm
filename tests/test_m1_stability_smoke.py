# tests/test_m1_stability_smoke.py
import subprocess
import pathlib
import json
import re

def run_cmd(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, shell=True, check=True,
        capture_output=True, text=True
    )

def _extract_json_blob(s: str) -> dict:
    """
    Be tolerant of extra prints/newlines: grab the last {...} blob from stdout.
    """
    m = None
    for m in re.finditer(r"\{.*\}", s, flags=re.DOTALL):
        pass
    if not m:
        raise AssertionError(f"Could not find JSON in stdout:\n---STDOUT---\n{s}\n--------------")
    return json.loads(m.group(0))

def test_m1_stability_smoke():
    # 1) Run milestone1
    proc = run_cmd(
        "python3 milestones/defi_milestone1.py "
        "--rails stage11 "
        "--policy '{\"mapper\":{\"model_path\": \".artifacts/defi_mapper.joblib\", "
        "\"confidence_threshold\":0.7}, \"ltv_max\":0.75}'"
    )

    # 2) Parse the runner’s stdout JSON (tolerant to extra lines)
    top = _extract_json_blob(proc.stdout)
    assert top.get("ok") is True, f"Milestone1 runner returned not ok: {top}"
    assert "summary" in top, f"No 'summary' path in runner output: {top}"

    # 3) Load the summary and assert pass
    summary_path = pathlib.Path(top["summary"])
    assert summary_path.exists(), f"Summary file not found: {summary_path}"
    data = json.loads(summary_path.read_text())
    assert data.get("status") == "pass", f"Summary not pass: {data}"

    # 4) All scenarios ok
    scenarios = data.get("scenarios", [])
    assert scenarios, "Summary has no scenarios"
    for sc in scenarios:
        assert sc.get("ok") is True, f"Scenario failed: {sc.get('name')} reason={sc.get('reason')}"

    # 5) Sanity on expected behaviors if present
    by_name = {sc.get("name"): sc for sc in scenarios}
    if "deposit_eth" in by_name:
        out_deposit = by_name["deposit_eth"]["output"]
        assert out_deposit.get("top1") == "deposit_asset", f"deposit_eth top1 unexpected: {out_deposit.get('top1')}"
    if "nonexec_abstain" in by_name:
        out_ne = by_name["nonexec_abstain"]["output"]
        assert out_ne.get("top1") is None, "nonexec_abstain should have empty plan"
        flags = out_ne.get("flags", {})
        assert flags.get("abstain_non_exec") is True, f"nonexec_abstain missing flag: {flags}"
