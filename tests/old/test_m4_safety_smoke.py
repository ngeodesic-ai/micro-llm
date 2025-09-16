# tests/test_m4_safety_smoke.py
import json, subprocess, shlex, pathlib

def run(cmd):
    p = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip().splitlines()[-1]

def test_m4_safety_smoke():
    _ = run(
      "python3 milestones/defi_milestone4.py "
      "--rails stage11 --runs 2 "
      "--policy '{\"ltv_max\":0.75}' "
      "--context '{\"risk\":{\"hf\":0.9},\"oracle\":{\"age_sec\":5,\"max_age_sec\":30}}'"
    )
    path = pathlib.Path(".artifacts/defi_milestone4_summary.json")
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["status"] == "pass"
    # Withdraw and borrow scenarios in m4 should top1==None (abstain)
    sc = {s["name"]: s for s in data["scenarios"]}
    assert sc["withdraw_high_ltv"]["ok"] is True
    assert sc["borrow_low_hf"]["ok"] is True