import json, pathlib, subprocess, sys

def run(cmd):
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout

def test_m2_smoke_stable_top1():
    # run with fewer repeats to keep CI fast
    run('python3 milestones/defi_milestone2.py --rails stage11 --runs 2 '
        '--policy \'{"mapper":{"model_path":".artifacts/defi_mapper.joblib","confidence_threshold":0.7},"rails":{"denoise": false},"ltv_max":0.75}\' '
        '--context \'{"oracle":{"age_sec":5,"max_age_sec":30}}\'')
    p = pathlib.Path(".artifacts/defi_milestone2_summary.json")
    data = json.loads(p.read_text())
    assert data["status"] == "pass"
    for sc in data["scenarios"]:
        assert sc["ok"], sc["reason"]
