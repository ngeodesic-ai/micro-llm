import subprocess, pathlib, json

def run(cmd):
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout

def extract_json(s: str):
    import re, json
    m = None
    for m in re.finditer(r"\{.*\}", s, flags=re.DOTALL):
        pass
    assert m, f"No JSON in:\n{s}"
    return json.loads(m.group(0))

def test_m3_denoise_smoke():
    out = run(
        "python3 milestones/defi_milestone3.py "
        "--rails stage11 --runs 2 "
        "--policy '{\"mapper\":{\"model_path\": \".artifacts/defi_mapper.joblib\", \"confidence_threshold\":0.7}, \"ltv_max\":0.75}' "
        "--context '{\"oracle\":{\"age_sec\":5,\"max_age_sec\":30}}'"
    )
    top = extract_json(out)
    assert top.get("ok") is True, top

    p = pathlib.Path(top["summary"])
    data = json.loads(p.read_text())
    assert data["status"] == "pass"
    for sc in data["scenarios"]:
        assert sc["ok"], f"{sc['name']} failed: {sc['reason']}"
