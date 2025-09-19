
import os, json, subprocess, sys, shutil, pytest, pathlib

FIX_PROMPTS = "tests/fixtures/defi/defi_mapper_5k_prompts.jsonl"
FIX_LABELS  = "tests/fixtures/defi/defi_mapper_labeled_5k.csv"

@pytest.mark.smoke
def test_m10_audit_smoke(tmp_path):
    # Skip if fixtures aren't present in the repo (CI should add them)
    if not (pathlib.Path(FIX_PROMPTS).exists() and pathlib.Path(FIX_LABELS).exists()):
        pytest.skip("fixtures not found; skipping smoke test")
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable, "scripts/milestones/defi_milestone10_audit.py",
        "--prompts_jsonl", FIX_PROMPTS,
        "--labels_csv", FIX_LABELS,
        "--sbert", "sentence-transformers/all-MiniLM-L6-v2",
        "--n_max", "4", "--tau_span", "0.50", "--tau_rel", "0.60", "--tau_abs", "0.93",
        "--L", "160", "--beta", "8.6", "--sigma", "0.0",
        "--out_dir", str(out_dir),
        "--competitive_eval",
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([env.get('PYTHONPATH',''), 'src', '.'])
    # keep output quiet in CI
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
    assert res.returncode == 0, res.stderr

    # Check expected files
    rows = out_dir / "rows_audit.csv"
    metrics = out_dir / "metrics_audit.json"
    summary = out_dir / "defi_milestone10_audit_summary.json"
    report = out_dir / "defi_milestone10_audit_report.md"
    for p in [rows, metrics, summary, report]:
        assert p.exists(), f"missing artifact: {p}"

    # Minimal metrics sanity
    M = json.loads(metrics.read_text())
    assert "coverage" in M and "abstain_rate" in M
    P = M.get("params", {})
    assert P.get("tau_rel") == 0.60 and P.get("tau_abs") == 0.93
