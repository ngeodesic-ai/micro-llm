from pathlib import Path
import subprocess, sys, json

def test_defi_sweep_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    cases = root / "tests" / "fixtures" / "defi" / "cases_small.jsonl"
    out = tmp_path / "artifacts"
    cmd = [sys.executable, str(root/"src/micro_lm/benches/sweep.py"),
           "--domain","defi","--cases",str(cases),"--out_dir",str(out),
           "--mapper","wordmap","--audit_backend","wdd","--mode","pure","--T","128"]
    res = subprocess.run(cmd, text=True, capture_output=True)
    assert res.returncode == 0, res.stdout + res.stderr
    tsdir = next(out.iterdir())
    topline = json.loads((tsdir / "topline.json").read_text())
    assert "hit_rate" in topline and topline["n"] >= 1
