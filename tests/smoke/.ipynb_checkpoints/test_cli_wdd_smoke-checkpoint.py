import subprocess, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def _run(path, args):
    return subprocess.run([sys.executable, str(ROOT / path)] + args,
                          text=True, capture_output=True)

def test_defi_rails_bench_wdd_smoke():
    bench = "src/micro_lm/domains/defi/benches/rails_bench.py"
    outdir = ROOT / ".artifacts" / "defi" / "rails_bench_wdd"
    res = _run(bench, ["--runs","1","--gate_min","0.0",
                       "--out_dir", str(outdir),
                       "--audit_backend","wdd"])
    assert res.returncode in (0,2,3), res.stdout + res.stderr
