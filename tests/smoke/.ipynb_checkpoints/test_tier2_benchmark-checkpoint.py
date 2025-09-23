# import json
# from pathlib import Path
# import subprocess
# import sys

# def test_tier2_harness_smoke(tmp_path: Path):
#     outdir = tmp_path / "tier2_out"
#     cmd = [
#         sys.executable, "scripts/tier2_benchmark.py",
#         "--domains", "defi,arc",
#         "--backend", "wordmap",
#         "--rails", "stage11",
#         "--audit-backend", "wdd",
#         "--det_runs", "2",
#         "--perturb_k", "1",
#         "--outdir", str(outdir),
#     ]
#     rc = subprocess.run(cmd, check=False, capture_output=True, text=True)
#     assert rc.returncode in (0, 2), rc.stderr  # allow threshold fail to still write artifacts

#     js = outdir / "tier2_summary.json"
#     md = outdir / "tier2_report.md"
#     assert js.exists() and md.exists(), "summary/report not written"

#     summary = json.loads(js.read_text())
#     assert "domains" in summary and isinstance(summary["domains"], list)
#     # Each domain must have cases and accuracy field
#     for dom in summary["domains"]:
#         assert "cases" in dom and "accuracy" in dom
#         assert all("out" in c and "latency_ms" in c for c in dom["cases"])
