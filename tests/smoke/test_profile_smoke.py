import json
from pathlib import Path
from micro_lm.core.runner import run_micro

def test_profile_smoke(tmp_path: Path):
    # enable profiling via policy (no env var required)
    policy = {"audit": {"backend":"wdd", "mode":"pure", "profile": True}}
    out = run_micro("defi", "deposit 1 ETH", context={}, policy=policy, rails="stage11", T=128, backend="wordmap")
    # find the newest profile dir
    prof_root = Path(".artifacts") / "defi" / "profile"
    assert prof_root.exists()
    ts_dirs = sorted([p for p in prof_root.iterdir() if p.is_dir()])
    assert ts_dirs, "no profile dir created"
    latest = ts_dirs[-1]
    prof = json.loads((latest / "profile.json").read_text())
    top = json.loads((latest / "topline.json").read_text())
    assert isinstance(prof, list) and any(x["phase"] == "parse" for x in prof)
    assert "latency_ms" in top and "n_keep" in top
