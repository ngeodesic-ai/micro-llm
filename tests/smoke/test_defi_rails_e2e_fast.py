# Robust smoke harness: discover project root by searching upwards for known bench locations.
import json, subprocess, sys, pathlib, os

def discover_root(start: pathlib.Path) -> pathlib.Path:
    cur = start.resolve()
    candidates = [
        ["src", "micro_lm", "domains", "defi", "benches"],
        ["benches"]
    ]
    while True:
        for parts in candidates:
            if (cur.joinpath(*parts)).exists():
                return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # Fallback: original heuristic (two levels up from tests/smoke/*)
    return start.resolve().parents[2]

HERE = pathlib.Path(__file__).resolve().parent
ROOT = discover_root(HERE)

def find_bench(root: pathlib.Path, rel_src: str, rel_alt: str) -> pathlib.Path:
    p1 = root / rel_src
    p2 = root / rel_alt
    return p1 if p1.exists() else p2

def run_cli(path, args):
    cmd = [sys.executable, str(path)] + args
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT))

def test_defi_rails_bench_fast():
    bench = find_bench(ROOT, "src/micro_lm/domains/defi/benches/rails_bench.py", "benches/rails_bench.py")
    outdir = ROOT / ".artifacts" / "defi" / "rails_bench"
    res = run_cli(bench, ["--runs","1","--gate_min","0.0", "--out_dir", str(outdir)])
    print(res.stdout)
    assert res.returncode in (0,2), "CLI should run and return a gate code"
    assert (outdir / "summary.json").exists()
    J = json.loads((outdir / "summary.json").read_text())
    assert "ok_acc" in J and "total" in J

def test_defi_e2e_bench_fast():
    bench = find_bench(ROOT, "src/micro_lm/domains/defi/benches/e2e_bench.py", "benches/e2e_bench.py")
    outdir = ROOT / ".artifacts" / "defi" / "e2e_bench"
    res = run_cli(bench, ["--runs","1","--gate_min","0.0", "--out_dir", str(outdir)])
    print(res.stdout)
    assert res.returncode in (0,2,3), "CLI should run and return a gate code"
    assert (outdir / "summary.json").exists()
    J = json.loads((outdir / "summary.json").read_text())
    assert "ok_acc" in J and "total" in J
