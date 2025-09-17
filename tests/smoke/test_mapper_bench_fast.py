# Robust smoke test for Stage-7 mapper bench
import json, subprocess, sys, pathlib, os

def discover_root(start: pathlib.Path) -> pathlib.Path:
    cur = start.resolve()
    while True:
        if (cur/"src/micro_lm/domains/defi/benches/mapper_bench.py").exists() or (cur/"benches/mapper_bench.py").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve().parents[2]

HERE = pathlib.Path(__file__).resolve().parent
ROOT = discover_root(HERE)

def find_bench() -> pathlib.Path:
    p1 = ROOT/"src/micro_lm/domains/defi/benches/mapper_bench.py"
    p2 = ROOT/"benches/mapper_bench.py"
    return p1 if p1.exists() else p2

def run_cli(args):
    bench = find_bench()
    return subprocess.run([sys.executable, str(bench)] + args,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT))

def test_mapper_bench_wordmap_fast():
    outdir = ROOT / ".artifacts" / "defi" / "mapper_bench"
    res = run_cli(["--backend","wordmap","--thresholds","0.3,0.35","--out_dir", str(outdir)])
    print(res.stdout)
    assert res.returncode in (0,2)
    assert (outdir / "summary.json").exists()
    J = json.loads((outdir / "summary.json").read_text())
    assert "chosen" in J

def test_mapper_bench_sbert_graceful():
    outdir = ROOT / ".artifacts" / "defi" / "mapper_bench"
    res = run_cli(["--backend","sbert","--thresholds","0.7","--out_dir", str(outdir)])
    print(res.stdout)
    assert res.returncode in (0,2)
    assert (outdir / "summary.json").exists()
