# tests/test_cli_defi_examples.py
import json, subprocess, shlex

def run(cmd):
    p = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    out = p.stdout.strip()
    return json.loads(out)

def test_cli_deposit_stage11():
    res = run('micro-defi --prompt "deposit 10 ETH into aave" --rails stage11')
    assert res["plan"]["sequence"] == ["deposit_asset"]
    assert res["verify"]["ok"] is True
    # Optional if you wired mapper_confidence:
    # assert isinstance(res["aux"].get("mapper_confidence"), (int, float))

def test_cli_swap_stage11():
    res = run('micro-defi --prompt "swap 2 ETH for USDC" --rails stage11')
    assert res["plan"]["sequence"] == ["swap_asset"]
    assert res["verify"]["ok"] is True
