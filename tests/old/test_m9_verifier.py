# tests/test_m9_verifier.py
import json, pathlib
def test_m9_guards():
    S = json.loads(pathlib.Path(".artifacts/defi_milestone9_summary.json").read_text())
    m = S["metrics"]
    assert m["edge_coverage"] == 1.0
    assert m["false_approvals"] == 0
    assert m["exec_accuracy"] >= 0.90