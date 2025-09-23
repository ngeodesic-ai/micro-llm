# tests/domains/defi/test_defi_smoke_cases.py
from micro_lm.core.runner import run_micro

def test_defi_case(case):
    out = run_micro("defi", case["prompt"], context={}, policy={}, rails="stage11", T=128)
    assert out.get("label") == case["label"] or (case["label"] == "abstain" and out["verdict"] == "ABSTAIN")
