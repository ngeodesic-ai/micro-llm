import types
import json
import builtins
import importlib

import pytest


"""
pytest -q tests/test_m11_unit.py
"""

# We import the module under test. In the repo it should be milestones.defi_milestone11.
# For local runs where the file is at project root or /mnt/data, this fallback import helps:
M = None
try:
    M = importlib.import_module("milestones.defi_milestone11")
except Exception:
    # fallback to path-based import
    import importlib.util, sys, pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "defi_milestone11.py"
    spec = importlib.util.spec_from_file_location("defi_milestone11", str(p))
    M = importlib.util.module_from_spec(spec)
    sys.modules["defi_milestone11"] = M
    spec.loader.exec_module(M)


def test_decision_from_verify():
    assert M.decision_from_verify({"verify": {"ok": True}}) == "approve"
    assert M.decision_from_verify({"verify": {"ok": False}}) == "reject"
    assert M.decision_from_verify({"verify": {}}) == "reject"


def test_extract_reason_tokens():
    out = {"verify": {"reason": "LTV_BREACH", "tags": ["Risk", "LTV"]}, "flags": {"oracle_stale": True}}
    s = M.extract_reason_tokens({"verify": out["verify"], "flags": out["flags"]})
    s = s.lower()
    # Should include normalized tokens from reason, tags, and flags keys
    assert "ltv" in s
    assert "risk" in s
    assert "oracle_stale" in s or "oracle:true" in s


def test_consolidate_metrics_basic():
    # Two exec approvals (TP), one edge correctly rejected (TN), one edge incorrectly approved (FP).
    recs = [
        {"truth":"approve", "decision":"approve", "output":{"verify":{"reason":""}}},  # TP
        {"truth":"approve", "decision":"approve", "output":{"verify":{"reason":""}}},  # TP
        {"truth":"reject" , "decision":"reject" , "output":{"verify":{"reason":"oracle_stale"}}},  # TN
        {"truth":"reject" , "decision":"approve", "output":{"verify":{"reason":"ltv_breach"}}},    # FP
    ]
    m = M.consolidate_metrics(recs)
    # precision = TP/(TP+FP) = 2/3
    assert pytest.approx(m["precision"], rel=1e-6) == 2/3
    # recall = TP/(TP+FN) = 2/2 = 1.0
    assert m["recall"] == 1.0
    # hallucination_rate = FP / (TP+FP) = 1/3
    assert pytest.approx(m["hallucination_rate"], rel=1e-6) == 1/3
    # omission_rate = FN/(TP+FN) = 0
    assert m["omission_rate"] == 0.0


def test_fallback_priority_ltv_over_oracle(monkeypatch):
    # Build a synthetic result where both LTV and oracle flags are present
    result = {
        "plan": {"sequence": []},
        "flags": {"ltv_breach": True, "oracle_stale": True},
    }
    vb = M._fallback_verify(result)  # uses M11's priority logic
    assert vb["ok"] is False
    assert "ltv" in (vb.get("reason") or "")


def test_run_suite_with_monkeypatched_run_once(monkeypatch):
    # Prepare a minimal edge and exec set
    edges = [{
        "name": "edge_ltv_withdraw_unsafe",
        "prompt": "withdraw 5 ETH",
        "expect_top1": None,
        "expect_verify_ok": False,
        "expect_reason_any": ["ltv","oracle","stale","abstain_non_exec"],
    }]
    exec_ok = [{
        "name": "ok_deposit",
        "prompt": "deposit 10 ETH into aave",
        "expect_top1": "deposit_asset",
        "expect_verify_ok": True,
    }]

    # Stub run_once to simulate deterministic behavior
    def fake_run_once(prompt, context, policy, rails, T):
        if "withdraw 5 ETH" in prompt:
            return {
                "prompt": prompt,
                "top1": None,
                "flags": {"ltv_breach": True},
                "verify": {"ok": False, "reason": "ltv_breach", "tags": ["ltv"]},
                "plan": {"sequence": []},
                "aux": {},
            }
        if "deposit 10 ETH" in prompt:
            return {
                "prompt": prompt,
                "top1": "deposit_asset",
                "flags": {},
                "verify": {"ok": True, "reason": ""},
                "plan": {"sequence": ["deposit_asset"]},
                "aux": {},
            }
        raise AssertionError("unexpected prompt")

    monkeypatch.setattr(M, "run_once", fake_run_once)

    ctx = {"oracle": {"age_sec": 5, "max_age_sec": 30}}
    pol = {"ltv_max": 0.75, "hf_min": 1.0, "mapper": {"model_path": ".artifacts/defi_mapper.joblib", "confidence_threshold": 0.7}}

    scenarios, failures = M.run_suite("stage11", 180, 2, ctx, pol, edges, exec_ok)

    # Should have no failures, one edge reject + one exec approve
    assert failures == []
    # decisions:
    d0 = scenarios[0]["decision"]
    d1 = scenarios[1]["decision"]
    assert d0 == "reject" and d1 == "approve"
    # metrics consistency check
    metrics = M.consolidate_metrics([
        {"truth": scenarios[0]["truth"], "decision": scenarios[0]["decision"], "output": scenarios[0]["output"]},
        {"truth": scenarios[1]["truth"], "decision": scenarios[1]["decision"], "output": scenarios[1]["output"]},
    ])
    assert metrics["hallucination_rate"] == 0.0
    assert metrics["omission_rate"] == 0.0
