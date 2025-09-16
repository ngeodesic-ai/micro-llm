import types
from micro_lm.core.rails_shim import Rails

def test_rails_calls_ngeodesic_when_available(monkeypatch):
    # Create a fake module tree: ngeodesic.stage11.verify_action
    fake_mod = types.SimpleNamespace()
    def fake_verify_action(**kwargs):
        assert kwargs["label"] == "deposit_asset"
        assert kwargs["rails"] == "stage11"
        return {"ok": True, "reason": "verified", "latency_ms": 5}

    fake_stage11 = types.SimpleNamespace(verify_action=fake_verify_action)
    fake_ng = types.SimpleNamespace(stage11=fake_stage11)

    monkeypatch.setitem(dict(globals()), "ngeodesic", fake_ng)  # not used; extra safety

    # Proper monkeypatch of import system
    import sys
    sys.modules["ngeodesic"] = types.ModuleType("ngeodesic")
    sys.modules["ngeodesic.stage11"] = types.ModuleType("ngeodesic.stage11")
    sys.modules["ngeodesic.stage11"].verify_action = fake_verify_action

    r = Rails(rails="stage11", T=180)
    out = r.verify(domain="defi", label="deposit_asset", context={}, policy={})
    assert out["ok"] is True
    assert out["reason"] == "verified"
    assert "latency_ms" in out
