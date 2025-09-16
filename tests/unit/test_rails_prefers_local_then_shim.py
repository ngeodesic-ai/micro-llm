import os
from micro_lm.core.rails_shim import Rails

def test_rails_uses_local_when_no_ngeodesic(monkeypatch):
    import sys
    for m in ("ngeodesic.stage11", "ngeodesic"):
        sys.modules.pop(m, None)  # ensure import fails regardless of env
    monkeypatch.delenv("MICROLM_STRICT_SHIM", raising=False)

    r = Rails(rails="stage11", T=180)
    out = r.verify(domain="defi", label="deposit_asset",
                   context={"oracle":{"age_sec":5,"max_age_sec":30}},
                   policy={})
    assert out["ok"] is True
    assert out["reason"] in {"local:verified", "shim:accept:stage-4"}

def test_rails_respects_forced_shim(monkeypatch):
    monkeypatch.setenv("MICROLM_STRICT_SHIM", "1")
    r = Rails(rails="stage11", T=180)
    out = r.verify(domain="defi", label="deposit_asset", context={}, policy={})
    assert out["reason"] == "shim:forced"
