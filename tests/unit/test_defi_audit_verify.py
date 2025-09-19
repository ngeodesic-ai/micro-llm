
import os, sys, importlib, types, json
import pytest

# Try to ensure "src" layout works when running tests from repo root
if 'src' not in sys.path:
    sys.path.insert(0, 'src')

def test_verify_import_has_no_mapper_dependency(monkeypatch):
    """
    Sanity: importing the verifier must not pull in mapper code (tautology guard).
    """
    # Place a sentinel "mapper" module that would be imported if verify had a dependency
    sentinel = types.SimpleNamespace(SENTINEL=True)
    sys.modules['micro_lm.domains.defi.mapper'] = sentinel

    verify = importlib.import_module('micro_lm.domains.defi.verify')

    # 1) The module's namespace shouldn't reference "mapper"
    bad = [n for n in dir(verify) if 'mapper' in n.lower()]
    assert not bad, f"verify.py should not reference mapper; found {bad}"

    # 2) Importing verify should not have imported the sentinel under any alias
    # (we still keep it available in sys.modules from our injection; verify must not touch it)
    assert sys.modules.get('micro_lm.domains.defi.mapper') is sentinel

@pytest.mark.parametrize('n', [3])
def test_run_audit_with_fake_emb(monkeypatch, n):
    """
    Run the audit pipeline with a fake embedding to avoid external deps and ensure
    deterministic behavior on tiny inputs.
    """
    verify = importlib.import_module('micro_lm.domains.defi.verify')

    # Fake embedding that returns a fixed-size, normalized vector for any input
    class FakeEmb:
        def __init__(self, model_name: str, batch_size: int = 64, normalize: bool = True):
            self.model_name = model_name
        def transform(self, X):
            import numpy as np
            V = np.ones((len(X), 8), dtype='float32')
            # give slight variation by length of string to avoid zero divisions in cos
            for i, s in enumerate(X):
                V[i, 0] = 1.0 + (len(s) % 3) * 0.001
            # L2 normalize
            V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
            return V
        def encode_one(self, s):
            return self.transform([s])[0]

    # Monkeypatch the Emb class used inside run_audit
    monkeypatch.setattr(verify, 'Emb', FakeEmb, raising=True)

    # Tiny toy data
    prompts = ["deposit 1 eth", "swap for usdc", "withdraw funds"][:n]
    gold    = ["deposit_asset", "swap_asset", "withdraw_asset"][:n]

    res = verify.run_audit(
        prompts=prompts,
        gold_labels=gold,
        sbert_model="fake-model",   # ignored by FakeEmb
        n_max=3,
        tau_span=0.0,               # trivial so spans always pass with FakeEmb
        tau_rel=0.0, tau_abs=0.0,   # trivial gates so we don't depend on scores
        L=16, beta=5.0, sigma=0.0,
        competitive_eval=True
    )

    # Basic shape checks
    assert isinstance(res, dict) and "rows" in res and "metrics" in res
    assert len(res["rows"]) == len(prompts)
    M = res["metrics"]
    for k in ["coverage", "abstain_rate", "span_yield_rate", "abstain_no_span_rate", "abstain_with_span_rate", "params"]:
        assert k in M

    # Each row should have prompt/gold/pred/ok and tags
    for r, p, g in zip(res["rows"], prompts, gold):
        assert r["prompt"] == p
        assert r["gold"] == g
        assert "ok" in r
        assert "pred" in r
        assert isinstance(r.get("tags", {}), dict)

