import pytest
from micro_lm.mappers.joblib_mapper import JoblibMapper, JoblibMapperConfig

@pytest.mark.parametrize("text,expect_one_of", [
    ("deposit 10 ETH into aave", {"deposit_asset"}),
    ("swap 2 ETH for USDC", {"swap_asset"}),
])
def test_infer_top1_smoke(monkeypatch, text, expect_one_of):
    # Monkeypatch model load to avoid artifact dependency in stub tests.
    class DummyModel:
        classes_ = ["deposit_asset", "swap_asset", "withdraw_asset"]
        def predict_proba(self, X):
            t = X[0]
            if "deposit" in t: return [[0.9, 0.05, 0.05]]
            if "swap" in t:    return [[0.05, 0.9, 0.05]]
            return [[0.34, 0.33, 0.33]]

    import micro_lm.mappers.joblib_mapper as jm
    monkeypatch.setattr(jm, "load", lambda path: DummyModel())

    mapper = JoblibMapper(JoblibMapperConfig(model_path="dummy", confidence_threshold=0.7))
    res = mapper.infer(text)
    assert res.intent in expect_one_of
    assert res.score >= 0.7
    assert res.topk and res.topk[0][0] == res.intent
