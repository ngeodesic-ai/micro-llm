import types
import micro_lm.core.mapper_api as mapper_api

def test_mapper_degrades_to_wordmap(monkeypatch):
    # Force SBERT init to raise
    class DummySBert:
        def __init__(self, *a, **k):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(mapper_api, "sbert", types.SimpleNamespace(SBertMapper=DummySBert), raising=True)
    api = mapper_api.MapperAPI(backend="sbert", domain="defi", policy={})
    assert api.backend == "wordmap"
    assert getattr(api, "debug", {}).get("degraded") is True
