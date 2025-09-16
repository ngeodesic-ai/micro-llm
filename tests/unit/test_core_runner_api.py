import inspect
from micro_lm.core.runner import run_micro

def test_run_micro_signature_frozen():
    sig = str(inspect.signature(run_micro))
    assert sig == "(domain: str, prompt: str, *, context: dict, policy: dict, rails: str, T: int, backend: str = 'sbert') -> dict"

def test_run_micro_smoke_no_rails_abstain():
    out = run_micro(
        "defi",
        "",
        context={},
        policy={"mapper": {"confidence_threshold": 0.5}},
        rails="",
        T=180,
        backend="wordmap",
    )
    assert out["ok"] is False and out["label"] == "abstain"

def test_run_micro_smoke_with_rails_wordmap_hit():
    out = run_micro(
        "defi",
        "please deposit 1 ETH",
        context={},
        policy={"mapper": {"confidence_threshold": 0.5}},
        rails="stage11",
        T=180,
        backend="wordmap",
    )
    assert out["ok"] is True
    assert out["label"] == "deposit_asset"
    assert "artifacts" in out
