import numpy as np
from pathlib import Path
from micro_lm.core.runner import run_micro

def _write_prior(tmp_path: Path, mean, comps):
    p = tmp_path / "prior.npz"
    np.savez_compressed(
        p,
        mean=np.array(mean, dtype=np.float32),
        components=np.array(comps, dtype=np.float32),
    )
    return p

def test_audit_backend_pca_end_to_end(tmp_path):
    # Minimal viable prior (k=2, D=4)
    prior_path = _write_prior(
        tmp_path,
        mean=[0.0, 0.0, 0.0, 0.0],
        comps=[[1,0,0,0],[0,1,0,0]],
    )

    policy = {
        "audit": {
            "backend": "pca",
            "prior_path": str(prior_path),
            "mode": "pure",
        }
    }

    out = run_micro(
        domain="defi",
        prompt="deposit 1 ETH",
        context={},
        policy=policy,
        rails="stage11",
        T=128,
        backend="wordmap",
    )

    # We only assert the call succeeds and produces a sane response.
    assert isinstance(out, dict)
    assert "ok" in out and "label" in out and "reason" in out
