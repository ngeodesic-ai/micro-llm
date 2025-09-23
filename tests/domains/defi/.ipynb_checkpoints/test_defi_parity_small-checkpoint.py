import numpy as np
import pytest
from micro_lm.domains.defi.audit_wdd import wdd_defi_audit

# Skip if legacy Tier-1 isn't present on this branch
def _has_t1():
    try:
        from micro_lm.domains.defi.audit import defi_audit as _t1
        return _t1
    except Exception:
        return None

@pytest.mark.parity
def test_small_parity_topline():
    t1 = _has_t1()
    if t1 is None:
        pytest.skip("Tier-1 audit not available")

    D=12; K=8; T=180
    P = np.zeros((K,D), dtype=np.float32); 
    for k in range(K): P[k, k%D] = 1.0
    A = np.zeros((K,D), dtype=np.float32)
    emb = np.zeros(D, dtype=np.float32) + 0.2

    # Tier-1
    t1_out = t1(emb, P, A, {"T":T,"seed":0})
    # WDD
    wdd_out = wdd_defi_audit(emb, P, A, {"mode":"family","T":T,"seed":0})

    # Parity gate: at least one pick; overlapping classes if any
    assert len(wdd_out["keep"]) >= 1
    if len(t1_out.get("keep", [])) > 0:
        overlap = set(t1_out["keep"]) & set(wdd_out["keep"])
        assert len(overlap) >= 1
