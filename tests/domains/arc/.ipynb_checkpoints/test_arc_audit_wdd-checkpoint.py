import numpy as np
from micro_lm.domains.arc.audit_wdd import wdd_arc_audit

def test_arc_wrapper_basic():
    D=9; K=6; T=240
    P = np.zeros((K,D), dtype=np.float32)
    for k in range(K): P[k, k%D] = 1.0
    A = np.zeros((K,D), dtype=np.float32)
    emb = np.zeros(D, dtype=np.float32) + 0.3
    out = wdd_arc_audit(emb, P, A, {"mode":"family","T":T,"seed":0})
    assert set(out["keep"]) <= set(range(K))
    assert out["order"] == sorted(out["keep"], key=lambda k: out["debug"]["tstars"][k] if "tstars" in out["debug"] else k)