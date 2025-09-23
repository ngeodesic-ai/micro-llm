import numpy as np
from micro_lm.core.audit import AuditRequest
from micro_lm.core.audit.orchestrator import run_wdd

def test_traces_envelope_shapes_and_ordering():
    D=9; K=3; T=300
    P = np.zeros((K,D), dtype=np.float32); P[0,0]=1; P[1,1]=1; P[2,2]=1
    A = np.zeros((K,D), dtype=np.float32)
    emb = np.zeros(D, dtype=np.float32) + 0.25
    res = run_wdd(AuditRequest(emb=emb, prototypes=P, anchors=A, T=T, seed=123))
    assert set(res.keep) <= {0,1,2}
    assert res.order == sorted(res.keep, key=lambda k: res.peaks[k].t_star)