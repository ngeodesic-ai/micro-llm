import numpy as np
from micro_lm.core.audit import AuditRequest, FamilySpec, Mode
from micro_lm.core.audit.orchestrator import run_wdd

def _req(K=4, D=9, T=400, seed=7):
    P = np.zeros((K,D), dtype=np.float32)
    for k in range(K): P[k, k%D] = 1.0
    A = np.zeros((K,D), dtype=np.float32)
    emb = np.zeros(D, dtype=np.float32) + 0.2
    return AuditRequest(emb=emb, prototypes=P, anchors=A, T=T, seed=seed)

def test_dual_gates_and_null_floor():
    r = run_wdd(_req())
    assert len(r.keep) >= 1
    for k in r.keep:
        assert r.peaks[k].corr_max >= r.zfloor

def test_family_mode_tightens_selection():
    req = _req(K=4)
    req.families = [
        FamilySpec(name="fam0", idxs=[0,1], z_abs=0.6, keep_frac=0.9),
        FamilySpec(name="fam1", idxs=[2,3], z_abs=0.2, keep_frac=0.5),
    ]
    req.mode = Mode.FAMILY
    r = run_wdd(req)
    kept0 = [k for k in r.keep if k in (0,1)]
    kept1 = [k for k in r.keep if k in (2,3)]
    assert len(kept1) >= len(kept0)

def test_seed_determinism():
    r1 = run_wdd(_req(seed=42))
    r2 = run_wdd(_req(seed=42))
    assert r1.keep == r2.keep and r1.order == r2.order