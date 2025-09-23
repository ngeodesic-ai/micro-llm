import numpy as np
from micro_lm.domains.defi.audit_wdd import wdd_defi_audit

def test_keep_frac_effect_on_len_keep():
    D=12; K=8; T=180
    P = np.zeros((K,D), dtype=np.float32)
    for k in range(K): P[k, k%D] = 1.0
    A = np.zeros((K,D), dtype=np.float32)
    emb = np.zeros(D, dtype=np.float32) + 0.2

    cfg_lo = {"mode":"family","T":T,"seed":0,"family_overrides":{"default":{"keep_frac":0.5}}}
    cfg_hi = {"mode":"family","T":T,"seed":0,"family_overrides":{"default":{"keep_frac":0.9}}}

    out_lo = wdd_defi_audit(emb, P, A, cfg_lo)
    out_hi = wdd_defi_audit(emb, P, A, cfg_hi)

    assert len(out_lo.get("keep", [])) >= len(out_hi.get("keep", []))
