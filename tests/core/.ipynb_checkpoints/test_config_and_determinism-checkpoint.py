import numpy as np
from micro_lm.core.audit.checks import assert_deterministic
from micro_lm.domains.defi.audit_wdd import wdd_defi_audit

def _once(seed):
    D=12; K=8; T=180
    P = np.zeros((K,D), dtype=np.float32); 
    for k in range(K): P[k, k%D] = 1.0
    A = np.zeros((K,D), dtype=np.float32)
    emb = np.zeros(D, dtype=np.float32) + 0.1

    cfg = {"mode":"family","T":T,"seed":seed}  # no defaults/overrides → pulls YAML
    return wdd_defi_audit(emb, P, A, cfg)

def test_yaml_defaults_loaded_and_deterministic():
    assert_deterministic(_once, seed=123)
