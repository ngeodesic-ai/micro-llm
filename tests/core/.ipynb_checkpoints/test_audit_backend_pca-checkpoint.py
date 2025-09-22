import numpy as np
from pathlib import Path

def _write_prior(tmp_path: Path, mean, comps):
    p = tmp_path / "prior.npz"
    np.savez_compressed(
        p,
        mean=np.array(mean, dtype=np.float32),
        components=np.array(comps, dtype=np.float32),
    )
    return p

def test_get_audit_backend_pca_basic(tmp_path):
    from micro_lm.core.audit import get_audit_backend
    from micro_lm.core.audit.pca_prior import load_pca_prior, apply_pca_prior

    fn = get_audit_backend("pca")
    assert callable(fn)

    # basic prior + vector
    p = _write_prior(tmp_path,
                     mean=[0.0, 0.0, 0.0, 0.0],
                     comps=[[1,0,0,0],[0,1,0,0]])  # k=2, D=4

    prior = load_pca_prior(str(p))
    x = np.array([10.0, -3.0, 7.0, 2.0], dtype=np.float32)
    z = apply_pca_prior(x, prior)
    assert z.shape == (2,)
