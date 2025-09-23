# tests/core/test_pca_priors.py
import json
import numpy as np
from pathlib import Path

from micro_lm.core.audit.pca_prior import load_pca_prior, apply_pca_prior


def _write_prior(tmp_path: Path, *, mean, comps):
    mean = np.asarray(mean, dtype=np.float32)
    comps = np.asarray(comps, dtype=np.float32)  # shape (k, D)
    p = tmp_path / "prior.npz"
    np.savez_compressed(p, mean=mean, components=comps)
    return p


def test_load_pca_prior_roundtrip(tmp_path):
    # D=4, k=2
    p = _write_prior(tmp_path,
                     mean=[1, 2, 3, 4],
                     comps=[[1, 0, 0, 0],
                            [0, 1, 0, 0]])
    prior = load_pca_prior(str(p))  # should infer k from file; no k arg required
    assert set(prior.keys()) == {"mean", "components"}
    assert prior["mean"].dtype == np.float32
    assert prior["components"].dtype == np.float32
    assert prior["mean"].shape == (4,)
    assert prior["components"].shape == (2, 4)
    np.testing.assert_allclose(prior["mean"], [1, 2, 3, 4], rtol=0, atol=0)


def test_apply_pca_prior_identity_like():
    # components pick first two axes, mean zero
    prior = {
        "mean": np.zeros(4, dtype=np.float32),
        "components": np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0]], dtype=np.float32),
    }
    x = np.array([10.0, -3.0, 7.0, 2.0], dtype=np.float32)
    y = apply_pca_prior(x, prior)
    assert y.shape == (2,)
    np.testing.assert_allclose(y, [10.0, -3.0], rtol=0, atol=0)


def test_apply_pca_prior_centers_before_project():
    prior = {
        "mean": np.array([5.0, 5.0, 0.0, 0.0], dtype=np.float32),
        "components": np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0]], dtype=np.float32),
    }
    x = np.array([7.0, 9.0, 0.0, 0.0], dtype=np.float32)
    y = apply_pca_prior(x, prior)
    np.testing.assert_allclose(y, [2.0, 4.0], rtol=0, atol=0)


def test_apply_pca_prior_dimension_reduction_and_dtype():
    # D=3 -> k=2; output must be float32 even if input is float64
    prior = {
        "mean": np.zeros(3, dtype=np.float32),
        "components": np.array([[1, 0, 0],
                                [0, 1, 0]], dtype=np.float32),
    }
    x = np.array([1, 2, 3], dtype=np.float64)
    y = apply_pca_prior(x, prior)
    assert y.shape == (2,)
    assert y.dtype == np.float32
    np.testing.assert_allclose(y, [1.0, 2.0], rtol=0, atol=0)


def test_apply_pca_prior_with_loaded_file(tmp_path):
    # End-to-end: write → load → project (use non-trivial components)
    p = _write_prior(tmp_path,
                     mean=[0, 0, 0, 0],
                     comps=[[0.5, 0.5, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0]])
    prior = load_pca_prior(str(p))
    x = np.array([2.0, 6.0, 3.0, 0.0], dtype=np.float32)
    # First comp = avg of first two dims → (2+6)/2 = 4.0
    # Second comp = third dim
    y = apply_pca_prior(x, prior)
    np.testing.assert_allclose(y, [4.0, 3.0], rtol=0, atol=0)


def test_load_pca_prior_rejects_bad_shapes(tmp_path):
    # components must be 2D (k, D); mean must be 1D (D,)
    p = tmp_path / "bad.npz"
    np.savez_compressed(p,
                        mean=np.zeros((2, 2), dtype=np.float32),  # wrong shape
                        components=np.zeros((2, 3, 1), dtype=np.float32))  # wrong shape
    try:
        _ = load_pca_prior(str(p))
        raised = False
    except Exception:
        raised = True
    assert raised, "load_pca_prior should raise on invalid shapes"
