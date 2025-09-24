# tests/test_wdd_arc_harness_smoke.py
# Tiny smoke tests for the ARC Tier-2 WDD harness scaffold.
# These tests monkeypatch the notebook-dependent functions so we can
# verify interface + sequencing without the real encoder/warp.

import types
import numpy as np
import pytest

import importlib

mod = importlib.import_module("micro_lm.domains.arc.wdd_arc_harness")


@pytest.fixture(autouse=True)
def _patch_notebook_deps(monkeypatch):
    """Provide lightweight stand-ins for the notebook-specific functions so
    the harness can run deterministically in CI.
    """
    rng = np.random.default_rng(0)

    def synth_grid(cal):
        # 4x4 toy grid with a few colors; ignore `cal`, deterministic output
        return np.array([[3, 0, 1, 2], [3, 2, 1, 0], [3, 2, 0, 5], [6, 1, 4, 2]], dtype=int)

    def toks_from_grid(grid, layer_offset):
        # fake token stream: T x F with mild structure
        T, F = 64, 16
        base = rng.normal(0, 1, size=(T, F)).astype(np.float32)
        return base

    def fit_token_warp(H_list, d=3, whiten=True):
        # minimal warp payload
        return {"whitener": np.eye(3, dtype=np.float32), "mu": np.zeros(3, dtype=np.float32)}

    def traces_from_grid(warp, grid, layer_offset):
        # three channels with separated peaks @ t≈10, 22, 36
        T = 64
        t = np.arange(T)
        def bump(mu):
            return np.exp(-0.5 * ((t - mu) / 3.0) ** 2)
        ch_h = bump(10)
        ch_v = bump(22)
        ch_r = bump(36)
        return [ch_h.astype(np.float32), ch_v.astype(np.float32), ch_r.astype(np.float32)]

    def build_priors(warp, calibrations, L, proto_w=160):
        return {"proto_w": 25, "sigma": 5}

    def prior_pass(traces, priors, *, z, rel_floor, alpha, beta_s):
        # Simple gate: always pass and compute order by argmax
        names = ["flip_h", "flip_v", "rotate"]
        peaks = []
        info = {"keep": names.copy(), "sigma": priors.get("sigma", 5), "proto_w": priors.get("proto_w", 25)}
        for nm, ch in zip(names, traces):
            tpk = int(np.argmax(ch))
            peaks.append((nm, tpk))
        info["order"] = [n for n, _ in sorted(peaks, key=lambda x: x[1])]
        # stash representative t_peak into each channel for harness use (not strictly required)
        info["t_peak"] = {n: t for n, t in peaks}
        info["which_prior"] = "arc_smoke"
        return True, info

    monkeypatch.setattr(mod, "_synthesize_grid", synth_grid, raising=True)
    monkeypatch.setattr(mod, "_get_hidden_states_from_grid", toks_from_grid, raising=True)
    monkeypatch.setattr(mod, "_fit_token_warp", fit_token_warp, raising=True)
    monkeypatch.setattr(mod, "_traces_from_grid", traces_from_grid, raising=True)
    monkeypatch.setattr(mod, "_build_priors_feature_MFpeak", build_priors, raising=True)
    monkeypatch.setattr(mod, "_wdd_prior_pass", prior_pass, raising=True)


def test_family_mode_orders_three_primitives():
    grid = np.array([[3, 0, 1, 2], [3, 2, 1, 0], [3, 2, 0, 5], [6, 1, 4, 2]], dtype=int)
    policy = {"audit": {"mode": "family"}}
    out = mod.run_arc_wdd(
        prompt="flip the grid horizontally then rotate it",
        grid=grid,
        policy=policy,
        sequence=None,
        debug=False,
    )

    assert out["domain"] == "arc"
    assert out["verify"]["ok"] is True
    seq = out["plan"]["sequence"]
    # Expect chronological order: flip_h -> flip_v -> rotate based on synthetic peaks
    assert seq == ["flip_h", "flip_v", "rotate"], f"bad order: {seq}"

    # WDD summary should report PASS with all three kept
    summ = out["wdd_summary"]
    assert summ["decision"] == "PASS"
    assert set(summ["keep"]) == {"flip_h", "flip_v", "rotate"}


def test_detector_mode_keeps_but_no_plan():
    grid = np.array([[1, 2], [3, 4]], dtype=int)
    policy = {"audit": {"mode": "detector"}}
    out = mod.run_arc_wdd(
        prompt="rotate the grid 90 degrees",
        grid=grid,
        policy=policy,
        sequence=None,
        debug=False,
    )

    # In detector mode, plan.sequence should be empty even if detection keeps things
    assert out["plan"]["sequence"] == []
    # But summary should still have keep list
    assert set(out["wdd_summary"]["keep"]) == {"flip_h", "flip_v", "rotate"}
