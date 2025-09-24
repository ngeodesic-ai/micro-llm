# tests/domains/arc/test_wdd_arc_harness_smoke.py
# Smoke tests for the refactored ARC WDD harness.
# We monkeypatch ONLY the public audit hook (wdd_arc_audit) to avoid
# model/artifact dependencies and to keep CI deterministic.

import importlib
import numpy as np
import pytest

mod = importlib.import_module("micro_lm.domains.arc.wdd_arc_harness")

@pytest.fixture
def grid4():
    return np.array([[3,0,1,2],
                     [3,2,1,0],
                     [3,2,0,5],
                     [6,1,4,2]], dtype=int)

def _fake_audit(selected=("flip_h","rotate")):
    """Return a deterministic fake audit with per-class metrics."""
    sel = list(selected)
    # minimal per-class metrics in the shape the harness expects
    per_class = {
        "flip_h": {"corr_max": 0.30, "t_star": 10, "area": 1.0, "window": (5,15), "z_abs": 1.2},
        "flip_v": {"corr_max": 0.10, "t_star": 20, "area": 0.1, "window": (18,22), "z_abs": 0.1},
        "rotate": {"corr_max": 0.35, "t_star": 30, "area": 2.0, "window": (28,34), "z_abs": 2.0},
    }
    return {
        "prompt": "stub",
        "scores": {"rotate": 0.6, "flip_h": 0.3, "flip_v": 0.1},
        "candidates": ["rotate", "flip_h", "flip_v"],
        "selected": sel,
        "per_class": per_class,
    }

def test_family_mode_uses_selected_order(monkeypatch, grid4):
    # selected order should become plan.sequence in family mode
    monkeypatch.setattr(mod, "wdd_arc_audit", lambda *a, **k: _fake_audit(("rotate","flip_v")))
    out = mod.run_arc_wdd(
        prompt="rotate then flip vertically",
        grid=grid4,
        policy={"audit": {"mode": "family"}},
        sequence=None,
        debug=False,
    )
    assert out["domain"] == "arc"
    assert out["verify"]["ok"] is True
    assert out["plan"]["sequence"] == ["rotate", "flip_v"]  # family mode uses keep_order
    # summary reflects the same order and keep set
    summ = out["wdd_summary"]
    assert summ["decision"] == "PASS"
    assert summ["order"] == ["rotate", "flip_v"]
    assert summ["keep"] == ["rotate", "flip_v"]

def test_detector_mode_keeps_but_no_plan(monkeypatch, grid4):
    # In detector mode, plan.sequence must be empty but summary.keep mirrors selection
    monkeypatch.setattr(mod, "wdd_arc_audit", lambda *a, **k: _fake_audit(("flip_h","rotate")))
    out = mod.run_arc_wdd(
        prompt="flip then rotate",
        grid=grid4,
        policy={"audit": {"mode": "detector"}},
        sequence=None,
        debug=False,
    )
    assert out["plan"]["sequence"] == []  # detector mode -> no plan
    assert out["verify"]["ok"] is True
    # but the WDD summary still reports what was kept
    summ = out["wdd_summary"]
    assert summ["keep"] == ["flip_h","rotate"]
    assert summ["order"] == []  # detector mode -> no order

def test_results_map_shape_and_ok_flags(monkeypatch, grid4):
    # Validate aux map structure and ok flags per primitive
    monkeypatch.setattr(mod, "wdd_arc_audit", lambda *a, **k: _fake_audit(("flip_h","rotate")))
    out = mod.run_arc_wdd(
        prompt="flip then rotate",
        grid=grid4,
        policy={"audit": {"mode": "family"}},
        sequence=None,
        debug=False,
    )
    results = out["aux"]["stage11"]["wdd"]["arc"]["results"]
    # keys present
    assert set(results.keys()) == {"flip_h","flip_v","rotate"}
    # ok flags align with selection
    assert results["flip_h"]["ok"] is True
    assert results["rotate"]["ok"] is True
    assert results["flip_v"]["ok"] is False
    # minimal metrics present
    r = results["rotate"]["info"]
    assert "t_peak" in r and "corr_max" in r and "area" in r and "window" in r and "z_abs" in r
