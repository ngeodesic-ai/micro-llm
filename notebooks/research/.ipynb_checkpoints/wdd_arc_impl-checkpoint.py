# Create a ready-to-use implementation that follows the attached pseudocode,
# wiring: mapper (text) -> family routing -> encoder (grid->traces) -> Stage-11 parser (with priors) -> executor.
#
# It exposes:
#   - load_priors_npz(path)
#   - run_arc_wdd(prompt, grid, mapper, traces_from_grid, priors, use_priors=True)
#   - evaluate_arc(samples, mapper, traces_from_grid, priors, use_priors=True)
#
# Save as: wdd_arc_impl.py

from pathlib import Path
import json
import numpy as np

import numpy as np
from typing import Dict, Callable, Tuple, List

try:
    # Prefer package import if available
    from ngeodesic import geodesic_parse_with_prior, geodesic_parse_report
except Exception:
    # Fall back to harness import
    from stage11_benchmark_latest import geodesic_parse_with_prior, geodesic_parse_report

# ----------------------
# Config (fixed at inference; do NOT change per-example)
# ----------------------
WDD_CFG = dict(
    sigma=9,
    proto_width=160,
    alpha=0.05,        # only used in *_with_prior
    beta_s=0.25,
    q_s=2,
    tau_rel=0.60,
    tau_abs_q=0.93,
    null_K=128,
    seed=42,
)

# Family routing (include a foil so parser must discriminate; avoids tautology)
FAMILY = {
    "flip_h": {"flip_h", "flip_v"},
    "flip_v": {"flip_h", "flip_v"},
    "rot90":  {"rot90", "flip_h"},    # rotate vs a flip foil
    "rotate": {"rotate", "flip_h"},   # alias if label is 'rotate'
}

# Actual grid ops to execute (expand as your label set grows)
EXEC = {
    "flip_h": lambda A: A[:, ::-1],
    "flip_v": lambda A: A[::-1, :],
    "rot90":  lambda A: np.rot90(A, k=1),  # 90° CCW; flip sign if your dataset uses CW
    "rotate": lambda A: np.rot90(A, k=1),
}

def load_priors_npz(path: str) -> Dict[str, np.ndarray]:
    """Load Stage-11 priors saved via numpy.savez (keys: r, phi, g)."""
    data = np.load(path)
    priors = {k: data[k] for k in ("r", "phi", "g") if k in data.files}
    if not {"r", "phi", "g"} <= set(priors.keys()):
        raise ValueError("Priors file must contain keys: 'r', 'phi', 'g'.")
    return priors

def _shortlist_from_mapper(mapper_probs: Dict[str, float]) -> Tuple[str, List[str]]:
    """Top-1 + family/foil routing; returns (y_hat, route_list)."""
    if not mapper_probs:
        raise ValueError("Empty mapper probs.")
    y_hat = max(mapper_probs, key=mapper_probs.get)
    route = FAMILY.get(y_hat, set(mapper_probs.keys()))
    return y_hat, sorted(route)

def _stage11_parse(traces: Dict[str, np.ndarray], priors: Dict[str, np.ndarray] = None):
    """Call the Stage-11 parser with fixed gates; priors optional."""
    if priors is not None:
        keep, order = geodesic_parse_with_prior(
            traces, priors,
            sigma=WDD_CFG["sigma"], proto_width=WDD_CFG["proto_width"],
            alpha=WDD_CFG["alpha"], beta_s=WDD_CFG["beta_s"], q_s=WDD_CFG["q_s"],
            tau_rel=WDD_CFG["tau_rel"], tau_abs_q=WDD_CFG["tau_abs_q"],
            null_K=WDD_CFG["null_K"], seed=WDD_CFG["seed"],
        )
    else:
        keep, order = geodesic_parse_report(
            traces,
            sigma=WDD_CFG["sigma"], proto_width=WDD_CFG["proto_width"],
        )
    return keep, order



import numpy as np

def _norm_counts(a):
    a = a.astype(np.float32)
    a -= a.min()
    rng = a.max() - a.min()
    return a / (rng + 1e-6)

def _per_col_hist(grid, ncolors=10):
    # count of each color per column → sum of squared counts (emphasizes solid segments)
    H, W = grid.shape
    out = np.zeros(W, dtype=np.float32)
    for j in range(W):
        counts = np.bincount(grid[:, j], minlength=ncolors)
        out[j] = np.dot(counts, counts)
    return _norm_counts(out)

def _per_row_hist(grid, ncolors=10):
    H, W = grid.shape
    out = np.zeros(H, dtype=np.float32)
    for i in range(H):
        counts = np.bincount(grid[i, :], minlength=ncolors)
        out[i] = np.dot(counts, counts)
    return _norm_counts(out)

def _symmetry_trace_h(grid):
    # column-wise symmetry: compare column j with mirrored column W-1-j
    H, W = grid.shape
    sim = []
    for j in range(W // 2):
        sim.append(np.mean(grid[:, j] == grid[:, W - 1 - j]))
    sim = np.array(sim, dtype=np.float32)
    # interpolate to length W for parser convenience
    return np.interp(np.linspace(0, len(sim)-1, num=W), np.arange(len(sim)), sim)

def _symmetry_trace_v(grid):
    H, W = grid.shape
    sim = []
    for i in range(H // 2):
        sim.append(np.mean(grid[i, :] == grid[H - 1 - i, :]))
    sim = np.array(sim, dtype=np.float32)
    return np.interp(np.linspace(0, len(sim)-1, num=H), np.arange(len(sim)), sim)

def _rotation_consistency_trace(grid):
    # how consistent the grid is with a 90° rotation (proxy signal)
    rot = np.rot90(grid, k=1)
    # compare bands (diagonals approximated by offset rows/cols)
    H, W = grid.shape
    # compare main diagonal band-wise using downsampled stripes
    bands = min(H, W)
    vals = []
    for k in range(bands):
        i = k if k < H else H - 1
        j = k if k < W else W - 1
        vals.append(1.0 * (grid[i, :].shape[0] == rot[:, j].shape[0]) *
                    np.mean(grid[i, :] == rot[:, j] if grid[i, :].shape == rot[:, j].shape else 0.0))
    vals = np.array(vals, dtype=np.float32)
    # stretch to a common length (e.g., max(H, W))
    L = max(H, W)
    return np.interp(np.linspace(0, len(vals)-1, num=L), np.arange(len(vals)), vals)

def traces_from_grid(grid, ncolors=10):
    """
    Return 1-D traces per label for WDD. Plug this into your Stage-11 call.
    You can replace any of these with your real warp/detect channels when ready.
    """
    grid = np.asarray(grid, dtype=int)
    # generic texture/segment signals
    col_hist = _per_col_hist(grid, ncolors)   # length W
    row_hist = _per_row_hist(grid, ncolors)   # length H

    # label-specific traces
    flip_h_trace = _symmetry_trace_h(grid)    # strong if horizontal mirror is plausible
    flip_v_trace = _symmetry_trace_v(grid)    # strong if vertical mirror is plausible

    # rotation proxy; normalize to similar dynamic range
    rot90_trace  = _rotation_consistency_trace(grid)
    rot90_trace  = _norm_counts(rot90_trace)

    # Optional: concatenate generic bandpower to stabilize (parser likes smoother wells)
    # Here we fuse each label trace with hist-based context
    def fuse(a, b):
        L = max(len(a), len(b))
        aa = np.interp(np.linspace(0, len(a)-1, num=L), np.arange(len(a)), a)
        bb = np.interp(np.linspace(0, len(b)-1, num=L), np.arange(len(b)), b)
        return 0.7*aa + 0.3*bb

    return {
        "flip_h": fuse(flip_h_trace, col_hist),   # length ≈ max(W, W)
        "flip_v": fuse(flip_v_trace, row_hist),   # length ≈ max(H, H)
        "rot90":  rot90_trace,                    # length ≈ max(H, W)
    }



def run_arc_wdd(
    prompt: str,
    grid: np.ndarray,
    mapper: Callable[[str], Dict[str, float]],
    traces_fn: Callable[[np.ndarray], Dict[str, np.ndarray]],
    priors: Dict[str, np.ndarray] = None,
    use_priors: bool = True,
):
    """
    Implements the attached pseudocode:
      1) mapper(prompt) -> probs (NO threshold coupling)
      2) family routing (include a foil)
      3) traces_from_grid(grid) -> dict[label->1D trace]; restrict to route
      4) Stage-11 parser with fixed gates (+/- priors)
      5) Verdict: PASS iff exactly one kept AND it matches mapper top-1; else ABSTAIN
      6) If PASS, execute ops in returned order
    """
    # 1) mapper
    p_text = mapper(prompt)  # e.g., {"flip_h":0.9,"flip_v":0.08,"rot90":0.02}
    y_hat, route = _shortlist_from_mapper(p_text)

    # 2) traces
    if not callable(traces_fn):
        raise TypeError("traces_fn must be a function (grid -> dict[label->1D np.array]). "
                        "It looks like you passed the result dict instead.")
    all_traces = traces_fn(grid)      # {"flip_h":1D,"flip_v":1D,"rot90":1D,...}
    traces = {k: all_traces[k] for k in route if k in all_traces}
    if len(traces) == 0:
        return dict(verdict="ABSTAIN", reason="no_traces_for_route", mapper=p_text, route=route)

    # 3) parse (no per-example gate changes)
    keep, order = _stage11_parse(traces, priors if use_priors else None)

    # 4) decision
    if len(keep) == 1 and keep[0] == y_hat:
        verdict = "PASS"
    else:
        return dict(verdict="ABSTAIN", reason="ambiguous_or_mismatch", keep=keep, order=order,
                    mapper=p_text, route=route)

    # 5) execute
    out = np.array(grid, copy=True)
    for op in order:
        if op not in EXEC:
            return dict(verdict="PASS", label=y_hat, keep=keep, order=order,
                        warn=f"Executor missing op '{op}'", out=out, mapper=p_text, route=route)
        out = EXEC[op](out)

    return dict(verdict=verdict, label=y_hat, keep=keep, order=order, out=out,
                mapper=p_text, route=route)

# ----------------------
# Batch evaluator (macro-F1 with ABSTAIN bucket)
# ----------------------
def macro_f1_from_preds(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true) | set([l for l in y_pred if l != "ABSTAIN"]))
    f1s = []
    for lbl in labels:
        tp = sum((t == lbl) and (p == lbl) for t, p in zip(y_true, y_pred))
        fp = sum((t != lbl) and (p == lbl) for t, p in zip(y_true, y_pred))
        fn = sum((t == lbl) and (p != lbl) for t, p in zip(y_true, y_pred))
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0

def evaluate_arc(
    samples: List[dict],
    mapper: Callable[[str], Dict[str, float]],
    traces_from_grid: Callable[[np.ndarray], Dict[str, np.ndarray]],
    priors: Dict[str, np.ndarray] = None,
    use_priors: bool = True,
):
    y_true, y_pred = [], []
    abstain = 0
    for s in samples:
        res = run_arc_wdd(s["prompt"], s["grid"], mapper, traces_from_grid, priors, use_priors)
        y_true.append(s["true"])
        if res["verdict"] == "PASS":
            y_pred.append(res.get("label", "ABSTAIN"))
        else:
            y_pred.append("ABSTAIN")
            abstain += 1
    return dict(
        macro_f1=macro_f1_from_preds(y_true, y_pred),
        abstain_rate=abstain / max(1, len(samples)),
        counts={k: int(sum(1 for p in y_pred if p == k)) for k in set(y_pred)},
    )

