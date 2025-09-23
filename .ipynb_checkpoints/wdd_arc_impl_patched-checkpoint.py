# Write a patched version of wdd_arc_impl.py that:
# - Accepts priors in either schema:
#     (A) radial funnel: {"r_grid","z_grid"}  -> pass straight to parser
#     (B) temporal curves: {"r","phi","g"}    -> synthesize a neutral radial funnel
# - Falls back to no-priors parsing when priors are missing or invalid
# - Exposes: load_priors_npz, run_arc_wdd, evaluate_arc (unchanged API)
#
# Save to /mnt/data/wdd_arc_impl_patched.py for download.

# at top-level config:
DECISION_MODE = "strict"  # "strict" | "prefer_mapper" | "parser_singleton"


from pathlib import Path
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
    "rot90":  {"rot90", "flip_h"},
    "rotate": {"rotate", "flip_h"},   # alias
}

# Actual grid ops to execute (expand as your label set grows)
EXEC = {
    "flip_h": lambda A: A[:, ::-1],
    "flip_v": lambda A: A[::-1, :],
    "rot90":  lambda A: np.rot90(A, k=1),  # 90° CCW; flip sign if your dataset uses CW
    "rotate": lambda A: np.rot90(A, k=1),
}

# ----------------------
# Priors utilities
# ----------------------
def _make_neutral_radial_funnel(length: int = 64, rmax: float = 1.0) -> Dict[str, np.ndarray]:
    """Create a simple convex funnel prior over radius for parser compatibility."""
    length = max(2, int(length))
    r_grid = np.linspace(0.0, float(rmax), length, dtype=np.float32)
    z_grid = (1.0 - (r_grid / max(rmax, 1e-9))) ** 2  # high at center, taper outward
    return {"r_grid": r_grid, "z_grid": z_grid}

def _normalize_priors_dict(npz) -> Dict[str, np.ndarray]:
    """Accept priors in two schemas and normalize to {'r_grid','z_grid'} for the parser."""
    keys = set(npz.files)
    # Preferred schema: radial grid
    if {"r_grid", "z_grid"} <= keys:
        rg, zg = npz["r_grid"], npz["z_grid"]
        # basic sanity
        if rg.ndim != 1 or zg.ndim != 1 or len(rg) < 2 or len(rg) != len(zg) or not np.all(np.diff(rg) > 0):
            raise ValueError("Invalid r_grid/z_grid arrays in priors.")
        return {"r_grid": rg.astype(np.float32), "z_grid": zg.astype(np.float32)}
    # Legacy/temporal schema: r, phi, g  -> synthesize a neutral radial funnel
    if {"r", "phi", "g"} <= keys:
        T = int(npz["g"].shape[0])
        return _make_neutral_radial_funnel(length=T, rmax=1.0)
    raise ValueError(f"Priors file must contain either "
                     f"('r_grid','z_grid') or ('r','phi','g'); found keys={sorted(keys)}")

def load_priors_npz(path: str) -> Dict[str, np.ndarray]:
    """Load and normalize Stage-11 priors for geodesic_parse_with_prior."""
    data = np.load(path)
    return _normalize_priors_dict(data)

# ----------------------
# Core pipeline
# ----------------------
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

def run_arc_wdd(
    prompt: str,
    grid: np.ndarray,
    mapper: Callable[[str], Dict[str, float]],
    traces_from_grid: Callable[[np.ndarray], Dict[str, np.ndarray]],
    priors: Dict[str, np.ndarray] = None,
    use_priors: bool = True,
):
    """
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
    all_traces = traces_from_grid(grid)      # {"flip_h":1D,"flip_v":1D,"rot90":1D,...}
    traces = {k: all_traces[k] for k in route if k in all_traces}
    if len(traces) == 0:
        return dict(verdict="ABSTAIN", reason="no_traces_for_route", mapper=p_text, route=route)

    # 3) parse (no per-example gate changes)
    pri = priors if (use_priors and priors is not None) else None
    keep, order = _stage11_parse(traces, pri)


    # 4) decision
    if DECISION_MODE == "strict":
        # require agreement between mapper top1 and parser singleton
        if len(keep) == 1 and keep[0] == y_hat:
            verdict = "PASS"
        else:
            return dict(verdict="ABSTAIN", reason="ambiguous_or_mismatch",
                        keep=keep, order=order, mapper=p_text, route=route)

    elif DECISION_MODE == "prefer_mapper":
        # if parser kept mapper's top1 anywhere (even among >1), pass with mapper’s label
        if (len(keep) == 1 and keep[0] == y_hat) or (y_hat in keep):
            verdict = "PASS"
            # optional: constrain order to mapper’s op if present
            order = [op for op in order if op == y_hat] or order
        else:
            return dict(verdict="ABSTAIN", reason="mapper_not_in_keep",
                        keep=keep, order=order, mapper=p_text, route=route)

    elif DECISION_MODE == "parser_singleton":
        # accept any singleton from parser (even if it disagrees with mapper)
        if len(keep) == 1:
            verdict = "PASS"
            y_hat = keep[0]  # adopt parser label
        else:
            return dict(verdict="ABSTAIN", reason="parser_not_singleton",
                        keep=keep, order=order, mapper=p_text, route=route)

    else:
        return dict(verdict="ABSTAIN", reason=f"unknown_decision_mode:{DECISION_MODE}",
                    keep=keep, order=order, mapper=p_text, route=route)
    
    # # 4) decision
    # if len(keep) == 1 and keep[0] == y_hat:
    #     verdict = "PASS"
    # else:
    #     return dict(verdict="ABSTAIN", reason="ambiguous_or_mismatch", keep=keep, order=order,
    #                 mapper=p_text, route=route)

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

