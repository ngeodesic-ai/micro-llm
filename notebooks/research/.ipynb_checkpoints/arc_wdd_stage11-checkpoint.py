
# arc_wdd_stage11.py
# Stage-11 Warp → Detect → Denoise for ARC-style prompts
# Drop-in sidecar to your existing ARC pipeline.
# It wraps your existing geodesic helpers with a robust detector:
# - robust null (circular shifts with large min-shift, MAD-backed std)
# - per-label calibrated z floors (0.80 quantile)
# - one-shot retry if prior clips everything
# - soft-pass near gate when clearly above null
# - z-margin + smax tie-break across priors
#
# Usage (example):
#   from arc_wdd_stage11 import decide_arc, derive_per_prior_thresholds, build_priors_feature_MFpeak
#
#   priors = build_priors_feature_MFpeak(warp, train_grids)
#   per_prior_thresh = derive_per_prior_thresholds(df_labeled, labels, warp, priors)
#   ok, info = decide_arc(grid, guess_label=None, family_only=False)
#
# NOTE: This module imports helpers from your existing arc_wdd.py when present.
# If a helper is missing, simple fallbacks are provided.

from __future__ import annotations
import numpy as np

# --- Try to import project helpers from existing arc_wdd ---
try:
    from arc_wdd import (
        traces_from_grid,
        geodesic_parse_with_prior,
        fit_radial_profile,
        analytic_core_template,
        blend_profiles,
        load_grid
    )
except Exception:
    # Minimal fallbacks (you likely won't need these if arc_wdd.py is available)
    def traces_from_grid(grid, warp):
        # Expect grid already transformed to traces externally
        # Here, just pass through if it's already traces-like: list[np.ndarray] length 3
        if isinstance(grid, (list, tuple)) and len(grid) == 3:
            return [np.asarray(ch, float) for ch in grid]
        raise RuntimeError("traces_from_grid fallback: please provide your project's implementation.")
    def geodesic_parse_with_prior(traces, priors, sigma, proto_width, z, rel_floor, alpha, beta_s, q_s):
        # Fallback: accept all channels/windows (no clipping). This keeps detector usable.
        T = len(traces[0])
        return [0,1,2], [0,1,2]
    def fit_radial_profile(R, Zs, n_r=220, fit_quantile=0.70):
        r = np.linspace(0, 1, n_r)
        z = np.quantile(np.stack(Zs,0), fit_quantile, axis=0) if len(Zs) else np.zeros_like(r)
        return r, z
    def analytic_core_template(r_grid, k=0.16, p=1.7, r0_frac=0.16):
        return np.exp(-k * ((np.maximum(0, r_grid - r0_frac))**p))
    def blend_profiles(z_data, z_core, blend_core=0.22):
        a = np.clip(blend_core, 0, 1)
        return a*z_core + (1-a)*z_data
    def load_grid(raw):  # identity fallback
        return raw

LABELS = ['cc_mask_largest','color_map','crop_pad','flip_h','flip_v','rot90','tile','translate']

# ---------------- Stage-11 knobs (tunable) ----------------
DELTA_MARGIN_Z = 0.10     # required z margin vs runner-up
DELTA_MARGIN_S = 0.08     # alternative smax margin to break ties
MF_FALLBACK_FLOOR = 0.18  # minimal smax to allow guarded fallback
NULL_K = 128              # circular shift permutations
REL_FLOOR_DEFAULT = 0.45  # prior relative gate floor

# ---------------- Small helpers ----------------
def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    if k == 1:
        return x
    w = np.ones(k, dtype=float) / k
    return np.convolve(x, w, mode="same")

def _mf_peak(x: np.ndarray, proto_w: int) -> float:
    # Normalized cross-correlation with a unimodal window (flat/box as baseline)
    w = max(3, int(proto_w))
    w = min(w, len(x))
    if w <= 2:
        return 0.0
    tpl = np.hanning(w)  # smooth unimodal template
    tpl = (tpl - tpl.mean())
    tpl = tpl / (np.linalg.norm(tpl) + 1e-12)
    # sliding dot with normalization
    best = 0.0
    half = w // 2
    for t in range(half, len(x)-half):
        seg = x[t-half:t+half+1]
        seg = seg - seg.mean()
        denom = np.linalg.norm(seg) + 1e-12
        best = max(best, float(np.dot(seg, tpl) / denom))
    return best

def adaptive_windows_short(T: int):
    """Keep kernels local to avoid flattening peaks."""
    proto_w = max(15, min(int(0.45*T), 41))
    sigma   = max(3,  min(int(T/10), 7))
    return sigma, proto_w

def _smax_over_channels(traces, proto_w):
    # mild channel-wise smoothing, then take max MF peak across channels
    mx = 0.0
    for ch in traces:
        ch = np.asarray(ch, float)
        T = len(ch)
        k = min(9, max(3, T//8))
        mx = max(mx, _mf_peak(moving_average(ch, k=k), proto_w))
    return mx

def _null_circ_shift_scores(traces, proto_w, K=NULL_K):
    """Robust null: more permutations, large min shift, MAD-backed std."""
    rng = np.random.default_rng(0)
    T = len(traces[0])
    min_shift = max(proto_w + 3, int(0.6*T))
    ks = []
    for _ in range(K):
        hi = T - min_shift if T - min_shift > min_shift else T - 1
        offs = rng.integers(low=min_shift, high=hi)
        tr = [np.roll(ch, offs) for ch in traces]
        ks.append(_smax_over_channels(tr, proto_w))
    arr = np.asarray(ks, float)
    mu  = float(arr.mean())
    sd  = float(arr.std(ddof=1))
    mad = float(np.median(np.abs(arr - mu))) * 1.4826
    sd_eff = max(sd, 0.8*mad, 1e-6)
    return mu, sd_eff

# ---------------- Core Stage-11 detector ----------------
def wdd_prior_pass_scored(traces, priors,
                          *, z_min=1.7, rel_floor=0.55,
                          alpha=0.16, beta_s=0.54, q_s=2.0,
                          null_K=NULL_K):
    """
    Prior + evidence:
      1) geodesic funnel prior (clips windows)
      2) MF peak evidence on accepted windows
      3) null via circular shifts → z-score
    Returns (ok, info)
    """
    T = len(traces[0])
    if T < 6:
        return False, {"reason":"too_short","T":T}

    sigma, proto_w = adaptive_windows_short(T)

    keep, order = geodesic_parse_with_prior(
        traces, priors=priors, sigma=sigma, proto_width=proto_w,
        z=z_min, rel_floor=rel_floor, alpha=alpha, beta_s=beta_s, q_s=q_s
    )

    # One-shot retry if the prior clipped everything
    if not keep:
        keep2, order2 = geodesic_parse_with_prior(
            traces, priors=priors, sigma=sigma, proto_width=proto_w,
            z=max(1.2, z_min-0.2), rel_floor=max(0.40, rel_floor-0.10),
            alpha=max(0.10, alpha-0.04), beta_s=max(0.40, beta_s-0.10), q_s=q_s
        )
        if keep2:
            keep, order = keep2, order2
        else:
            return False, {"keep":[], "order":order, "sigma":sigma, "proto_w":proto_w,
                           "smax":0.0, "z":-999, "reason":"prior_clip"}

    s_obs = _smax_over_channels(traces, proto_w)
    mu0, sd0 = _null_circ_shift_scores(traces, proto_w, K=null_K)
    z = (s_obs - mu0) / sd0

    ok = (z >= z_min)
    # Soft-pass near the gate if clearly above null
    if (not ok) and (z >= z_min - 0.25) and (s_obs >= mu0 + 0.6*sd0):
        info = {"keep":keep, "order":order, "sigma":sigma, "proto_w":proto_w,
                "smax":s_obs, "z":z, "reason":"soft_pass"}
        return True, info

    info = {"keep":keep, "order":order, "sigma":sigma, "proto_w":proto_w, "smax":s_obs, "z":z}
    return ok, info

def decide_arc_with_margins_per_prior(grid, warp, priors, per_prior_thresh, delta_margin=DELTA_MARGIN_Z):
    traces = traces_from_grid(grid, warp)
    T = len(traces[0]); _, proto_w = adaptive_windows_short(T)

    scored = []
    for name, pr in priors.items():
        z_gate = max(1.2, min(per_prior_thresh.get(name, 1.7), 3.0))
        ok, info = wdd_prior_pass_scored(
            traces, pr,
            z_min=z_gate, rel_floor=REL_FLOOR_DEFAULT, alpha=0.14, beta_s=0.50, q_s=1.8,
            null_K=NULL_K
        )
        scored.append((name, info.get("z",-999), info.get("smax",0.0), ok, info))

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best = scored[0]; second = scored[1] if len(scored)>1 else None
    best_name, best_z, best_s, _, best_info = best
    sec_z = second[1] if second else -999

    if best_z < per_prior_thresh.get(best_name, 1.7):
        return False, {"which_prior":"unknown", "reason":"weak_evidence", **best_info}
    if (best_z - sec_z) < delta_margin:
        # tie-break by MF peak if clearly separated
        sec_s = second[2] if second else -999
        if (best_s - sec_s) >= DELTA_MARGIN_S:
            best_info["which_prior"] = best_name
            best_info["reason"] = "margin_by_smax"
            return True, best_info
        return False, {"which_prior":"ambiguous", "reason":"no_margin",
                       "best":(best_name,best_z), "second":(second[0],sec_z) if second else None, **best_info}
    best_info["which_prior"] = best_name
    return True, best_info

def _mf_fallback_pass(traces, floor=MF_FALLBACK_FLOOR):
    T = len(traces[0]); proto_w = max(15, min(int(0.45*T), 41))
    return (_smax_over_channels(traces, proto_w) >= floor)

def decide_arc(grid, warp, priors, per_prior_thresh, guess_label=None, *, family_only=True):
    """Top-level ARC decision with Stage-11 guards."""
    traces = traces_from_grid(grid, warp)

    if guess_label is not None and family_only:
        pr = priors[guess_label]
        ok, info = wdd_prior_pass_scored(
            traces, pr,
            z_min=max(1.2, per_prior_thresh.get(guess_label, 1.4)), rel_floor=REL_FLOOR_DEFAULT,
            alpha=0.14, beta_s=0.50, q_s=1.8,
            null_K=NULL_K
        )
        if not ok and info.get("reason") in {"soft_pass","prior_clip"}:
            if _mf_fallback_pass(traces, floor=MF_FALLBACK_FLOOR):
                ok = True
                info.setdefault("note", "mf_fallback")
        return ok, {**info, "which_prior": guess_label}

    ok, info = decide_arc_with_margins_per_prior(
        grid, warp, priors, per_prior_thresh, delta_margin=DELTA_MARGIN_Z
    )
    return ok, info

# --------- Prior building + per-label thresholds (optional helpers) ---------
def build_priors_feature_MFpeak(warp, grids, proto_w=96):
    """Example prior builder that adapts smoothing/window per trace."""
    Zs = []
    R  = np.linspace(0, 1, 220)
    priors = {}
    for name in LABELS:
        Zs_label = []
        for g in grids.get(name, []):
            tr = traces_from_grid(g, warp)
            T = len(tr[0])
            k_smooth = min(9, max(3, T//8))
            w_adapt  = min(proto_w, max(15, int(0.5*T)))
            # Simple feature: max MF across channels with adapted window
            Zs_label.append(_smax_over_channels(tr, w_adapt))
        Zs.append(Zs_label or [0.0])
        priors[name] = {"name": name, "params": {}}
    # Construct a blended radial core (placeholder, depends on your project data)
    r_grid, z_data = fit_radial_profile(R, Zs, n_r=220, fit_quantile=0.70)
    z_core  = analytic_core_template(r_grid, k=0.16, p=1.7, r0_frac=0.16)
    z_blend = blend_profiles(z_data, z_core, blend_core=0.22)
    return priors  # keep your project's own structure

def derive_per_prior_thresholds(df, labels, warp, priors, holdout_K=24, quantile=0.80):
    """Estimate per-label z floors from a small holdout."""
    per_prior_thresh = {}
    rng = np.random.default_rng(1234)
    for lab in labels:
        rows = df[df.label==lab]
        if len(rows) == 0:
            per_prior_thresh[lab] = 1.7
            continue
        take = min(holdout_K, len(rows))
        idxs = rng.choice(len(rows), size=take, replace=False)
        zs = []
        for i in idxs:
            r = rows.iloc[i]
            g = load_grid(r.grid) if hasattr(r, 'grid') else r  # project-specific
            tr = traces_from_grid(g, warp)
            ok, info = wdd_prior_pass_scored(
                tr, priors[lab],
                z_min=-999, rel_floor=REL_FLOOR_DEFAULT, alpha=0.14, beta_s=0.50, q_s=1.8,
                null_K=NULL_K
            )
            zs.append(info.get("z", -999))
        z_min = float(np.quantile([z for z in zs if np.isfinite(z)], quantile))
        per_prior_thresh[lab] = max(1.2, min(z_min, 3.0))
    return per_prior_thresh
