# ==== ARC WDD — notebook tester (mirrors DeFi WDD flow) ====
import os, json, math, random, numpy as np, pandas as pd
import torch, torch.nn as nn
from pathlib import Path

# ---------- 0) Data ----------
CSV = "tests/fixtures/arc/arc_mapper_labeled.csv"  # <-- change if yours is elsewhere
df = pd.read_csv(CSV)
assert {"grid","label"}.issubset(df.columns), "CSV must have grid,label columns"

def load_grid(cell):
    if isinstance(cell, str): return np.asarray(json.loads(cell), dtype=int)
    return np.asarray(cell, dtype=int)

LABELS = sorted(df["label"].unique())
print("labels:", LABELS, "n=", len(df))

# ---------- 1) Tiny ARC encoder (H×W grid -> per-cell hidden states) ----------
# Inputs: one-hot color (10) + (x,y) pos sin/cos (8)  => 18 channels
# Tiny CNN -> [B, D, H, W] ; we'll flatten per cell to [T,H]
class TinyARCEncoder(nn.Module):
    def __init__(self, in_ch=18, hid=64, depth=3):
        super().__init__()
        ch = hid
        layers = [nn.Conv2d(in_ch, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth-1):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        # x: [B, C, H, W] -> [B, D, H, W]
        return self.net(x)

def posenc_xy(H, W):
    # simple sin/cos pe (4 per axis = 8 total)
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    ys = ys / max(1,H-1); xs = xs / max(1,W-1)
    freqs = np.array([1.0, 2.0, 4.0, 8.0], dtype=float)
    pe = []
    for f in freqs:
        pe.append(np.sin(2*math.pi*f*xs)[None,:,:])
        pe.append(np.cos(2*math.pi*f*xs)[None,:,:])
    for f in freqs:
        pe.append(np.sin(2*math.pi*f*ys)[None,:,:])
        pe.append(np.cos(2*math.pi*f*ys)[None,:,:])
    return np.vstack(pe).astype(np.float32)  # [8,H,W]

def grid_to_tensor(g):
    g = np.asarray(g, dtype=int)
    H,W = g.shape
    # color one-hot (10)
    oh = np.zeros((10,H,W), dtype=np.float32)
    idx = np.clip(g, 0, 9)
    for c in range(10): oh[c] = (idx==c).astype(np.float32)
    pe = posenc_xy(H,W)  # [8,H,W]
    x = np.vstack([oh, pe])  # [18,H,W]
    return torch.from_numpy(x[None])  # [1,18,H,W]

ENC_IN_CH  = 26   # 10 color one-hot + 16 pos-enc (x/y, 4 freqs, sin+cos)
ENC_HID    = 64
ARC_ENCODER = TinyARCEncoder(in_ch=ENC_IN_CH, hid=ENC_HID, depth=3).eval()

@torch.no_grad()
def encode_grid(g):
    x = grid_to_tensor(g)
    h = ARC_ENCODER(x)          # [1,D,H,W]
    D,H,W = h.shape[1], h.shape[2], h.shape[3]
    T = H*W
    return h.view(1, D, T).permute(0,2,1).squeeze(0).cpu().float().numpy()  # [T,D]

# ---------- 2) PCA warp H->3 ----------
def fit_token_warp(mats, d=3, whiten=True):
    X = np.vstack(mats); mu = X.mean(0); Xc = X - mu
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Vt[:d,:]; Y = Xc @ pcs.T
    scales = Y.std(0, ddof=1) + 1e-8 if whiten else np.ones(d, float)
    return {"mean": mu.astype(np.float32), "pcs": pcs.astype(np.float32), "scales": scales.astype(np.float32)}

def apply_token_warp(Z, warp):
    Y = (Z - warp["mean"]) @ warp["pcs"].T
    return Y / (warp["scales"] + 1e-8)

def traces_from_grid(grid, warp):
    Z  = encode_grid(grid)           # [T,H]
    Yw = apply_token_warp(Z, warp)   # [T,3]
    return [Yw[:,0], Yw[:,1], Yw[:,2]]

# ---------- 3) NGF: smoothing, parser, matched filter, priors ----------
from ngeodesic.core.parser import moving_average, geodesic_parse_with_prior
from ngeodesic.core.matched_filter import half_sine_proto, nxcorr
from ngeodesic.core.funnel_profile import fit_radial_profile, analytic_core_template, blend_profiles, priors_from_profile, attach_projection_info

def adaptive_windows_short(T: int):
    """
    Smaller template + slightly tighter smoother to avoid flattening peaks.
    Empirically steadier on ARC grids across labels.
    """
    # target ~0.45*T, never exceed 41 (keeps kernels local)
    proto_w = max(15, min(int(0.45*T), 41))
    # light smoothing: ~T/10, clamp [3,7]
    sigma   = max(3,  min(int(T/10), 7))
    return sigma, proto_w

def parser_features(x, w):
    pos=np.maximum(0.0,x); ma=moving_average(pos, k=w); j=int(np.argmax(ma))
    halfw=max(1,w//2); area=float(pos[max(0,j-halfw):j+halfw+1].sum()); meanp=float(pos.mean())
    return np.array([j/max(1,len(x)-1), area, meanp], float)

def _mf_peak(x, proto_w):
    q = half_sine_proto(width=proto_w)
    c = nxcorr(x, q, mode="same")
    return float(np.maximum(0.0, c).max())

def _smax_over_channels(traces, proto_w):
    # mild channel-wise smoothing, then take max MF peak across 3 channels
    mx = 0.0
    for ch in traces:
        T = len(ch)
        k = min(9, max(3, T//6))
        mx = max(mx, _mf_peak(moving_average(ch, k=k), proto_w))
    return mx


def _null_circ_shift_scores(traces, proto_w, K=96):
    # Seeded RNG → deterministic null floor; more perms → tighter z
    rng = np.random.default_rng(0)
    T = len(traces[0])
    min_shift = max(proto_w, int(0.4*T))
    ks = []
    for _ in range(K):
        offs = rng.integers(low=min_shift, high=T-min_shift if T-min_shift>min_shift else T-1)
        tr = [np.roll(ch, offs) for ch in traces]
        ks.append(_smax_over_channels(tr, proto_w))
    arr = np.asarray(ks, float)
    return float(arr.mean()), float(arr.std(ddof=1) + 1e-8)



def decide_arc_with_margins(grid, warp, priors,
                            z_min=1.7, rel_floor=0.55, alpha=0.16, beta_s=0.54, q_s=2.0,
                            delta_margin=0.08):
    """
    Evaluate ALL priors, pick the one with highest z (or smax), require margin over 2nd best.
    If best < z_min or (best - second) < delta_margin -> ABSTAIN.
    """
    traces = traces_from_grid(grid, warp)
    T = len(traces[0]); _, proto_w = adaptive_windows_short(T)

    scored = []
    for name, pr in priors.items():
        ok, info = wdd_prior_pass_scored(
            traces, pr, z_min=z_min, rel_floor=rel_floor, alpha=alpha, beta_s=beta_s, q_s=q_s
        )
        # We record even if not ok so we can compare margins
        scored.append((name, info.get("z",-999), info.get("smax",0.0), ok, info))

    # sort by z then smax
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None

    best_name, best_z, best_s, best_ok, best_info = best
    sec_z = second[1] if second else -999

    # require actual PASS (z ≥ z_min) and margin over next-best
    if best_z < z_min:  # weak evidence
        return False, {"which_prior":"unknown", "reason":"weak_evidence", **best_info}
    if (best_z - sec_z) < delta_margin:
        return False, {"which_prior":"ambiguous", "reason":"no_margin", "best":(best_name,best_z), "second":(second[0],sec_z), **best_info}

    best_info["which_prior"] = best_name
    return True, best_info

def build_priors_feature_MFpeak(warp, grids, proto_w=96):
    """
    Build priors from ARC grids with per-trace adaptive windows.
    For each grid:
      - T = H*W
      - k_smooth = min(9, max(3, T//6))
      - w_proto  = min(proto_w, max(13, int(0.7*T)))   # never exceed signal length
    """
    from ngeodesic.core.parser import moving_average
    from ngeodesic.core.matched_filter import half_sine_proto, nxcorr
    from ngeodesic.core.funnel_profile import (
        fit_radial_profile, analytic_core_template, blend_profiles,
        priors_from_profile, attach_projection_info
    )
    import numpy as np

    def parser_features(x, w):
        pos = np.maximum(0.0, x)
        ma  = moving_average(pos, k=w)
        j   = int(np.argmax(ma))
        halfw = max(1, w//2)
        area  = float(pos[max(0,j-halfw): j+halfw+1].sum())
        meanp = float(pos.mean())
        return np.array([j/max(1,len(x)-1), area, meanp], float)

    def mf_peak(x, w_proto):
        q = half_sine_proto(width=w_proto)
        c = nxcorr(x, q, mode="same")
        return float(np.maximum(0.0, c).max())

    F, Zs = [], []
    for g in grids:
        # traces
        ch = traces_from_grid(g, warp)  # 3 channels
        T  = len(ch[0])
        # per-trace adaptive windows
        k_smooth = min(7, max(3, T//8))            # slightly tighter
        w_adapt  = min(proto_w, max(15, int(0.5*T)))  # narrower proto 

        # smooth + features
        S = [moving_average(c, k=k_smooth) for c in ch]
        for s in S:
            F.append(parser_features(s, w=w_adapt))
            Zs.append(mf_peak(s, w_adapt))

    F  = np.asarray(F); Zs = np.asarray(Zs)
    center  = np.median(F[:,:2], axis=0)
    R       = np.linalg.norm(F[:,:2] - center[None,:], axis=1)
    r_grid, z_data = fit_radial_profile(R, Zs, n_r=220, fit_quantile=0.70)
    z_core  = analytic_core_template(r_grid, k=0.16, p=1.7, r0_frac=0.16)
    z_blend = blend_profiles(z_data, z_core, blend_core=0.22)
    pri     = priors_from_profile(r_grid, z_blend)
    proj    = {"mean": np.zeros(3), "pcs": np.eye(3), "scales": np.ones(3), "center": center.astype(float)}
    return attach_projection_info(pri, proj)

def wdd_prior_pass_scored(traces, priors,
                          *, z_min=1.6, rel_floor=0.50,  # slightly softer default
                          alpha=0.16, beta_s=0.54, q_s=2.0,
                          null_K=96):
    """
    Prior + evidence:
      1) run geodesic_parse_with_prior to apply funnel gates
      2) compute smax (MF peak) on the accepted traces/windows
      3) null via circular shifts -> z-score
      4) return PASS/ABSTAIN with smax, z
    """
    T = len(traces[0])
    if T < 6:
        return False, {"reason":"too_short","T":T}

    sigma, proto_w = adaptive_windows_short(T)

    keep, order = geodesic_parse_with_prior(
        traces, priors=priors, sigma=sigma, proto_width=proto_w,
        z=z_min, rel_floor=rel_floor, alpha=alpha, beta_s=beta_s, q_s=q_s
    )

    # Fallback retry: if prior gates clipped everything, relax a notch once.
    if not keep:
        keep_retry, order_retry = geodesic_parse_with_prior(
            traces, priors=priors, sigma=sigma, proto_width=proto_w,
            z=max(1.2, z_min-0.2), rel_floor=max(0.40, rel_floor-0.10),
            alpha=max(0.10, alpha-0.04), beta_s=max(0.40, beta_s-0.10), q_s=q_s
        )
        if keep_retry:
            keep, order = keep_retry, order_retry
        else:
            return False, {"keep":[], "order":order, "sigma":sigma, "proto_w":proto_w,
                           "smax":0.0, "z":-999, "reason":"prior_clip"}
    # evidence score on the original (not shifted)
    s_obs = _smax_over_channels(traces, proto_w)
    mu0, sd0 = _null_circ_shift_scores(traces, proto_w, K=null_K)
    z = (s_obs - mu0) / sd0

    ok = (z >= z_min)  # require actual evidence above null
    info = {"keep":keep, "order":order, "sigma":sigma, "proto_w":proto_w,
            "smax":s_obs, "z":z}
    return ok, info

def decide_arc_with_margins_per_prior(grid, warp, priors, per_prior_thresh,
                                      delta_margin=0.10, rel_floor=0.45):
    traces = traces_from_grid(grid, warp)
    T = len(traces[0]); _, proto_w = adaptive_windows_short(T)

    scored = []
    for name, pr in priors.items():
        # allow per-label calibration; clamp to sane band
        z_gate = max(1.2, min(per_prior_thresh.get(name, 1.6), 3.0))
        ok, info = wdd_prior_pass_scored(
            tr, priors[lab],
            z_min=per_prior_thresh.get(lab, 1.6),
            rel_floor=0.45, alpha=0.14, beta_s=0.50, q_s=1.8, null_K=96
        )
        scored.append((name, info.get("z",-999), info.get("smax",0.0), ok, info))

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best = scored[0]; second = scored[1] if len(scored)>1 else None
    best_name, best_z, _, _, best_info = best
    sec_z = second[1] if second else -999

    # require (i) above that prior’s gate and (ii) margin over runner-up
    if best_z < per_prior_thresh.get(best_name, 1.6): 
        return False, {"which_prior":"unknown", "reason":"weak_evidence", **best_info}
    if (best_z - sec_z) < delta_margin:
        return False, {"which_prior":"ambiguous", "reason":"no_margin",
                       "best":(best_name,best_z), "second":(second[0],sec_z) if second else None, **best_info}
    best_info["which_prior"] = best_name
    return True, best_info


def wdd_prior_pass(traces, priors, *, z=1.7, rel_floor=0.55, alpha=0.16, beta_s=0.54, q_s=2.0):
    T = len(traces[0])
    if T < 6: return False, {"reason":"too_short","T":T}
    sigma, proto_w = adaptive_windows_short(T)
    keep, order = geodesic_parse_with_prior(
        traces, priors=priors, sigma=sigma, proto_width=proto_w,
        z=z, rel_floor=rel_floor, alpha=alpha, beta_s=beta_s, q_s=q_s
    )
    return bool(keep), {"keep":keep, "order":order, "sigma":sigma, "proto_w":proto_w, "mode":"prior"}

from collections import Counter, defaultdict

def eval_holdout_abstains(df, labels, warp, priors, per_prior_thresh, n_per=48, family_only=True):
    reasons = Counter()
    per_label = defaultdict(Counter)
    rng = np.random.default_rng(0)
    for lab in labels:
        rows = df[df.label==lab].sample(min(n_per, (df.label==lab).sum()), random_state=999)
        for _, r in rows.iterrows():
            g = load_grid(r.grid)
            tr = traces_from_grid(g, warp)
            if family_only:
                ok, info = wdd_prior_pass_scored(
                    tr, priors[lab],
                    z_min=per_prior_thresh.get(lab, 1.7),
                    rel_floor=0.45, alpha=0.14, beta_s=0.50, q_s=1.8, null_K=48
                )
                if ok:
                    per_label[lab]["PASS"] += 1
                else:
                    per_label[lab]["ABSTAIN"] += 1
                    reasons[info.get("reason","abs_gate")] += 1
            else:
                ok, info = decide_arc_with_margins_per_prior(
                    g, warp, priors, per_prior_thresh, delta_margin=0.10, rel_floor=0.45
                )
                if ok:
                    per_label[lab]["PASS"] += 1
                else:
                    per_label[lab]["ABSTAIN"] += 1
                    reasons[info.get("reason","no_margin")] += 1
    return per_label, reasons


# ---------- 4) Build warp + priors from your labeled CSV ----------
# Sample K examples per label to fit a single shared warp (more stable),
# then build a prior per primitive family.
K_WARP = 40     # per label for warp fit
K_PRI  = 40     # per label for prior fit
rng = np.random.default_rng(0)

# collect hidden states across labels for warp
mats = []
label_to_grids_for_prior = {}
for lab in LABELS:
    rows = df[df.label==lab].sample(min(K_WARP, (df.label==lab).sum()), random_state=42)
    for _, r in rows.iterrows():
        mats.append(encode_grid(load_grid(r.grid)))
    # separate sample for prior (can overlap; ok)
    rows_p = df[df.label==lab].sample(min(K_PRI, (df.label==lab).sum()), random_state=7)
    label_to_grids_for_prior[lab] = [load_grid(x) for x in rows_p.grid.tolist()]

warp = fit_token_warp(mats, d=3, whiten=True)
print("warp fitted on", len(mats), "grids; H=", mats[0].shape[1])

priors = {}
for lab in LABELS:
    grids_for_lab = label_to_grids_for_prior[lab]
    priors[lab] = build_priors_feature_MFpeak(warp, grids_for_lab, proto_w=160)  # now safely adapted

# after you build priors, make a small holdout per label
per_prior_thresh = {}
holdout_K = 24
for lab in LABELS:
    rows = df[df.label==lab].sample(min(holdout_K, (df.label==lab).sum()), random_state=1234)
    zs = []
    for _, r in rows.iterrows():
        tr = traces_from_grid(load_grid(r.grid), warp)
        ok, info = wdd_prior_pass_scored(
            tr, priors[lab],
            z_min=-999,                # collect raw z
            rel_floor=0.45, alpha=0.14, beta_s=0.50, q_s=1.8,
            null_K=96                  # tighter null stats
        )
        zs.append(info.get("z", -999))
    z_min = float(np.quantile([z for z in zs if np.isfinite(z)], 0.85))
    per_prior_thresh[lab] = max(1.2, min(z_min, 3.0))  # clamp sanity

# priors = {}
# for lab in LABELS:
#     priors[lab] = build_priors_feature_MFpeak(warp, label_to_grids_for_prior[lab], proto_w=160)
print("built priors for:", list(priors.keys()))

# ---------- 5) Test a handful of samples per label & print a table ----------
def decide_arc(grid, guess_label=None, *, family_only=True):
    traces = traces_from_grid(grid, warp)

    if guess_label is not None and family_only:
        pr = priors[guess_label]  # <-- ensure this uses guess_label
        ok, info = wdd_prior_pass_scored(
            traces, pr,
            z_min=1.4, rel_floor=0.45,
            alpha=0.14, beta_s=0.50, q_s=1.8,
            null_K=96
        )
        return ok, {**info, "which_prior": guess_label}
    
    ok, info = decide_arc_with_margins_per_prior(
        grid, warp, priors, per_prior_thresh, delta_margin=0.10, rel_floor=0.45
    )
    return ok, info



N_TEST_PER = 6
tests = []
for lab in LABELS:
    rows = df[df.label==lab].sample(min(N_TEST_PER, (df.label==lab).sum()), random_state=123)
    for _, r in rows.iterrows():
        tests.append((lab, load_grid(r.grid)))

print(f"{'label':<18} | prior  | keep | sigma | proto_w | which_prior")
print("-"*80)
for lab, g in tests:
    ok, info = decide_arc(g, guess_label=lab)   # evaluate against its own family prior
    keep_str = ",".join(info.get("keep",[])) if info.get("keep") else "-"
    sig = info.get("sigma") if info.get("sigma") is not None else "-"
    pw  = info.get("proto_w") if info.get("proto_w") is not None else "-"
    print(f"{lab:<18} | {('PASS' if ok else 'ABSTAIN'):>6} | {keep_str:^4} | {sig:^5} | {pw:^7} | {info.get('which_prior','-')}")

# Optional: quick sanity on cross-prior robustness (try the "wrong" prior)
print("\nCross-prior sanity (first 8 rows):")
for i, (lab, g) in enumerate(tests[:8]):
    wrong = [x for x in LABELS if x!=lab][0]
    ok, info = decide_arc(g, guess_label=wrong)
    print(f"{lab:>18} vs {wrong:<18} -> {'PASS' if ok else 'ABSTAIN'}")