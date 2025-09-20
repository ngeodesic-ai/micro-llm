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
    """Smaller template + tighter smoother to keep peaks sharp and avoid flattening."""
    proto_w = max(15, min(int(0.45*T), 41))
    sigma   = max(3,  min(int(T/10), 7))
    return sigma, proto_w
def decide_arc(grid, guess_label=None, *, family_only=True):
    traces = traces_from_grid(grid, warp)

    if guess_label is not None and family_only:
        pr = priors[guess_label]  # <-- ensure this uses guess_label
        ok, info = wdd_prior_pass_scored(
            traces, pr,
            z_min=1.3, rel_floor=0.45,
            alpha=0.14, beta_s=0.50, q_s=1.8,
            null_K=30
        )
        return ok, {**info, "which_prior": guess_label}
    
    # otherwise evaluate all priors with margin (or swap to per_prior_thresh version)
    # ok, info = decide_arc_with_margins(
    #     grid, warp, priors,
    #     z_min=1.3, rel_floor=0.45,
    #     alpha=0.14, beta_s=0.50, q_s=1.8,
    #     delta_margin=0.06
    # )
    ok, info = decide_arc_with_margins_per_prior(
        grid, warp, priors, per_prior_thresh, delta_margin=0.10
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

def _mf_fallback_pass(traces, floor=0.18):
    """Optional fallback: if prior is borderline but MF peak is decent, accept."""
    T = len(traces[0]); proto_w = max(15, min(int(0.45*T), 41))
    return (_smax_over_channels(traces, proto_w) >= floor)


# NOTE: If your file defines build_priors_feature_MFpeak, consider tightening its inner smoothing/window:
#   k_smooth = min(9, max(3, T//8))
#   w_adapt  = min(proto_w, max(15, int(0.5*T)))
# NOTE: When deriving per_prior_thresh from a holdout, prefer the 0.80 quantile with larger null_K (>=128).
# Example:
#   z_min = float(np.quantile([z for z in zs if np.isfinite(z)], 0.80))
#   per_prior_thresh[lab] = max(1.2, min(z_min, 3.0))
