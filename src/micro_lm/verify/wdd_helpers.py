"""
# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Copyright 2025 Ian C. Moore (Provisional Patents #63/864,726, #63/865,437, #63/871,647 and #63/872,334)
# Email: ngeodesic@gmail.com
# Part of Noetic Geodesic Framework (NGF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from __future__ import annotations
import numpy as np, joblib, torch
from typing import Dict, Any, Sequence, Tuple, Optional
from transformers import AutoTokenizer, AutoModel

# ---- SBERT token-time traces ----
def make_trace_encoder(base="sentence-transformers/all-MiniLM-L6-v2", layer_offset=-4):
    tok = AutoTokenizer.from_pretrained(base)
    mdl = AutoModel.from_pretrained(base, output_hidden_states=True).eval()
    def _hs(text: str) -> np.ndarray:
        with torch.no_grad():
            out = mdl(**tok(text, return_tensors="pt"))
        hs = out.hidden_states
        k  = max(-(len(hs)-1), min(layer_offset, -1))
        return hs[k].squeeze(0).float().cpu().numpy()  # [T,H]
    return _hs

# ---- Token-latent PCA warp (H -> 3 channels), saved as artifact ----
def fit_token_warp(hiddens: Sequence[np.ndarray], d: int = 3, whiten=True) -> dict:
    X = np.vstack(hiddens)                      # [ΣT, H]
    mu = X.mean(0)
    Xc = X - mu
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Vt[:d,:]                              # (3,H)
    Y  = Xc @ pcs.T                             # (ΣT,3)
    scales = Y.std(0, ddof=1) + 1e-8 if whiten else np.ones(d)
    return {"mean": mu, "pcs": pcs, "scales": scales}

def apply_token_warp(Z: np.ndarray, warp: dict) -> np.ndarray:
    Y = (Z - warp["mean"]) @ warp["pcs"].T
    return Y / (warp["scales"] + 1e-8)         # [T,3]

# ---- Build / load artifacts ----
def build_and_save_warp(enc, texts, path=".artifacts/wdd_token_warp.joblib"):
    H = [enc(t) for t in texts]
    warp = fit_token_warp(H, d=3, whiten=True)
    joblib.dump(warp, path)
    return warp

def load_warp(path=".artifacts/wdd_token_warp.joblib"):
    return joblib.load(path)

# ---- Stage-11 parser (Detect + Denoise) ----
from ngeodesic.core.parser import geodesic_parse_with_prior, geodesic_parse_report_conf
from ngeodesic.core.funnel_profile import (
    fit_radial_profile, analytic_core_template, blend_profiles,
    priors_from_profile, attach_projection_info
)
from ngeodesic.core.matched_filter import half_sine_proto, nxcorr, null_threshold

# priors live in *feature space* (3 numbers/channel); use identity 3x3 projection
def identity_feature_projection():
    return {"mean": np.zeros(3), "pcs": np.eye(3), "scales": np.ones(3), "center": np.array([0.0,0.0])}

def build_priors_from_texts(enc, warp, texts, layer_offset=-4):
    # Simple manifold prior from radius of warped channels over calibration texts
    r_all, z_all = [], []
    for t in texts:
        Z = enc(t); Yw = apply_token_warp(Z, warp)      # [T,3]
        pos = np.maximum(0.0, Yw)                       # crude positive energy
        r   = np.linalg.norm(Yw[:,:2], axis=1)
        z   = pos.mean(axis=1)
        r_all.append(r); z_all.append(z)
    r_flat = np.concatenate(r_all); z_flat = np.concatenate(z_all)
    r_grid, z_data  = fit_radial_profile(r_flat, z_flat, n_r=220, fit_quantile=0.65)
    z_core          = analytic_core_template(r_grid, k=0.18, p=1.7, r0_frac=0.14)
    z_blend         = blend_profiles(z_data, z_core, blend_core=0.25)
    priors          = priors_from_profile(r_grid, z_blend)
    priors          = attach_projection_info(priors, identity_feature_projection())
    return priors

def traces_from_text(enc, warp, text) -> Sequence[np.ndarray]:
    Z = enc(text); Yw = apply_token_warp(Z, warp)       # [T,3]
    return [Yw[:,0], Yw[:,1], Yw[:,2]]

# Adaptive windows for short prompts (good defaults for DeFi)
def adaptive_windows(T: int) -> Tuple[int,int]:
    proto_w = max(11, min(int(0.6*T), 61))
    sigma   = max(3,  min(int(T/10), 9))
    return sigma, proto_w

# Strict pass/abstain (report path + CFAR absolute gate + area floor)
def ngf_pass_report(traces, *, z_abs=2.4) -> Tuple[bool, Dict[str,Any]]:
    T = len(traces[0])
    if T < 6:  # tiny sequences are unreliable
        return False, {"reason": "too_short", "T": T}
    sigma, proto_w = adaptive_windows(T)
    keep, order, dbg = geodesic_parse_report_conf(traces, sigma=sigma, proto_width=proto_w)

    def _area_ok(x, j, w=proto_w, min_area=6.0):
        pos = np.maximum(0.0, x); L=max(1, w//2)
        return float(pos[max(0, j-L): j+L+1].sum()) >= min_area

    passed = False
    if keep:
        k = keep[0]; ch = int(k); peak = dbg["channels"][k]["peak_idx"]
        if _area_ok(traces[ch], peak):
            # residual CFAR gate on the winner channel
            X = np.stack(traces, 1); Xc = X - X.mean(1, keepdims=True)
            resid = Xc[:, ch]
            q = half_sine_proto(width=proto_w)
            c = nxcorr(resid, q, mode="same")
            thr = null_threshold(resid, q, shifts=600, z=z_abs, mode="perm")
            passed = float(c.max()) >= float(thr)

    return passed, {"keep": keep, "order": order, "sigma": sigma, "proto_w": proto_w, "mode": "report", "dbg": dbg}

# Prior-aware pass/abstain (preferred for production)
def ngf_pass_prior(traces, priors, *, z=2.2, rel_floor=0.70, alpha=0.08, beta_s=0.35, q_s=2.0):
    T = len(traces[0])
    if T < 6:
        return False, {"reason": "too_short", "T": T}
    sigma, proto_w = adaptive_windows(T)
    keep, order = geodesic_parse_with_prior(
        traces, priors=priors, sigma=sigma, proto_width=proto_w,
        z=z, rel_floor=rel_floor, alpha=alpha, beta_s=beta_s, q_s=q_s
    )
    return bool(keep), {"keep": keep, "order": order, "sigma": sigma, "proto_w": proto_w, "mode": "prior"}
