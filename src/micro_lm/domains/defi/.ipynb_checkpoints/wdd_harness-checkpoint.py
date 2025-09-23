# src/micro_lm/domains/defi/wdd_harness.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os, re, difflib, numpy as np, joblib, torch
from transformers import AutoTokenizer, AutoModel
import logging, sys, os

logger = logging.getLogger("micro_lm.defi.wdd")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[WDD] %(message)s"))
    logger.addHandler(_h)
logger.propagate = False
logger.setLevel(logging.WARNING)  # quiet by default
for h in logger.handlers:
    h.setLevel(logging.WARNING)

def _is_debug(debug_flag: bool, policy: dict):
    audit = (policy or {}).get("audit") or {}
    env_on = os.getenv("MICRO_LM_WDD_DEBUG")
    return bool(debug_flag or audit.get("debug") or env_on)


# If you really want to call the “golden” orchestrator later, keep this import:
# from micro_lm.core.audit.orchestrator import run_wdd

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- Lazy globals to avoid re-loading each call ----
_tok = None
_mdl = None

def _load_model():
    global _tok, _mdl
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if _mdl is None:
        _mdl = AutoModel.from_pretrained(BASE_MODEL, output_hidden_states=True).eval()

def _get_hidden_states(text: str, layer_offset: int) -> np.ndarray:
    _load_model()
    with torch.no_grad():
        out = _mdl(**_tok(text, return_tensors="pt"))
    hs = out.hidden_states
    k = max(-(len(hs)-1), min(layer_offset, -1))
    return hs[k].squeeze(0).float().cpu().numpy()  # [T,H]

def _fit_token_warp(hiddens, d=3, whiten=True):
    X = np.vstack(hiddens)
    mu = X.mean(0); Xc = X - mu
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Vt[:d,:]
    Y = Xc @ pcs.T
    scales = Y.std(0, ddof=1) + 1e-8 if whiten else np.ones(d)
    return {"mean": mu, "pcs": pcs, "scales": scales}

def _apply_token_warp(Z, warp):  # Z:[T,H]
    Y = (Z - warp["mean"]) @ warp["pcs"].T  # [T,3]
    return Y / (warp["scales"] + 1e-8)

def _traces_from_text(warp, text, layer_offset) -> List[np.ndarray]:
    Z = _get_hidden_states(text, layer_offset=layer_offset)
    Yw = _apply_token_warp(Z, warp)   # [T,3]
    return [Yw[:,0], Yw[:,1], Yw[:,2]]

# ---- NGF helpers ----
from ngeodesic.core.parser import moving_average, geodesic_parse_with_prior
from ngeodesic.core.matched_filter import half_sine_proto, nxcorr
from ngeodesic.core.funnel_profile import (
    fit_radial_profile, analytic_core_template, blend_profiles,
    priors_from_profile, attach_projection_info
)

def _normalize_protocols(text: str) -> str:
    vocab = ["uniswap","maker","makerdao","aave","compound","curve","balancer","lido","yearn"]
    toks = text.split()
    fixed = []
    for t in toks:
        cand = difflib.get_close_matches(t.lower(), vocab, n=1, cutoff=0.75)
        fixed.append(cand[0] if cand else t)
    txt = " ".join(fixed)
    txt = re.sub(r"\bmaker\b", "makerdao", txt, flags=re.I)
    return txt

def _infer_action(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(supply|deposit|top up|top-up)\b", t): return "deposit"
    if re.search(r"\b(swap|exchange|trade)\b", t): return "swap"
    return "unknown"

def _adaptive_windows_short(T: int) -> Tuple[int,int]:
    proto_w = max(13, min(int(0.7*T), 61))
    sigma   = max(4,  min(int(T//8),  9))
    return sigma, proto_w

def _parser_features(x, w):
    pos = np.maximum(0.0, x)
    ma  = moving_average(pos, k=w)
    j   = int(np.argmax(ma))
    halfw = max(1, w//2)
    area  = float(pos[max(0,j-halfw):j+halfw+1].sum())
    meanp = float(pos.mean())
    return np.array([j/max(1,len(x)-1), area, meanp], float)

def _mf_peak(x, proto_w):
    q = half_sine_proto(width=proto_w)
    c = nxcorr(x, q, mode="same")
    return float(np.maximum(0.0, c).max())

def _build_priors_feature_MFpeak(warp, texts, layer_offset, proto_w=160):
    F, Zs = [], []
    for t in texts:
        tr = _traces_from_text(warp, t, layer_offset=layer_offset)
        S  = [moving_average(ch, k=min(9, max(3, len(ch)//6))) for ch in tr]
        for ch in S:
            F.append(_parser_features(ch, w=proto_w))
            Zs.append(_mf_peak(ch, proto_w))
    F = np.asarray(F); Zs = np.asarray(Zs)
    center = np.median(F[:,:2], axis=0)
    R = np.linalg.norm(F[:,:2] - center[None,:], axis=1)
    r_grid, z_data = fit_radial_profile(R, Zs, n_r=220, fit_quantile=0.65)
    z_core  = analytic_core_template(r_grid, k=0.18, p=1.7, r0_frac=0.14)
    z_blend = blend_profiles(z_data, z_core, blend_core=0.25)
    pri     = priors_from_profile(r_grid, z_blend)
    proj    = {"mean": np.zeros(3), "pcs": np.eye(3), "scales": np.ones(3), "center": center.astype(float)}
    return attach_projection_info(pri, proj)

def _wdd_prior_pass(traces, priors, *, z=1.7, rel_floor=0.55, alpha=0.16, beta_s=0.54, q_s=2.0):
    T = len(traces[0])
    if T < 6:
        return False, {"reason":"too_short","T":T}
    sigma, proto_w = _adaptive_windows_short(T)
    keep, order = geodesic_parse_with_prior(
        traces, priors=priors, sigma=sigma, proto_width=proto_w,
        z=z, rel_floor=rel_floor, alpha=alpha, beta_s=beta_s, q_s=q_s
    )
    return bool(keep), {"keep": keep, "order": order, "sigma": sigma, "proto_w": proto_w}

# ---- Calibrations ----
SWAP_LAYER = -4
SWAP_WARP   = ".artifacts/wdd_warp_swap_L-4.joblib"
SWAP_PRIORS = ".artifacts/wdd_priors_swap_L-4.joblib"
SWAP_CAL = [
    "swap 10 ETH to USDC on uniswap",
    "swap 2000 USDC to ETH on uniswap",
    "swap 1 WBTC for WETH on curve",
    "swap 50 SOL to USDC on uniswap",
    "swap 0.75 ETH to DAI on balancer",
    "swap 250 DAI to USDC on uniswap",
]

DEP_CAND_LAYERS = [-5, -6, -7]
DEP_CAL = [
    "supply 7.0245 SOL to makerdao",
    "deposit 3 WBTC into vault",
    "supply 150 USDC to aave",
    "deposit 2 ETH to compound",
    "supply 0.5 WETH to makerdao",
    "deposit 200 DAI into vault",
    "supply 10 SOL to aave",
    "deposit 25 USDC to makerdao",
    "supply 3 ETH to makerdao",
    "top up lido with 10671 USDC on solana — safe mode",
    "top up curve with 6.5818 AVAX on polygon — ok with higher gas",
    "top up yearn with 31.7832 MATIC on base",
]

def _load_or_fit_swap():
    if os.path.exists(SWAP_WARP):
        warp = joblib.load(SWAP_WARP)
    else:
        Hs = [_get_hidden_states(t, layer_offset=SWAP_LAYER) for t in SWAP_CAL]
        warp = _fit_token_warp(Hs, d=3, whiten=True); joblib.dump(warp, SWAP_WARP)
    if os.path.exists(SWAP_PRIORS):
        pri = joblib.load(SWAP_PRIORS)
    else:
        pri = _build_priors_feature_MFpeak(warp, SWAP_CAL, SWAP_LAYER, proto_w=160)
        joblib.dump(pri, SWAP_PRIORS)
    return warp, pri

def _best_deposit_layer():
    best_layer, best_warp, best_score = None, None, -1.0
    for L in DEP_CAND_LAYERS:
        Hd = [_get_hidden_states(t, layer_offset=L) for t in DEP_CAL]
        w  = _fit_token_warp(Hd, d=3, whiten=True)
        peaks = []
        for t in DEP_CAL:
            tr = _traces_from_text(w, t, layer_offset=L)
            pks = []
            for ch in tr:
                T = len(ch); proto_w = max(13, min(int(0.7*T), 61))
                pks.append(_mf_peak(moving_average(ch, k=min(9, max(3, T//6))), proto_w))
            peaks.append(max(pks))
        score = float(np.mean(peaks))
        if score > best_score:
            best_score, best_layer, best_warp = score, L, w
    return best_layer, best_warp

def detect(prompt: str,
           sequence: List[str],
           policy: Dict[str, Any],
           context: Dict[str, Any],
           pca_prior: str | None = None,
           debug: bool = False) -> Dict[str, Any]:   # <-- NEW arg
    """
    Return {decision, sigma, proto_w, which_prior, mf_peak, keep, debug?}
    """
    dbg = _is_debug(debug, policy)
    if dbg:
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)
        logger.debug(f"prompt={prompt!r} | seq={sequence}")

    raw = _normalize_protocols(prompt)
    act = _infer_action(raw)
    note = ""
    layer_used = None

    if act == "swap":
        warp, priors = _load_or_fit_swap()
        traces = _traces_from_text(warp, raw, layer_offset=SWAP_LAYER)
        ok, info = _wdd_prior_pass(traces, priors, z=1.7, rel_floor=0.55, alpha=0.14, beta_s=0.50)
        which = "swap(L-4)"
        layer_used = SWAP_LAYER
        # Optional: compute a numeric MF peak for logging
        try:
            T = len(traces[0]); proto_w = info.get("proto_w") or max(13, min(int(0.7*T), 61))
            mx = 0.0
            for ch in traces:
                mx = max(mx, _mf_peak(moving_average(ch, k=min(9, max(3, len(ch)//6))), proto_w))
        except Exception:
            mx = None

    elif act == "deposit":
        DEP_LAYER, best_warp = _best_deposit_layer()
        traces = _traces_from_text(best_warp, raw, layer_offset=DEP_LAYER)
        priors_dep = _build_priors_feature_MFpeak(best_warp, DEP_CAL, DEP_LAYER, proto_w=160)
        ok, info = _wdd_prior_pass(traces, priors_dep, z=1.7, rel_floor=0.55, alpha=0.16, beta_s=0.54)
        which = f"deposit(L{DEP_LAYER})"
        layer_used = DEP_LAYER
        # fallback: MF floor
        if not ok:
            T = len(traces[0]); proto_w = max(13, min(int(0.7*T), 61))
            mx = 0.0
            for ch in traces:
                mx = max(mx, _mf_peak(moving_average(ch, k=min(9, max(3, len(ch)//6))), proto_w))
            if mx >= 0.18:
                ok = True
                info.setdefault("keep", [])
                note = f"fallback: MF_peak={mx:.2f}"
        else:
            # compute mx for logging even on PASS
            try:
                mx = 0.0
                T = len(traces[0]); proto_w = info.get("proto_w") or max(13, min(int(0.7*T), 61))
                for ch in traces:
                    mx = max(mx, _mf_peak(moving_average(ch, k=min(9, max(3, len(ch)//6))), proto_w))
            except Exception:
                mx = None
    else:
        ok, info, which = False, {"keep": [], "sigma": None, "proto_w": None}, "unknown"

    # ---- logging
    keep_list = info.get("keep", [])
    if dbg:
        logger.debug(
            f"act={act} layer={layer_used} sigma={info.get('sigma')} proto_w={info.get('proto_w')} "
            f"mf_peak={None if 'mx' not in locals() else mx} keep_len={len(keep_list) if keep_list is not None else 0} "
            f"decision={'PASS' if ok else 'ABSTAIN'}"
        )
        if note:
            logger.debug(note)



    # Numeric mf_peak for JSON (avoid note duplication in quickstart)
    mf_val = None
    if 'mx' in locals() and isinstance(mx, (int, float)):
        mf_val = float(mx)

    out = {
        "decision": "PASS" if ok else "ABSTAIN",
        "sigma": info.get("sigma"),
        "proto_w": info.get("proto_w"),
        "which_prior": which,
        "mf_peak": mf_val,            # numeric preferred; quickstart will format note
        "keep": keep_list or [],
    }

    if dbg:
        out["debug"] = {
            "raw": raw,
            "action": act,
            "layer": layer_used,
            "note": note,
        }

    return out

