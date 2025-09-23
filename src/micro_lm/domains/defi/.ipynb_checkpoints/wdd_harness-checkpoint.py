# src/micro_lm/domains/defi/wdd_harness.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os, re, difflib, numpy as np, joblib, torch
from dataclasses import dataclass, field
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

@dataclass
class PrimitiveSpec:
    name: str
    # Either set a fixed layer OR a list of candidates to search over
    default_layer: int | None = None
    candidate_layers: list[int] = field(default_factory=list)
    # A small set of calibration prompts representative of the primitive
    calibrations: list[str] = field(default_factory=list)
    # Optional persisted artifacts
    warp_path_tpl: str | None = None      # e.g. ".artifacts/wdd_warp_{name}_L{L}.joblib"
    priors_path_tpl: str | None = None    # e.g. ".artifacts/wdd_priors_{name}_L{L}.joblib"
    # NGF thresholds (can be tuned per primitive)
    z: float = 1.7
    rel_floor: float = 0.55
    alpha: float = 0.16
    beta_s: float = 0.54
    # MF fallback threshold (if prior PASS fails)
    mf_floor: float = 0.18



PRIMS = {
    "swap": PrimitiveSpec(
        name="swap",
        default_layer=-4,
        calibrations=[
            "swap 10 ETH to USDC on uniswap",
            "swap 2000 USDC to ETH on uniswap",
            "swap 1 WBTC for WETH on curve",
            "swap 50 SOL to USDC on uniswap",
            "swap 0.75 ETH to DAI on balancer",
            "swap 250 DAI to USDC on uniswap",
        ],
        warp_path_tpl=".artifacts/wdd_warp_{name}_L{L}.joblib",
        priors_path_tpl=".artifacts/wdd_priors_{name}_L{L}.joblib",
        z=1.7, rel_floor=0.55, alpha=0.14, beta_s=0.50, mf_floor=0.18
    ),
    "deposit": PrimitiveSpec(
        name="deposit",
        candidate_layers=[-5, -6, -7],
        calibrations=[
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
        ],
        z=1.7, rel_floor=0.55, alpha=0.16, beta_s=0.54, mf_floor=0.18
    ),
    "withdraw": PrimitiveSpec(
        name="withdraw", candidate_layers=[-5, -6, -7],
        calibrations=[
            "withdraw 3 ETH from aave",
            "withdraw 1200 USDC from makerdao",
            "withdraw 0.4 WBTC from compound",
            "withdraw 5 SOL from vault",
        ]
    ),
    "stake": PrimitiveSpec(
        name="stake", candidate_layers=[-5, -6, -7],
        calibrations=["stake 2 ETH in lido", "stake 1000 USDC in curve", "stake 5 SOL in aave"]
    ),
    "unstake": PrimitiveSpec(
        name="unstake", candidate_layers=[-5, -6, -7],
        calibrations=["unstake 2 ETH from lido", "unstake 500 USDC from curve"]
    ),
    "borrow": PrimitiveSpec(
        name="borrow", candidate_layers=[-5, -6, -7],
        calibrations=["borrow 300 DAI from aave", "borrow 0.2 WBTC from makerdao"]
    ),
    "repay": PrimitiveSpec(
        name="repay", candidate_layers=[-5, -6, -7],
        calibrations=["repay 200 DAI on aave", "repay 0.1 WBTC on makerdao"]
    ),
    "claim_rewards": PrimitiveSpec(
        name="claim_rewards", candidate_layers=[-5, -6, -7],
        calibrations=["claim rewards on aave", "claim liquidity mining rewards on curve"]
    ),
}

def _adaptive_proto(T: int) -> tuple[int, int]:
    sigma = max(4, min(int(T // 8), 9))
    proto_w = max(13, min(int(0.7 * T), 61))
    return sigma, proto_w

def _best_layer_by_peak(cals: list[str], layers: list[int]) -> tuple[int, dict]:
    """
    Pick the layer with highest mean MF peak over calibrations.
    Return (best_layer, warp) where warp is pre-fit for that layer.
    """
    best_layer, best_warp, best_score = None, None, -1.0
    for L in layers:
        H = [_get_hidden_states(t, layer_offset=L) for t in cals]
        w = _fit_token_warp(H, d=3, whiten=True)
        peaks = []
        for t in cals:
            tr = _traces_from_text(w, t, layer_offset=L)
            mx = 0.0
            for ch in tr:
                T = len(ch)
                _, pw = _adaptive_proto(T)
                mx = max(mx, _mf_peak(moving_average(ch, k=min(9, max(3, T // 6))), pw))
            peaks.append(mx)
        score = float(np.mean(peaks)) if peaks else -1.0
        if score > best_score:
            best_score, best_layer, best_warp = score, L, w
    return best_layer, best_warp

def _load_or_fit_for_spec(spec: PrimitiveSpec) -> tuple[int, dict, dict]:
    """
    Returns (layer_used, warp, priors) for a primitive spec.
    Uses persisted artifacts if paths are provided; otherwise fits on the fly.
    """
    # decide layer
    if spec.default_layer is not None:
        L = spec.default_layer
        # load/build warp
        if spec.warp_path_tpl:
            wp = spec.warp_path_tpl.format(name=spec.name, L=str(L).replace("-", "m"))
            if os.path.exists(wp):
                warp = joblib.load(wp)
            else:
                H = [_get_hidden_states(t, layer_offset=L) for t in spec.calibrations]
                warp = _fit_token_warp(H, d=3, whiten=True)
                joblib.dump(warp, wp)
        else:
            H = [_get_hidden_states(t, layer_offset=L) for t in spec.calibrations]
            warp = _fit_token_warp(H, d=3, whiten=True)
    else:
        # choose best among candidates
        L, warp = _best_layer_by_peak(spec.calibrations, spec.candidate_layers)

    # load/build priors
    if spec.priors_path_tpl:
        pp = spec.priors_path_tpl.format(name=spec.name, L=str(L).replace("-", "m"))
        if os.path.exists(pp):
            pri = joblib.load(pp)
        else:
            pri = _build_priors_feature_MFpeak(warp, spec.calibrations, L, proto_w=160)
            joblib.dump(pri, pp)
    else:
        pri = _build_priors_feature_MFpeak(warp, spec.calibrations, L, proto_w=160)

    return L, warp, pri

def _run_primitive(spec: PrimitiveSpec, raw_text: str) -> tuple[bool, dict, str, int, float | None]:
    """
    Runs WDD for `spec` on `raw_text`.
    Returns (ok, info, which_prior_str, layer_used, mf_peak_val)
    """
    L, warp, pri = _load_or_fit_for_spec(spec)
    traces = _traces_from_text(warp, raw_text, layer_offset=L)
    ok, info = _wdd_prior_pass(
        traces, priors=pri,
        z=spec.z, rel_floor=spec.rel_floor, alpha=spec.alpha, beta_s=spec.beta_s
    )
    which = f"{spec.name}(L{L})"

    # compute MF peak (for logging/summary)
    try:
        mx = 0.0
        for ch in traces:
            T = len(ch); _, pw = _adaptive_proto(T)
            mx = max(mx, _mf_peak(moving_average(ch, k=min(9, max(3, T // 6))), pw))
    except Exception:
        mx = None

    # MF fallback if prior didn’t pass
    if not ok and (mx is not None) and (mx >= spec.mf_floor):
        ok = True
        info.setdefault("keep", [])
        info.setdefault("proto_w", info.get("proto_w"))
        info.setdefault("sigma", info.get("sigma"))
        info.setdefault("note", f"fallback: MF_peak={mx:.2f}")

    return ok, info, which, L, (float(mx) if isinstance(mx, (int, float)) else None)





# If you really want to call the “golden” orchestrator later, keep this import:
# from micro_lm.core.audit.orchestrator import run_wdd

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- Lazy globals to avoid re-loading each call ----
_tok = None
_mdl = None

def _seq_to_act(sequence):
    """Map canonical primitives or raw labels to WDD act space."""
    if not sequence:
        return "unknown"
    s = (sequence[0] or "").strip().lower()
    # accept both canonical primitives and raw mapper labels
    if s.endswith("_asset"):
        s = s[:-6]  # strip '_asset'
    if s in ("swap", "deposit", "withdraw"):
        return s
    return "unknown"

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



def detect(prompt: str,
           sequence: list[str],
           policy: dict,
           context: dict,
           pca_prior: str | None = None,
           debug: bool = False) -> dict:

    dbg = _is_debug(debug, policy)
    if dbg:
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)
        logger.debug(f"prompt={prompt!r} | seq={sequence}")

    raw = _normalize_protocols(prompt)
    note = ""

    # Trust mapper/runner-provided sequence for the act
    act = _seq_to_act(sequence)
    if act not in PRIMS:
        # keep regex as a last-ditch fallback if you want
        act = _infer_action(raw)

    if act in PRIMS:
        ok, info, which, layer_used, mf_val = _run_primitive(PRIMS[act], raw)
    else:
        ok, info, which, layer_used, mf_val = False, {"keep": [], "sigma": None, "proto_w": None}, "unknown", None, None

    if dbg:
        keep_list = info.get("keep", []) or []
        logger.debug(
            f"act={act} layer={layer_used} sigma={info.get('sigma')} proto_w={info.get('proto_w')} "
            f"mf_peak={mf_val} keep_len={len(keep_list)} decision={'PASS' if ok else 'ABSTAIN'}"
        )
        if info.get("note"):
            logger.debug(info["note"])

    out = {
        "decision": "PASS" if ok else "ABSTAIN",
        "sigma": info.get("sigma"),
        "proto_w": info.get("proto_w"),
        "which_prior": which,
        "mf_peak": mf_val,
        "keep": info.get("keep") or [],
    }
    if dbg:
        out["debug"] = {
            "raw": raw,
            "action": act,
            "layer": layer_used,
            "note": info.get("note"),
        }
    return out




# ---- Calibrations ----
# SWAP_LAYER = -4
# SWAP_WARP   = ".artifacts/wdd_warp_swap_L-4.joblib"
# SWAP_PRIORS = ".artifacts/wdd_priors_swap_L-4.joblib"
# SWAP_CAL = [
#     "swap 10 ETH to USDC on uniswap",
#     "swap 2000 USDC to ETH on uniswap",
#     "swap 1 WBTC for WETH on curve",
#     "swap 50 SOL to USDC on uniswap",
#     "swap 0.75 ETH to DAI on balancer",
#     "swap 250 DAI to USDC on uniswap",
# ]

# DEP_CAND_LAYERS = [-5, -6, -7]
# DEP_CAL = [
#     "supply 7.0245 SOL to makerdao",
#     "deposit 3 WBTC into vault",
#     "supply 150 USDC to aave",
#     "deposit 2 ETH to compound",
#     "supply 0.5 WETH to makerdao",
#     "deposit 200 DAI into vault",
#     "supply 10 SOL to aave",
#     "deposit 25 USDC to makerdao",
#     "supply 3 ETH to makerdao",
#     "top up lido with 10671 USDC on solana — safe mode",
#     "top up curve with 6.5818 AVAX on polygon — ok with higher gas",
#     "top up yearn with 31.7832 MATIC on base",
# ]

# def _load_or_fit_swap():
#     if os.path.exists(SWAP_WARP):
#         warp = joblib.load(SWAP_WARP)
#     else:
#         Hs = [_get_hidden_states(t, layer_offset=SWAP_LAYER) for t in SWAP_CAL]
#         warp = _fit_token_warp(Hs, d=3, whiten=True); joblib.dump(warp, SWAP_WARP)
#     if os.path.exists(SWAP_PRIORS):
#         pri = joblib.load(SWAP_PRIORS)
#     else:
#         pri = _build_priors_feature_MFpeak(warp, SWAP_CAL, SWAP_LAYER, proto_w=160)
#         joblib.dump(pri, SWAP_PRIORS)
#     return warp, pri

# def _best_deposit_layer():
#     best_layer, best_warp, best_score = None, None, -1.0
#     for L in DEP_CAND_LAYERS:
#         Hd = [_get_hidden_states(t, layer_offset=L) for t in DEP_CAL]
#         w  = _fit_token_warp(Hd, d=3, whiten=True)
#         peaks = []
#         for t in DEP_CAL:
#             tr = _traces_from_text(w, t, layer_offset=L)
#             pks = []
#             for ch in tr:
#                 T = len(ch); proto_w = max(13, min(int(0.7*T), 61))
#                 pks.append(_mf_peak(moving_average(ch, k=min(9, max(3, T//6))), proto_w))
#             peaks.append(max(pks))
#         score = float(np.mean(peaks))
#         if score > best_score:
#             best_score, best_layer, best_warp = score, L, w
#     return best_layer, best_warp


# def detect(prompt: str,
#            sequence: List[str],
#            policy: Dict[str, Any],
#            context: Dict[str, Any],
#            pca_prior: str | None = None,
#            debug: bool = False) -> Dict[str, Any]:   # <-- NEW arg
#     """
#     Return {decision, sigma, proto_w, which_prior, mf_peak, keep, debug?}
#     """
#     dbg = _is_debug(debug, policy)
#     if dbg:
#         logger.setLevel(logging.DEBUG)
#         for h in logger.handlers:
#             h.setLevel(logging.DEBUG)
#         logger.debug(f"prompt={prompt!r} | seq={sequence}")

#     raw = _normalize_protocols(prompt)
#     note = ""
#     layer_used = None
    
#     # Prefer mapper/runner-provided sequence; then mapper model; finally regex fallback
#     src = "sequence"
#     act = _seq_to_act(sequence)
#     if act == "unknown":
#         src = "regex"
#         act = _infer_action(raw)

#     if act == "swap":
#         warp, priors = _load_or_fit_swap()
#         traces = _traces_from_text(warp, raw, layer_offset=SWAP_LAYER)
#         ok, info = _wdd_prior_pass(traces, priors, z=1.7, rel_floor=0.55, alpha=0.14, beta_s=0.50)
#         which = "swap(L-4)"
#         layer_used = SWAP_LAYER
#         # Optional: compute a numeric MF peak for logging
#         try:
#             T = len(traces[0]); proto_w = info.get("proto_w") or max(13, min(int(0.7*T), 61))
#             mx = 0.0
#             for ch in traces:
#                 mx = max(mx, _mf_peak(moving_average(ch, k=min(9, max(3, len(ch)//6))), proto_w))
#         except Exception:
#             mx = None

#     elif act == "deposit":
#         DEP_LAYER, best_warp = _best_deposit_layer()
#         traces = _traces_from_text(best_warp, raw, layer_offset=DEP_LAYER)
#         priors_dep = _build_priors_feature_MFpeak(best_warp, DEP_CAL, DEP_LAYER, proto_w=160)
#         ok, info = _wdd_prior_pass(traces, priors_dep, z=1.7, rel_floor=0.55, alpha=0.16, beta_s=0.54)
#         which = f"deposit(L{DEP_LAYER})"
#         layer_used = DEP_LAYER
#         # fallback: MF floor
#         if not ok:
#             T = len(traces[0]); proto_w = max(13, min(int(0.7*T), 61))
#             mx = 0.0
#             for ch in traces:
#                 mx = max(mx, _mf_peak(moving_average(ch, k=min(9, max(3, len(ch)//6))), proto_w))
#             if mx >= 0.18:
#                 ok = True
#                 info.setdefault("keep", [])
#                 note = f"fallback: MF_peak={mx:.2f}"
#         else:
#             # compute mx for logging even on PASS
#             try:
#                 mx = 0.0
#                 T = len(traces[0]); proto_w = info.get("proto_w") or max(13, min(int(0.7*T), 61))
#                 for ch in traces:
#                     mx = max(mx, _mf_peak(moving_average(ch, k=min(9, max(3, len(ch)//6))), proto_w))
#             except Exception:
#                 mx = None
#     else:
#         ok, info, which = False, {"keep": [], "sigma": None, "proto_w": None}, "unknown"

#     # ---- logging
#     keep_list = info.get("keep", [])
#     if dbg:
#         logger.debug(
#             f"act={act} layer={layer_used} sigma={info.get('sigma')} proto_w={info.get('proto_w')} "
#             f"mf_peak={None if 'mx' not in locals() else mx} keep_len={len(keep_list) if keep_list is not None else 0} "
#             f"decision={'PASS' if ok else 'ABSTAIN'}"
#         )
#         if note:
#             logger.debug(note)



#     # Numeric mf_peak for JSON (avoid note duplication in quickstart)
#     mf_val = None
#     if 'mx' in locals() and isinstance(mx, (int, float)):
#         mf_val = float(mx)

#     out = {
#         "decision": "PASS" if ok else "ABSTAIN",
#         "sigma": info.get("sigma"),
#         "proto_w": info.get("proto_w"),
#         "which_prior": which,
#         "mf_peak": mf_val,            # numeric preferred; quickstart will format note
#         "keep": keep_list or [],
#     }

#     if dbg:
#         out["debug"] = {
#             "raw": raw,
#             "action": act,
#             "layer": layer_used,
#             "note": note,
#         }

#     return out
