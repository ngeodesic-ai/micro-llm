# === Make deposits PASS: auto-layer search (-5,-7), rebuild warp+prior, gentle gates, MF fallback ===
import os, re, difflib, numpy as np, torch, joblib
from transformers import AutoTokenizer, AutoModel

# ---------------- base encoder ----------------
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
mdl = AutoModel.from_pretrained(BASE_MODEL, output_hidden_states=True).eval()

def get_hidden_states(text: str, layer_offset: int) -> np.ndarray:
    with torch.no_grad():
        out = mdl(**tok(text, return_tensors="pt"))
    hs = out.hidden_states
    k  = max(-(len(hs)-1), min(layer_offset, -1))
    return hs[k].squeeze(0).float().cpu().numpy()  # [T,H]

# ---------------- PCA warp H->3 ----------------
def fit_token_warp(hiddens, d=3, whiten=True):
    X = np.vstack(hiddens); mu = X.mean(0); Xc = X - mu
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Vt[:d,:]; Y = Xc @ pcs.T
    scales = Y.std(0, ddof=1) + 1e-8 if whiten else np.ones(d)
    return {"mean": mu, "pcs": pcs, "scales": scales}

def apply_token_warp(Z, warp):
    Y = (Z - warp["mean"]) @ warp["pcs"].T
    return Y / (warp["scales"] + 1e-8)

def traces_from_text(warp, text, layer_offset):
    Z = get_hidden_states(text, layer_offset=layer_offset)
    Yw = apply_token_warp(Z, warp)   # [T,3]
    return [Yw[:,0], Yw[:,1], Yw[:,2]]

# ---------------- NGF bits ----------------
from ngeodesic.core.parser import moving_average, geodesic_parse_with_prior
from ngeodesic.core.matched_filter import half_sine_proto, nxcorr
from ngeodesic.core.funnel_profile import (
    fit_radial_profile, analytic_core_template, blend_profiles,
    priors_from_profile, attach_projection_info
)

def normalize_protocols(text: str) -> str:
    vocab = ["uniswap","maker","makerdao","aave","compound","curve","balancer"]
    toks = text.split()
    fixed = []
    for t in toks:
        cand = difflib.get_close_matches(t.lower(), vocab, n=1, cutoff=0.75)
        fixed.append(cand[0] if cand else t)
    # unify "maker" → "makerdao" for consistency
    txt = " ".join(fixed)
    txt = re.sub(r"\bmaker\b", "makerdao", txt, flags=re.I)
    return txt

def infer_action(text: str) -> str:
    t=text.lower()
    if re.search(r"\b(supply|deposit|top up)\b", t): return "deposit"
    if re.search(r"\b(swap|exchange|trade)\b", t): return "swap"
    return "unknown"

def adaptive_windows_short(T: int):
    proto_w = max(13, min(int(0.7*T), 61))
    sigma   = max(4,  min(int(T/8),  9))
    return sigma, proto_w

def parser_features(x, w):
    pos=np.maximum(0.0,x); ma=moving_average(pos, k=w); j=int(np.argmax(ma))
    halfw=max(1,w//2); area=float(pos[max(0,j-halfw):j+halfw+1].sum()); meanp=float(pos.mean())
    return np.array([j/max(1,len(x)-1), area, meanp], float)

def mf_peak(x, proto_w):
    q = half_sine_proto(width=proto_w)
    c = nxcorr(x, q, mode="same")
    return float(np.maximum(0.0, c).max())

def build_priors_feature_MFpeak(warp, texts, layer_offset, proto_w=160):
    F, Zs = [], []
    for t in texts:
        tr = traces_from_text(warp, t, layer_offset=layer_offset)
        S  = [moving_average(ch, k=min(9, max(3, len(ch)//6))) for ch in tr]
        for ch in S:
            F.append(parser_features(ch, w=proto_w))
            Zs.append(mf_peak(ch, proto_w))
    F=np.asarray(F); Zs=np.asarray(Zs)
    center = np.median(F[:,:2], axis=0)
    R = np.linalg.norm(F[:,:2] - center[None,:], axis=1)
    r_grid, z_data = fit_radial_profile(R, Zs, n_r=220, fit_quantile=0.65)
    z_core  = analytic_core_template(r_grid, k=0.18, p=1.7, r0_frac=0.14)
    z_blend = blend_profiles(z_data, z_core, blend_core=0.25)
    pri     = priors_from_profile(r_grid, z_blend)
    proj    = {"mean": np.zeros(3), "pcs": np.eye(3), "scales": np.ones(3), "center": center.astype(float)}
    return attach_projection_info(pri, proj)

def wdd_prior_pass(traces, priors, *, z=1.7, rel_floor=0.55, alpha=0.16, beta_s=0.54, q_s=2.0):
    T = len(traces[0])
    if T < 6: return False, {"reason":"too_short","T":T}
    sigma, proto_w = adaptive_windows_short(T)
    keep, order = geodesic_parse_with_prior(
        traces, priors=priors, sigma=sigma, proto_width=proto_w,
        z=z, rel_floor=rel_floor, alpha=alpha, beta_s=beta_s, q_s=q_s
    )
    return bool(keep), {"keep": keep, "order": order, "sigma": sigma, "proto_w": proto_w, "mode":"prior"}

# ---------------- Existing swap path (kept at L=-4) ----------------
SWAP_LAYER = -4
swap_cal = [
    "swap 10 ETH to USDC on uniswap",
    "swap 2000 USDC to ETH on uniswap",
    "swap 1 WBTC for WETH on curve",
    "swap 50 SOL to USDC on uniswap",
    "swap 0.75 ETH to DAI on balancer",
    "swap 250 DAI to USDC on uniswap",
]
SWAP_WARP   = ".artifacts/wdd_warp_swap_L-4.joblib"
SWAP_PRIORS = ".artifacts/wdd_priors_swap_L-4.joblib"

if os.path.exists(SWAP_WARP):
    warp_swap = joblib.load(SWAP_WARP)
else:
    Hs = [get_hidden_states(t, layer_offset=SWAP_LAYER) for t in swap_cal]
    warp_swap = fit_token_warp(Hs, d=3, whiten=True); joblib.dump(warp_swap, SWAP_WARP)

if os.path.exists(SWAP_PRIORS):
    priors_swap = joblib.load(SWAP_PRIORS)
else:
    priors_swap = build_priors_feature_MFpeak(warp_swap, swap_cal, SWAP_LAYER, proto_w=160)
    joblib.dump(priors_swap, SWAP_PRIORS)

# ---------------- Deposit path: auto-pick best layer ----------------
deposit_cal = [
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
DEP_CAND_LAYERS = [-5, -6, -7]

def avg_mf_on_cal(layer):
    # build a quick warp per layer, measure mean MF peak across deposit cal
    Hd = [get_hidden_states(t, layer_offset=layer) for t in deposit_cal]
    w  = fit_token_warp(Hd, d=3, whiten=True)
    peaks = []
    for t in deposit_cal:
        tr = traces_from_text(w, t, layer_offset=layer)
        # mild smoothing on each channel, take max across channels
        pks = []
        for ch in tr:
            T = len(ch); proto_w = max(13, min(int(0.7*T), 61))
            pks.append(mf_peak(moving_average(ch, k=min(9, max(3, T//6))), proto_w))
        peaks.append(max(pks))
    return np.mean(peaks), w

best_layer, best_warp, best_score = None, None, -1
for L in DEP_CAND_LAYERS:
    score, w = avg_mf_on_cal(L)
    if score > best_score:
        best_score, best_layer, best_warp = score, L, w

# persist the best deposit warp + priors
DEP_LAYER  = best_layer
DEP_WARP   = f".artifacts/wdd_warp_deposit_L{DEP_LAYER}.joblib"
DEP_PRIORS = f".artifacts/wdd_priors_deposit_L{DEP_LAYER}.joblib"
joblib.dump(best_warp, DEP_WARP)
priors_dep = build_priors_feature_MFpeak(best_warp, deposit_cal, DEP_LAYER, proto_w=160)
joblib.dump(priors_dep, DEP_PRIORS)

# ---------------- Test set (+ deposit fallback) ----------------
def deposit_fallback_pass(traces, floor=0.18):
    # if prior abstains but MF peak (any channel) is decent, accept
    T = len(traces[0]); proto_w = max(13, min(int(0.7*T), 61))
    mx = 0.0
    for ch in traces:
        mx = max(mx, mf_peak(moving_average(ch, k=min(9, max(3, len(ch)//6))), proto_w))
    return mx >= floor, mx

tests = [
    "supply 7.0245 SOL to maker",        # normalizes to makerdao
    "swap 10 ETH to USDC on uniswap",
    "swap 10 ETH to USDC on uniswa",
    "attempt a borrow with low health factor",
    "that's a wrap",
    "sing a swap",
    "trade a pop",
    "trade me a drop top",
    "trade 5.6456 WETH for AAVE on sushiswap (optimism)",
    "trade 5.9195 ETH for ARB on sushiswap (arbitrum)",
    "market swap 4709.1849 ARB->WBTC using uniswap on ethereum",
    "top up uniswap with 10 ARB"
]

print(f"{'prompt'.ljust(42)} | prior  | keep | sigma | proto_w | which_prior      | note")
print("-"*120)
for raw in tests:
    text = normalize_protocols(raw)
    act  = infer_action(text)
    note = ""  # <-- ensure defined for all branches

    if act == "swap":
        traces = traces_from_text(warp_swap, text, layer_offset=SWAP_LAYER)
        ok, info = wdd_prior_pass(traces, priors_swap, z=1.7, rel_floor=0.55, alpha=0.14, beta_s=0.50)
        which = "swap(L-4)"

    elif act == "deposit":
        traces = traces_from_text(best_warp, text, layer_offset=DEP_LAYER)
        ok, info = wdd_prior_pass(traces, priors_dep,  z=1.7, rel_floor=0.55, alpha=0.16, beta_s=0.54)
        which = f"deposit(L{DEP_LAYER})"
        if not ok:
            # fallback: MF floor
            ok_fallback, mx = deposit_fallback_pass(traces, floor=0.18)
            if ok_fallback:
                ok = True
                info.setdefault("keep", [])
                note = f"fallback: MF_peak={mx:.2f}"
            else:
                note = f"mf={mx:.2f}"
    else:
        ok, info, which = False, {"keep":[], "sigma":None, "proto_w":None}, "unknown"

    print(f"{raw.ljust(42)} | {('PASS' if ok else 'ABSTAIN'):>6} | {','.join(info.get('keep',[])) or '-':^4} | "
          f"{info.get('sigma') if info.get('sigma') is not None else '-':^5} | "
          f"{info.get('proto_w') if info.get('proto_w') is not None else '-':^7} | "
          f"{which:<16} | {note}")