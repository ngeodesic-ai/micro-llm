# src/micro_lm/domains/arc/wdd_arc_harness.py
#
# Tier-2 ARC WDD harness — mirrors the DeFi harness shape but follows
# the ARC notebook template. Supports **three primitives** with ordered
# multi-primitive sequences via the baked-in parser (matched filter + peak ordering).
#
# Primitives implemented (per notebook):
#   - flip_h
#   - flip_v
#   - rotate   (90° steps; k inferred by parser/order)
#
# Public surface (kept deliberately similar to DeFi `wdd_harness.py`):
#   - PrimitiveSpec (dataclass)
#   - PRIMS (registry)
#   - arc_detect(prompt:str, grid:np.ndarray, sequence:list[str]|None, policy:dict, debug:bool=False) -> dict
#       * uses mapper-provided `sequence` if available; otherwise runs ARC regex fallback
#       * produces `aux.stage11.wdd`, `wdd_summary`, and (if family mode) `plan.sequence`
#   - run_arc_wdd(prompt:str, grid:np.ndarray, policy:dict, sequence:list[str]|None=None, debug:bool=False) -> dict
#       * thin wrapper that aligns to INTERFACE.md output contract
#
# Notes
# - We fit/restore a tiny PCA(3) token-warp per primitive and compute energy traces
#   from the ARC grid → latent paths (as in the notebook). The exact trace wiring
#   is kept in helpers `_traces_from_grid()` and `_build_priors_feature_MFpeak()` to
#   preserve reproduction.
# - Parser semantics: we detect channels that PASS gates, compute matched-filter peaks,
#   and sort by peak time t*. This yields an ordered multi-primitive sequence.
# - Family vs Detector: in ARC we usually want **family** so WDD can produce an order.
#   `policy.audit.mode` controls this exactly like DeFi.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import os, sys, logging, re
from collections import defaultdict
import numpy as np
import joblib

# Optional: torch is only needed if your encoder uses it; left for parity with DeFi harness
try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = None  # type: ignore

logger = logging.getLogger("micro_lm.arc.wdd")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[WDD-ARC] %(message)s"))
    logger.addHandler(_h)
logger.propagate = False
logger.setLevel(logging.WARNING)
for h in logger.handlers:
    h.setLevel(logging.WARNING)


def _is_debug(debug_flag: bool, policy: dict) -> bool:
    audit = (policy or {}).get("audit") or {}
    env_on = os.getenv("MICRO_LM_WDD_DEBUG")
    return bool(debug_flag or audit.get("debug") or env_on)




# --------------------------- Config (edit if needed) ------------------------------------------
ARC_MAPPER_PATH = locals().get("ARC_MAPPER_PATH", ".artifacts/arc_mapper.joblib")
ARC_LABELS_CSV  = locals().get("ARC_LABELS_CSV",  "")  # optional: labeled CSV to build centroids
ARC_CLASS_NAMES = locals().get("ARC_CLASS_NAMES", [])  # optional static list; else inferred

# Latent-path + parser settings
TOP_K       = 3
TRACE_LEN   = 128
ALPHA_MAX   = 2.2
SMOOTH_WIN  = 9
NULL_SHIFTS = 64

# Bring-up family-local thresholds (tighten later)
F_TAU = {
    "flip":   {"tau_rel": 0.05, "tau_corr": 0.25, "z_abs": 0.5, "area": 0.0},
    "rotate": {"tau_rel": 0.05, "tau_corr": 0.20, "z_abs": 0.30, "area": 0.0},
    "other":  {"tau_rel": 0.05, "tau_corr": 0.25, "z_abs": 0.50, "area": 0.0},
}

# Canonical mini-templates (used only if coef_/centroids absent)
ARC_CANON = {
    "flip_h": "flip the grid horizontally",
    "flip_v": "flip the grid vertically",
    "rotate": "rotate the grid 90 degrees clockwise",
    "rot90":  "rotate the grid 90 degrees clockwise",
    "rotate_cw": "rotate the grid 90 degrees clockwise",
    "rotate_ccw": "rotate the grid 90 degrees counterclockwise",
}

ARC_CLASS_NAMES = ["rotate", "flip_h", "flip_v"] 

# --------------------------------- Helpers -----------------------------------------------------
def _normalize(v):
    v = np.asarray(v, float).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-12)

def _unwrap_mapper(mapper):
    """Return (encoder_step, clf_step) from a sklearn Pipeline-like mapper."""
    enc = clf = None
    if hasattr(mapper, "named_steps"):
        steps = dict(mapper.named_steps)
        # guess encoder by capability & name
        for name, step in steps.items():
            if hasattr(step, "transform") and ("sbert" in str(step).lower() or "sentence" in str(step).lower()):
                enc = step; break
        clf = list(steps.values())[-1]
    else:
        # best-effort (non-pipeline)
        if hasattr(mapper, "transform"): enc = mapper
        if hasattr(mapper, "predict_proba") or hasattr(mapper, "decision_function"): clf = mapper
    # calibrated wrappers sometimes store base_estimator
    base = getattr(clf, "base_estimator", None)
    if base is not None:
        clf = base
    return enc, clf

def _get_classes(mapper):
    """
    Safely extract class names from the final estimator in the mapper pipeline.
    Handles numpy arrays and missing attributes; falls back to ARC_CLASS_NAMES.
    """
    classes = None

    # Try the last pipeline step first
    if hasattr(mapper, "named_steps"):
        last = list(mapper.named_steps.values())[-1]
        classes = getattr(last, "classes_", None)

    # Fallback: mapper itself
    if classes is None:
        classes = getattr(mapper, "classes_", None)

    # If still missing or empty, use configured fallback
    if classes is None or (hasattr(classes, "__len__") and len(classes) == 0):
        classes = ARC_CLASS_NAMES

    # Final sanity: must have something
    if classes is None or (hasattr(classes, "__len__") and len(classes) == 0):
        raise ValueError(
            "Could not infer class names from mapper. "
            "Set ARC_CLASS_NAMES to a non-empty list (e.g., "
            "['rotate','flip_h','flip_v','recolor'])."
        )

    # Convert numpy arrays to a Python list of strings
    if hasattr(classes, "tolist"):
        classes = classes.tolist()
    return [str(c) for c in classes]

def _embed_texts(mapper, texts):
    """Encode texts with the SBERT encoder inside the mapper; fallback to SentenceTransformer."""
    enc, _ = _unwrap_mapper(mapper)
    if enc is not None and hasattr(enc, "transform"):
        E = enc.transform(list(texts))
        return np.asarray(E)
    # fallback
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return np.asarray(model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))

def _classifier_for_embed(mapper):
    """Return the last estimator (accepts embeddings). May be calibrated."""
    if hasattr(mapper, "named_steps"):
        return list(mapper.named_steps.values())[-1]
    return mapper

def _classifier_coef(clf):
    """Try to recover coef_ matrix from linear models or wrapped models."""
    co = getattr(clf, "coef_", None)
    if co is not None: return np.asarray(co, float)
    # calibrated form
    for attr in ("estimator_", "estimator", "base_estimator_", "base_estimator"):
        inner = getattr(clf, attr, None)
        if inner is not None and hasattr(inner, "coef_"):
            return np.asarray(inner.coef_, float)
    # OneVsRest
    if hasattr(clf, "estimators_"):
        rows = []
        for est in clf.estimators_:
            got = _classifier_coef(est)
            if got is None: return None
            rows.append(got.ravel())
        return np.vstack(rows) if rows else None
    # CalibratedClassifierCV (list of per-class calibrators)
    if hasattr(clf, "calibrated_classifiers_"):
        rows = []
        for cc in clf.calibrated_classifiers_:
            est = getattr(cc, "estimator", None)
            if est is not None and hasattr(est, "coef_"):
                rows.append(est.coef_.ravel())
        return np.vstack(rows) if rows else None
    return None

def _class_centroids_from_csv(mapper, labels_csv, class_names):
    """Optional: build SBERT centroids per class from a labeled CSV."""
    if not labels_csv or not os.path.exists(labels_csv): return None
    import pandas as pd
    df = pd.read_csv(labels_csv)
    if not {"prompt","label"}.issubset(df.columns): return None
    df = df.dropna(subset=["prompt","label"]).copy()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["label"]  = df["label"].astype(str).str.strip()
    if df.empty: return None
    # embed in chunks
    B = 256; Es = []
    for i in range(0, len(df), B):
        Es.append(_embed_texts(mapper, df["prompt"].iloc[i:i+B].tolist()))
    X = np.vstack(Es)
    y = df["label"].tolist()
    cents = {}
    for c in class_names:
        idx = [i for i, yy in enumerate(y) if yy == c]
        if idx:
            m = X[idx].mean(axis=0)
            cents[c] = _normalize(m)
    return cents if cents else None

def _proto_directions(mapper, class_names):
    """Return dict label->unit direction. Try coef_, else CSV centroids, else canonical templates."""
    enc, clf = _unwrap_mapper(mapper)
    # 1) coef_
    co = _classifier_coef(clf)
    if co is not None:
        cls = getattr(clf, "classes_", class_names)
        cls = [str(c) for c in cls]
        P = {}
        for c in class_names:
            if c in cls:
                j = cls.index(c)
                P[c] = _normalize(co[j])
        if len(P) == len(class_names):
            return P
    # 2) centroids
    cents = _class_centroids_from_csv(mapper, ARC_LABELS_CSV, class_names)
    if cents:
        mu = _normalize(np.mean(np.vstack(list(cents.values())), axis=0))
        return {c: _normalize(cents[c] - mu) for c in class_names if c in cents}
    # 3) canonical templates
    T = [ARC_CANON.get(c, c) for c in class_names]
    E = _embed_texts(mapper, T)
    mu = _normalize(E.mean(axis=0))
    return {c: _normalize(E[i] - mu) for i, c in enumerate(class_names)}

def _half_sine_path(e0, p_hat, T=TRACE_LEN, alpha_max=ALPHA_MAX):
    p_hat = _normalize(p_hat)
    alphas = np.sin(np.linspace(0.0, np.pi, T)) * alpha_max
    return e0[None, :] + alphas[:, None] * p_hat[None, :]

def _smooth(x, w=SMOOTH_WIN):
    if w <= 1: return x
    k = np.ones(w, float) / float(w)
    return np.convolve(np.asarray(x, float), k, mode="same")

def _energies(Y, p, c=None):
    p = _normalize(p)
    D = Y.shape[1]
    if c is None: c = np.zeros(D, float)
    Z = Y - c[None, :]
    s = Z @ p
    E_par  = s**2
    E_tot  = np.sum(Z*Z, axis=1)
    E_perp = E_tot - E_par
    return _smooth(E_par), _smooth(E_perp)

def _nxcorr(env, q):
    env = np.asarray(env, float)
    q   = np.asarray(q, float)
    q   = (q - q.mean()); qn = np.linalg.norm(q) + 1e-12
    L   = len(q)
    pad = L // 2
    xx  = np.pad(env, (pad, pad), mode="edge")
    C   = np.zeros_like(env)
    for t in range(len(env)):
        seg = xx[t:t+L]; seg = seg - seg.mean()
        C[t] = float(np.dot(seg, q) / ((np.linalg.norm(seg) + 1e-12) * qn))
    return C

def _matched_filter_score(env):
    L = max(50, min(80, len(env)-4))
    t = np.arange(L)
    q = np.sin(np.pi * (t / (L-1)))
    C = _nxcorr(env, q)
    t_star = int(np.argmax(C))
    # half-max window
    xm = float(np.max(env)); half = 0.5 * xm
    a = t_star
    while a > 0 and env[a] > half: a -= 1
    b = t_star
    while b < len(env)-1 and env[b] > half: b += 1
    area = float(np.sum(env[max(0,a):min(len(env)-1,b)+1]))
    return float(C[t_star]), t_star, area, (int(a), int(b)), C

def _circular_shifts(x, K=NULL_SHIFTS):
    n = len(x)
    if n <= 2: return []
    shifts = np.random.choice(np.arange(1, n), size=min(K, n-1), replace=False)
    return [np.roll(x, s) for s in shifts]

def _absolute_null_z(env):
    nulls = _circular_shifts(env, NULL_SHIFTS)
    if not nulls: return 0.0
    peaks = [_matched_filter_score(ne)[0] for ne in nulls]
    mu, sd = float(np.mean(peaks)), float(np.std(peaks) + 1e-9)
    cm, *_ = _matched_filter_score(env)
    return (cm - mu) / sd

def _exclusive_residual(E_list):
    """Exclusive positive residual across siblings (QR projection; fallback to mean subtraction)."""
    if len(E_list) == 1:
        return [np.clip((E_list[0] - np.mean(E_list[0])) / (np.std(E_list[0]) + 1e-9), 0, None)]
    Z = np.vstack([ (e - np.mean(e)) / (np.std(e) + 1e-9) for e in E_list ])  # K x T
    Rex = []
    for k in range(Z.shape[0]):
        others = [j for j in range(Z.shape[0]) if j != k]
        A = Z[others].T  # T x (K-1)
        try:
            Q, _ = np.linalg.qr(A, mode="reduced")
            proj = Q @ (Q.T @ Z[k])
            ex = Z[k] - proj
        except Exception:
            ex = Z[k] - np.mean(Z[others], axis=0)
        Rex.append(np.clip(ex, 0, None))
    return Rex

def _family(lbl: str) -> str:
    s = lbl.lower()
    if s.startswith("flip_"):                 return "flip"
    if "rot" in s or s in {"rotate","rot90","rotate_cw","rotate_ccw"}: return "rotate"
    if s in {"recolor","color_map","color"}: return "color"
    return "other"

def _parse_hints(prompt: str):
    s = prompt.lower()
    want_flip   = ("flip" in s) or ("mirror" in s)
    want_rotate = ("rotate" in s) or ("cw" in s) or ("clockwise" in s) or ("counterclockwise" in s) or ("ccw" in s) or ("right" in s) or ("left" in s)
    want_color  = ("color" in s) or ("recolor" in s) or ("colour" in s) or ("recolour" in s) or ("map" in s)
    orient = None
    # add synonyms in _parse_hints
    if re.search(r"\b(hori(zontal)?|mirror\s*(left|right)|left\s*↔\s*right)\b", s): orient = "flip_h"
    elif re.search(r"\b(vert(ical)?|top\s*↕\s*bottom|up\s*down)\b", s): orient = "flip_v"
    return {"flip": want_flip, "rotate": want_rotate, "color": want_color}, orient




# --------------------------------- Main audit ---------------------------------
def wdd_arc_audit(prompt: str, mapper=None, candidates=None):
    # if mapper not provided, fallback to artifact load
    if mapper is None:
        import joblib
        mapper = joblib.load(ARC_MAPPER_PATH)
    class_names = _get_classes(mapper)

    # 0) text hints
    hints, orient_hint = _parse_hints(prompt)

    # 1) embed prompt (start point e0)
    e0 = _normalize(_embed_texts(mapper, [prompt])[0])

    # 2) mapper probabilities (for shortlist + tiny flip tie-break)
    if hasattr(mapper, "predict_proba"):
        probs = mapper.predict_proba([prompt])[0]
        smap  = {c: float(p) for c, p in zip(class_names, probs)}
    elif hasattr(mapper, "decision_function"):
        z = np.asarray(mapper.decision_function([prompt])[0], float).ravel()
        ex = np.exp(z - z.max()); pr = ex / (ex.sum() + 1e-12)
        smap = {c: float(p) for c, p in zip(class_names, pr)}
    else:
        pred = str(mapper.predict([prompt])[0])
        smap = {c: (1.0 if c==pred else 0.0) for c in class_names}
    mapper_top = max(smap.items(), key=lambda kv: kv[1])[0]

    # 3) route: text-gated shortlist
    MIN_CAND_PROB = 0.05     # tiny floor to consider a class from mapper
    STRICT_TEXT_GATING = True
    
    def _family(lbl: str) -> str:
        s = lbl.lower()
        if s.startswith("flip_"):                 return "flip"
        if "rot" in s or s in {"rotate","rot90","rotate_cw","rotate_ccw"}: return "rotate"
        if s in {"recolor","color_map","color"}:  return "color"
        return "other"
    
    # families explicitly mentioned in the prompt
    allowed_fams = set()
    if hints["flip"]:   allowed_fams.add("flip")
    if hints["rotate"]: allowed_fams.add("rotate")
    if hints["color"]:  allowed_fams.add("color")
    
    # base set: keep top-k classes with non-trivial probability
    topk = sorted(smap.items(), key=lambda kv: kv[1], reverse=True)[:max(1, TOP_K)]
    base = {c for c, p in topk if p >= MIN_CAND_PROB}
    base.add(mapper_top)  # always include mapper top-1
    
    # text inclusions: make sure families explicitly requested are represented
    text_added = set()
    for fam in allowed_fams:
        for c in class_names:
            if _family(c) == fam:
                text_added.add(c)
    
    # strict text gating: if any families are mentioned, restrict to them,
    # but still allow any class with prob >= MIN_CAND_PROB
    # shortlist = (base | text_added)
    # if STRICT_TEXT_GATING and allowed_fams:
    #     shortlist = {c for c in shortlist if (_family(c) in allowed_fams) or (smap.get(c, 0.0) >= MIN_CAND_PROB)}


    if candidates:
        cand_set = set(candidates)
        labs = [c for c in class_names if c in cand_set]
        STRICT_TEXT_GATING = False  # disable gating when mapper already chose
    else:
        labs = [c for c in class_names if c in shortlist]
        if not labs:
            return {"prompt": prompt, "selected": [], "why": "no candidates", "scores": smap}


    # 4) prototypes & anchors
    P = _proto_directions(mapper, class_names)
    D = e0.shape[0]
    for k,v in list(P.items()):
        v = np.asarray(v, float).reshape(-1)
        if v.shape[0] != D:
            v = (v[:D] if v.shape[0] > D else np.pad(v, (0, D - v.shape[0])))
        P[k] = _normalize(v)
    anchors = defaultdict(lambda: np.zeros(D, float))
    cents = _class_centroids_from_csv(mapper, ARC_LABELS_CSV, class_names)
    if cents:
        for c, m in cents.items():
            mv = np.asarray(m, float).reshape(-1)
            mv = (mv[:D] if mv.shape[0] > D else np.pad(mv, (0, D - mv.shape[0])))
            anchors[c] = _normalize(mv)

    # 5) latent paths + energy/prob envelopes
    enc, clf = _unwrap_mapper(mapper)
    clf_embed = _classifier_for_embed(mapper)
    clf_cls   = getattr(clf_embed, "classes_", class_names)
    idx_map   = {str(c): i for i, c in enumerate(clf_cls)}

    per_class = {}
    fam2labs  = defaultdict(list)
    for lab in labs:
        if lab not in P: continue
        Y = _half_sine_path(e0, P[lab], TRACE_LEN, ALPHA_MAX)      # (T, D)
        E_par, E_perp = _energies(Y, P[lab], anchors[lab])
        # prob envelope along path (if classifier exposes proba over embeddings)
        if hasattr(clf_embed, "predict_proba"):
            P_path = clf_embed.predict_proba(Y)                    # (T, C)
            j = idx_map.get(lab, None)
            prob_env = _smooth(P_path[:, j], SMOOTH_WIN) if j is not None else None
        else:
            prob_env = None
        per_class[lab] = {"E_par": E_par, "E_perp": E_perp, "prob": prob_env}
        fam2labs[_family(lab)].append(lab)

    if not per_class:
        return {"prompt": prompt, "selected": [], "why": "no envelopes", "scores": smap}

    # 6) family-local residuals & scoring (rotate uses prob; flip/color use energy residual)
    for fam, labs_f in fam2labs.items():
        Elist = [per_class[k]["E_perp"] for k in labs_f]
        Rex_f = _exclusive_residual(Elist) if len(labs_f) > 0 else []
        for idx, k in enumerate(labs_f):
            if fam == "rotate" and per_class[k]["prob"] is not None:
                env = _smooth(np.clip(per_class[k]["prob"], 0, None), SMOOTH_WIN)
            else:
                ex = Rex_f[idx] if idx < len(Rex_f) else per_class[k]["E_perp"]
                env = _smooth(np.clip(ex, 0, None), SMOOTH_WIN)
            cm, t_star, area, (a,b), C = _matched_filter_score(env)
            z_abs = _absolute_null_z(env)
            per_class[k].update(dict(env=env, corr_max=cm, t_star=int(t_star),
                                     area=float(area), window=(int(a), int(b)),
                                     corr_trace=C, z_abs=float(z_abs)))

    # 7) family-wise gating then merge
    kept = []
    for fam, labs_f in fam2labs.items():
        thr = F_TAU.get(fam, F_TAU["other"])
        ranked_f = sorted(labs_f, key=lambda k: (per_class[k]["corr_max"], per_class[k]["area"]), reverse=True)
        best_cm = per_class[ranked_f[0]]["corr_max"]
        decided_f = []
        for k in ranked_f:
            r = per_class[k]
            rel_ok  = (r["corr_max"] >= (1.0 - thr["tau_rel"]) * best_cm)
            corr_ok = (r["corr_max"] >= thr["tau_corr"])
            abs_ok  = (r["z_abs"]    >= thr["z_abs"])
            area_ok = (r["area"]     >= thr["area"])
            accept  = (abs_ok or (corr_ok and area_ok))
            if accept:
                decided_f.append((k, r["t_star"]))
        if len(decided_f) > 1:
            decided_f = [(k,t) for (k,t) in decided_f if per_class[k]["corr_max"] >= (1.0 - thr["tau_rel"]) * best_cm]
        kept.extend(decided_f)

    # 8) orientation pin + tiny mapper nudge inside flip family
    flip_kept = [k for (k,_) in kept if k in ("flip_h","flip_v")]
    hints_flip_orient = orient_hint in ("flip_h","flip_v")
    if hints["flip"] and hints_flip_orient and orient_hint not in flip_kept:
        rh = per_class.get(orient_hint)
        if rh and (rh["z_abs"] >= 0.0 or rh["corr_max"] >= 0.15):
            kept = [(k,t) for (k,t) in kept if k not in ("flip_h","flip_v")]
            kept.append((orient_hint, rh["t_star"]))
    flip_kept = [k for (k,_) in kept if k in ("flip_h","flip_v")]
    if hints["flip"] and mapper_top in ("flip_h","flip_v") and flip_kept:
        chosen = flip_kept[0]
        if chosen != mapper_top:
            r_bad = per_class[chosen]; r_good = per_class.get(mapper_top)
            if r_good and (r_bad["corr_max"] - r_good["corr_max"] <= 0.05):
                kept = [(k,t) for (k,t) in kept if k not in ("flip_h","flip_v")]
                kept.append((mapper_top, r_good["t_star"]))

    # 9) order by peak time
    kept.sort(key=lambda kv: kv[1])
    sequence = [k for k,_ in kept]

    return {
        "prompt": prompt,
        "scores": dict(sorted(smap.items(), key=lambda kv: kv[1], reverse=True)),
        "candidates": labs,
        "selected": sequence,
        "per_class": {k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                          for kk, vv in v.items()
                          if kk in ["corr_max","t_star","area","window","z_abs"]}
                      for k, v in per_class.items()},
    }




def run_arc_wdd(prompt: str, grid: np.ndarray, policy: Optional[Dict[str, Any]] = None,
                sequence: Optional[List[str]] = None, mapper=None, debug: bool = False) -> Dict[str, Any]:
    policy = policy or {}
    mode = ((policy.get("audit") or {}).get("mode") or "detector")

    # Run the audit (mapper + candidates already wired in)
    res = wdd_arc_audit(prompt, mapper=mapper, candidates=sequence)

    # Ordered keep list (family ordering)
    keep_order: List[str] = res.get("selected", [])
    plan_seq = keep_order if mode == "family" else []

    # Per-class stats from the audit
    per_class = res.get("per_class", {})  # {lab: {corr_max, t_star, area, window, z_abs}}

    # Build aux.stage11.wdd.arc.results as a dict keyed by primitive
    results_map: Dict[str, Dict[str, Any]] = {}
    for lab, s in per_class.items():
        ok = lab in keep_order
        info = {
            # ARC-specific fields we do have from the audit:
            "t_peak": {lab: s.get("t_star")},  # expose peak time like Tier-2 DeFi
            "corr_max": s.get("corr_max"),
            "area": s.get("area"),
            "window": s.get("window"),
            "z_abs": s.get("z_abs"),
            # Placeholders (ARC research harness doesn’t produce these yet)
            "sigma": None,
            "proto_w": None,
            "which_prior": f"arc:{lab}",
        }
        # layer/mf_peak parity with DeFi WDD; we can map mf_peak to corr_max for now
        results_map[lab] = {
            "ok": ok,
            "info": info,
            "which": lab,
            "layer": None,
            "mf_peak": s.get("corr_max"),
        }

    # Summary maps (only for kept primitives)
    which_prior_map = {lab: results_map[lab]["info"]["which_prior"] for lab in keep_order}
    sigma_map       = {lab: results_map[lab]["info"]["sigma"]       for lab in keep_order}
    proto_w_map     = {lab: results_map[lab]["info"]["proto_w"]     for lab in keep_order}

    wdd_summary = {
        "decision": ("PASS" if keep_order else "ABSTAIN"),
        "keep": keep_order,
        "order": (keep_order if mode == "family" else []),
        "which_prior": (which_prior_map or None),
        "sigma": (sigma_map or None),
        "proto_w": (proto_w_map or None),
        "note": f"mode={mode}",
    }

    out = {
        "prompt": prompt,
        "domain": "arc",
        "rails": "stage11",
        "T": policy.get("T", 180),
        "top1": None,
        "sequence": [],  # keep empty like DeFi; the plan carries the executable order
        "plan": {"sequence": plan_seq},
        "verify": (
            {"ok": True, "reason": "wdd:family:arc", "tags": ["audit:wdd"]}
            if (mode == "family" and plan_seq)
            else (
                {"ok": False, "reason": "wdd:family:abstain", "tags": ["audit:wdd"]}
                if mode == "family"
                else {"ok": True, "reason": "shim:accept:stage-4", "tags": ["audit:wdd"]}
            )
        ),
        "flags": {"wdd_family": (mode == "family")},
        "aux": {"stage11": {"wdd": {"arc": {"mode": mode, "results": results_map}}}},
        "wdd_summary": wdd_summary,
        "det_hash": "",
        "abstained": (mode == "family" and not plan_seq),
    }
    return out



# def run_arc_wdd(prompt: str, grid: np.ndarray, policy: Optional[Dict[str, Any]] = None, sequence: Optional[List[str]] = None, debug: bool = False) -> Dict[str, Any]:
#     """Thin wrapper returning an INTERFACE.md-shaped dict for ARC.

#     Fields:
#       - domain: "arc"
#       - plan.sequence: ordered list if family mode; else []
#       - wdd_summary: summary of WDD decision (keep/order/which_prior/...)
#     """
#     policy = policy or {}
#     res = wdd_arc_audit(prompt)

#     # wdd_arc_audit output
#     # {'prompt': 'quarter turn clockwise on the matrix',
#     #  'scores': {'rotate': 1.0, 'flip_h': 0.0, 'flip_v': 0.0},
#     #  'candidates': ['rotate'],
#     #  'selected': ['rotate'],
#     #  'per_class': {'rotate': {'corr_max': 0.547179951108328,
#     #    't_star': 0,
#     #    'area': 6.265300263505406,
#     #    'window': (0, 14),
#     #    'z_abs': -1.4115122082150304}}}
        
#     out = {
#         "prompt": prompt,
#         "domain": "arc",
#         "rails": "stage11",
#         "T": policy.get("T", 180),
#         "top1": None,  # ARC doesn’t use a Tier-1 mapper label; we can set canonical primitive later if needed
#         "sequence": res["selected"],
#         "verify": {"ok": True, "reason": ("wdd:family:arc" if mode == "family" else "shim:accept:stage-4"), "tags": ["audit:wdd"]},
#         "flags": {"wdd_family": (mode == "family")},
#         "wdd_summary": res["per_class"],

#         "abstained": (len(det["summary"]["keep"]) == 0),
#     }

#     # # Family gate: if nothing kept in family mode, mark verify as abstain
#     # if mode == "family" and not det["summary"]["keep"]:
#     #     out["verify"] = {"ok": False, "reason": "wdd:family:abstain", "tags": ["audit:wdd"]}
#     #     out["plan"] = {"sequence": []}

#     return out
