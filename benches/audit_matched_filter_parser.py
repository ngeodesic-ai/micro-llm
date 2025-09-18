#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_matched_filter_parser.py
--------------------------------
Audit a single prompt after the mapper proposes a candidate primitive.
Build independent latents from text using an encoder (toy|sbert), then
run matched filter + parser restricted to the candidate.

Usage:
  python3 audit_matched_filter_parser.py \
    --prompt "supply 7.0245 SOL to maker" \
    --candidate deposit_asset \
    --encoder sbert \
    --phrases_json phrases.json \
    --tau_ood 0.45 --tau_rel 0.60 --tau_abs 0.93 --tau_span 0.55 --alpha 0.7 --beta 0.3 --T 720 --sigma 0.02
"""

import argparse, json, math, hashlib
import numpy as np
from typing import Dict, List

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_OK = True
except Exception:
    _SBERT_OK = False

PRIMS = ["deposit_asset","withdraw_asset","borrow_asset","repay_asset","swap_asset"]

# Fallback small phrase bank (used if --phrases_json is missing or invalid)
PHRASES_FALLBACK = {
    "deposit_asset": ["deposit","top up","add funds","fund","supply","put in","credit"],
    "withdraw_asset": ["withdraw","cash out","take out","remove","pull"],
    "borrow_asset": ["borrow","take loan","credit line","obtain loan"],
    "repay_asset": ["repay","pay back","return loan","settle debt"],
    "swap_asset": ["swap","exchange","trade","convert"]
}

# ---------- Embeddings ----------
class PlaceholderEmbedding:
    def __init__(self, dim=256):
        self.dim = dim
    def encode_one(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.lower().encode("utf-8")).digest()
        arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        vec = np.tile(arr, int(np.ceil(self.dim/len(arr))))[:self.dim]
        vec = (vec - vec.mean()) / (vec.std() + 1e-6)
        return vec

class SbertEmbedding:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if not _SBERT_OK:
            raise RuntimeError("sentence-transformers not available; pip install sentence-transformers")
        self.m = SentenceTransformer(model_name)
    def encode_one(self, text: str) -> np.ndarray:
        v = self.m.encode([text], normalize_embeddings=True)[0]
        return v.astype(np.float32)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = a / (np.linalg.norm(a) + 1e-8)
    bn = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(an, bn))

def build_prototypes(emb, phrases: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    proto = {}
    for k, plist in phrases.items():
        if not plist:
            plist = [k]
        vecs = [emb.encode_one(p) for p in plist]
        proto[k] = np.stack(vecs, axis=0).mean(axis=0)
    return proto

# ---------- Span mining ----------
def spans_from_prompt(prompt: str, prototypes: Dict[str,np.ndarray], emb, tau_span=0.55):
    toks = prompt.strip().split()
    span_map = {k: [] for k in prototypes.keys()}
    for n in range(1, min(6, len(toks))+1):
        for i in range(0, len(toks)-n+1):
            s = " ".join(toks[i:i+n])
            e = emb.encode_one(s)
            for k, v in prototypes.items():
                sc = max(0.0, cosine(e, v))
                if sc >= tau_span:
                    t_center = (i + n/2.0) / max(1.0, len(toks))
                    span_map[k].append({"term": s, "score": round(sc,4), "t_center": round(t_center,4)})
    for k in span_map:
        span_map[k] = sorted(span_map[k], key=lambda x: x["score"], reverse=True)[:3]
    return span_map

# ---------- Latent generator ----------
def rng_for(prompt: str):
    seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed)

def q_template(width=160):
    t = np.linspace(0, 1, width)
    q = np.sin(np.pi * t)
    return q / (np.linalg.norm(q) + 1e-8)

def text_to_latents(prompt, emb, prototypes, tau_ood=0.45, span_map=None, alpha=0.7, beta=0.3, sigma=0.02, T=720):
    e = emb.encode_one(prompt)
    sims = {k: max(0.0, cosine(e, prototypes[k])) for k in prototypes}
    s_max = max(sims.values()) if sims else 0.0

    rng = rng_for(prompt)
    traces = {k: rng.normal(0.0, sigma, size=T).astype("float32") for k in prototypes}

    if s_max < tau_ood:
        return traces, sims, s_max

    q = q_template(width=min(160, T//4))
    qlen = len(q)
    for k, s in sims.items():
        if s <= 0.0:
            continue
        span_max = 0.0
        t_center = int(0.35 * T)
        if span_map and span_map.get(k):
            best = max(span_map[k], key=lambda sp: sp["score"])
            span_max = best["score"]
            t_center = int(best["t_center"] * (T - qlen))
        A = (alpha * s + beta * span_max)
        if A <= 0.0:
            continue
        start = max(0, min(T - qlen, t_center))
        traces[k][start:start+qlen] += (A * q)

    return traces, sims, s_max

# ---------- Matched filter + parser ----------
def matched_filter_scores(traces, restrict_to=None):
    keys = list(traces.keys()) if restrict_to is None else [k for k in traces.keys() if k in restrict_to]
    q = q_template()
    scores, nulls, peaks_at = {}, {}, {}
    for k in keys:
        x = traces[k]
        vals = np.correlate(x, q, mode="valid")
        if len(vals) == 0:
            peak = 0.0; idx = 0
        else:
            idx = int(np.argmax(vals)); peak = float(vals[idx])
        shift = len(x)//3
        x_null = np.roll(x, shift)
        vals_null = np.correlate(x_null, q, mode="valid")
        null_floor = float(np.percentile(vals_null, 95)) if len(vals_null)>0 else 0.0
        scores[k] = peak; nulls[k] = null_floor; peaks_at[k] = idx
    return scores, nulls, peaks_at

def rails_audit(traces, restrict_to=None, tau_rel=0.60, tau_abs=0.93):
    scores, nulls, peaks_at = matched_filter_scores(traces, restrict_to=restrict_to)
    if not scores:
        return {"decision":"ABSTAIN","sequence":[],"per_channel":{}}

    max_peak = max(scores.values()) if scores else 0.0
    keep = {k: v for k, v in scores.items() if v >= tau_rel * max_peak}

    accepted = []
    for k, v in keep.items():
        thr = nulls[k] + tau_abs * abs(nulls[k])
        if v > thr:
            accepted.append((k, v, peaks_at[k], thr))

    if not accepted:
        return {"decision":"ABSTAIN","sequence":[],"per_channel":{k:{"peak":round(scores[k],4),"null":round(nulls[k],4),"passed":False} for k in keep.keys()}}

    accepted.sort(key=lambda x: x[2])
    seq = [k for (k,_,_,_) in accepted]
    details = {}
    for (k, v, idx, thr) in accepted:
        details[k] = {"peak": round(v,4), "null": round(nulls[k],4), "peak_at": int(idx), "threshold": round(thr,4), "passed": True}
    for k in keep.keys():
        if k not in details:
            details[k] = {"peak": round(scores[k],4), "null": round(nulls[k],4), "passed": False}
    return {"decision":"PASS","sequence": seq, "per_channel": details}

def load_phrases(path: str):
    if not path:
        return PHRASES_FALLBACK
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            return data
    except Exception:
        pass
    return PHRASES_FALLBACK

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--candidate", required=True, help="Primitive selected by mapper (e.g., deposit_asset)")
    ap.add_argument("--encoder", type=str, default="toy", help="toy|sbert")
    ap.add_argument("--phrases_json", type=str, default="", help="optional: {primitive:[phrases...]}")
    ap.add_argument("--tau_ood", type=float, default=0.45)
    ap.add_argument("--tau_rel", type=float, default=0.60)
    ap.add_argument("--tau_abs", type=float, default=0.93)
    ap.add_argument("--tau_span", type=float, default=0.55)
    ap.add_argument("--alpha", type=float, default=0.70)
    ap.add_argument("--beta", type=float, default=0.30)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--sigma", type=float, default=0.02)
    args = ap.parse_args()

    emb = SbertEmbedding() if args.encoder == "sbert" else PlaceholderEmbedding(dim=256)
    phrases = load_phrases(args.phrases_json)
    prototypes = build_prototypes(emb, phrases)

    # 1) whole-prompt mapping
    e = emb.encode_one(args.prompt)
    primitive_mapping = {k: round(max(0.0, cosine(e, v)),4) for k, v in prototypes.items()}

    # 2) span mining
    span_map = spans_from_prompt(args.prompt, prototypes, emb, tau_span=args.tau_span)

    # 3) latents
    traces, sims, s_max = text_to_latents(
        args.prompt, emb, prototypes,
        tau_ood=args.tau_ood, span_map=span_map,
        alpha=args.alpha, beta=args.beta, sigma=args.sigma, T=args.T
    )

    # 4) matched filter restricted to candidate
    restrict = {args.candidate}
    audit = rails_audit(traces, restrict_to=restrict, tau_rel=args.tau_rel, tau_abs=args.tau_abs)

    out = {
        "prompt": args.prompt,
        "mapper_candidate": args.candidate,
        "primitive_mapping": primitive_mapping,
        "primitive_to_term_mapping": span_map.get(args.candidate, []),
        "audit": audit,
        "notes": {
            "encoder": args.encoder,
            "phrases_used": list(phrases.get(args.candidate, []))[:8],
            "tau_ood": args.tau_ood, "tau_rel": args.tau_rel, "tau_abs": args.tau_abs,
            "tau_span": args.tau_span, "alpha": args.alpha, "beta": args.beta, "T": args.T, "sigma": args.sigma,
            "s_max": round(float(s_max),4)
        }
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
