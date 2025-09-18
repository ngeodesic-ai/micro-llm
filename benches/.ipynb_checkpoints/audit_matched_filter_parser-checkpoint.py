# Create a dedicated, self-contained script that audits a single prompt
# after the mapper has identified "deposit_asset" as the candidate primitive.
# It builds independent latents, runs matched filter + parser, and prints JSON.
import os, json, textwrap, numpy as np, hashlib

# -*- coding: utf-8 -*-
"""
audit_matched_filter_parser.py
--------------------------------
Given a prompt (already mapped by the mapper to a candidate primitive),
run an **independent** latent generation + matched filter + parser audit.

This script demonstrates the auditor pattern:
- Mapper proposes: restrict_to = {"deposit_asset"}  (passed in via CLI)
- Auditor disposes: builds text→latents WITHOUT using the mapper label to add energy,
  then runs matched-filter+null+relative gates and returns PASS/ABSTAIN.

Usage example:
  python3 audit_matched_filter_parser.py \
    --prompt "supply 7.0245 SOL to maker" \
    --candidate deposit_asset

Flags:
  --prompt <str>                       The user prompt
  --candidate <primitive>              Primitive selected by mapper (e.g., deposit_asset)
  --tau_ood 0.30                       OOD guard on max cosine to any prototype
  --tau_rel 0.50                       Relative gate (fraction of winning channel)
  --tau_abs 0.80                       Absolute gate vs null
  --tau_span 0.50                      Span match threshold
  --alpha 0.70 --beta 0.30             Amplitude blend: whole-sentence vs span evidence
  --T 720                              Trace length
  --sigma 0.02                         Noise scale
"""

import argparse, json, math, hashlib
from typing import Dict, List
import numpy as np

PRIMS = ["deposit_asset","withdraw_asset","borrow_asset","repay_asset","swap_asset"]

# --- Minimal phrase banks per primitive (replace with your harvested phrases) ---
PHRASES = {
    "deposit_asset": [
        "deposit", "top up", "add funds", "fund", "supply", "put in", "credit"
    ],
    "withdraw_asset": [
        "withdraw", "cash out", "take out", "remove", "pull"
    ],
    "borrow_asset": [
        "borrow", "take loan", "credit line", "obtain loan"
    ],
    "repay_asset": [
        "repay", "pay back", "return loan", "settle debt"
    ],
    "swap_asset": [
        "swap", "exchange", "trade", "convert"
    ]
}

# ---------- Placeholder embedding (replace with your true encoder) ----------
class PlaceholderEmbedding:
    def __init__(self, dim=256):
        self.dim = dim
    def encode_one(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.lower().encode("utf-8")).digest()
        arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        vec = np.tile(arr, int(np.ceil(self.dim/len(arr))))[:self.dim]
        vec = (vec - vec.mean()) / (vec.std() + 1e-6)
        return vec

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = a / (np.linalg.norm(a) + 1e-8)
    bn = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(an, bn))

def build_prototypes(emb, phrases: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    proto = {}
    for k, plist in phrases.items():
        vecs = [emb.encode_one(p) for p in plist] if plist else [emb.encode_one(k)]
        proto[k] = np.stack(vecs, axis=0).mean(axis=0)
    return proto

# ---------- Spans ----------
def spans_from_prompt(prompt: str, prototypes: Dict[str,np.ndarray], emb, tau_span=0.5):
    toks = prompt.strip().split()
    span_map = {}
    for k in prototypes.keys():
        span_map[k] = []
    for n in range(1, min(6, len(toks))+1):
        for i in range(0, len(toks)-n+1):
            s = " ".join(toks[i:i+n])
            e = emb.encode_one(s)
            for k, v in prototypes.items():
                sc = max(0.0, cosine(e, v))
                if sc >= tau_span:
                    t_center = (i + n/2.0) / max(1.0, len(toks))
                    span_map[k].append({"term": s, "score": round(sc,4), "t_center": round(t_center,4)})
    # keep top-3 per primitive
    for k in list(span_map.keys()):
        span_map[k] = sorted(span_map[k], key=lambda x: x["score"], reverse=True)[:3]
    return span_map

# ---------- Auditor: text -> independent latents ----------
def rng_for(prompt: str):
    seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed)

def q_template(width=160):
    t = np.linspace(0, 1, width)
    q = np.sin(np.pi * t)
    return q / (np.linalg.norm(q) + 1e-8)

def text_to_latents(prompt, emb, prototypes, tau_ood=0.30, span_map=None, alpha=0.7, beta=0.3, sigma=0.02, T=720):
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
        if len(vals)==0:
            peak = 0.0; idx = 0
        else:
            idx = int(np.argmax(vals)); peak = float(vals[idx])
        # null via circular shift + 95th percentile
        shift = len(x)//3
        x_null = np.roll(x, shift)
        vals_null = np.correlate(x_null, q, mode="valid")
        null_floor = float(np.percentile(vals_null, 95)) if len(vals_null)>0 else 0.0
        scores[k] = peak
        nulls[k] = null_floor
        peaks_at[k] = idx
    return scores, nulls, peaks_at

def rails_audit(traces, restrict_to=None, tau_rel=0.50, tau_abs=0.80):
    scores, nulls, peaks_at = matched_filter_scores(traces, restrict_to=restrict_to)
    if not scores:
        return {"decision":"ABSTAIN","sequence":[],"per_channel":{}}

    max_peak = max(scores.values())
    keep = {k: v for k, v in scores.items() if v >= tau_rel * max_peak}

    accepted = []
    for k, v in keep.items():
        thr = nulls[k] + tau_abs * abs(nulls[k])
        if v > thr:
            accepted.append((k, v, peaks_at[k], thr))

    if not accepted:
        return {"decision":"ABSTAIN","sequence":[],"per_channel":{k:{"peak":round(scores[k],4),"null":round(nulls[k],4),"passed":False} for k in keep.keys()}}

    accepted.sort(key=lambda x: x[2])  # order by peak location
    seq = [k for (k,_,_,_) in accepted]
    details = {}
    for (k, v, idx, thr) in accepted:
        details[k] = {"peak": round(v,4), "null": round(nulls[k],4), "peak_at": int(idx), "threshold": round(thr,4), "passed": True}
    # Also include losers for transparency
    for k in keep.keys():
        if k not in details:
            details[k] = {"peak": round(scores[k],4), "null": round(nulls[k],4), "passed": False}
    return {"decision":"PASS","sequence": seq, "per_channel": details}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--candidate", required=True, help="Primitive selected by mapper, e.g., deposit_asset")
    ap.add_argument("--tau_ood", type=float, default=0.30)
    ap.add_argument("--tau_rel", type=float, default=0.50)
    ap.add_argument("--tau_abs", type=float, default=0.80)
    ap.add_argument("--tau_span", type=float, default=0.50)
    ap.add_argument("--alpha", type=float, default=0.70)
    ap.add_argument("--beta", type=float, default=0.30)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--sigma", type=float, default=0.02)
    args = ap.parse_args()

    emb = PlaceholderEmbedding(dim=256)
    prototypes = build_prototypes(emb, PHRASES)

    # (1) Whole-prompt primitive mapping (for transparency only)
    e = emb.encode_one(args.prompt)
    primitive_mapping = {k: round(max(0.0, cosine(e, v)),4) for k, v in prototypes.items()}

    # (2) Primitive -> term mapping (span mining, independent of mapper)
    span_map = spans_from_prompt(args.prompt, prototypes, emb, tau_span=args.tau_span)

    # (3) Build independent latents, ignoring mapper for energy; we'll restrict scoring to candidate
    traces, sims, s_max = text_to_latents(
        args.prompt, emb, prototypes,
        tau_ood=args.tau_ood, span_map=span_map,
        alpha=args.alpha, beta=args.beta, sigma=args.sigma, T=args.T
    )

    # (4) Matched filter + parser restricted to mapper's candidate
    restrict = {args.candidate}
    audit = rails_audit(traces, restrict_to=restrict, tau_rel=args.tau_rel, tau_abs=args.tau_abs)

    out = {
        "prompt": args.prompt,
        "mapper_candidate": args.candidate,
        "primitive_mapping": primitive_mapping,
        "primitive_to_term_mapping": span_map.get(args.candidate, []),
        "audit": audit,
        "notes": {
            "tau_ood": args.tau_ood, "tau_rel": args.tau_rel, "tau_abs": args.tau_abs,
            "tau_span": args.tau_span, "alpha": args.alpha, "beta": args.beta, "T": args.T, "sigma": args.sigma,
            "s_max": round(float(s_max),4)
        }
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

