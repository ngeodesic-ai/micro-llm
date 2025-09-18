#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
single_prompt_audit.py
----------------------
Audit a SINGLE prompt end-to-end and print a JSON blob to stdout with 3 sections:
1) primitive_mapping: whole-sentence similarity scores per primitive
2) primitive_to_term_mapping: span (term) matches found in the prompt per primitive
3) audit: matched-filter decision over independent latents for all primitive->term mappings

Usage:
  python3 single_prompt_audit.py \
    --prompt "top up my account 50 USDC" \
    --phrases_json phrases.json \
    --tau_ood 0.45 --tau_span 0.55 --alpha 0.7 --beta 0.3 --T 720


PYTHONPATH=. python3 benches/single_prompt_audit.py \
  --prompt "top up my account 50 USDC" \
  --phrases_json benches/phrases.json \
  --mapper_backend sbert \
  --model_path .artifacts/defi_mapper.joblib \
  --map_threshold 0.7 \
  --tau_ood 0.5 --tau_span 0.6 --alpha 0.6 --beta 0.2


PYTHONPATH=. python3 benches/single_prompt_audit.py \
  --prompt "top up my account 50 USDC" \
  --phrases_json benches/phrases.json \
  --mapper_backend sbert \
  --model_path .artifacts/defi_mapper.joblib \
  --map_threshold 0.7\
  --tau_ood 0.45 --tau_span 0.55 --alpha 0.6 --beta 0.25


PYTHONPATH=. python3 benches/single_prompt_audit.py \
  --prompt "add 10 USDC into my account" \
  --phrases_json benches/phrases.json \
  --mapper_backend sbert \
  --model_path .artifacts/defi_mapper.joblib \
  --map_threshold 0.7 \
  --tau_ood 0.45 --tau_span 0.55 --alpha 0.6 --beta 0.2

PYTHONPATH=. python3 benches/single_prompt_audit.py \
  --prompt "deposit 4697 USDC into uniswap on base" \
  --phrases_json benches/phrases.json \
  --mapper_backend sbert \
  --model_path .artifacts/defi_mapper.joblib \
  --map_threshold 0.7 \
  --tau_ood 0.45 --tau_span 0.55 --alpha 0.6 --beta 0.2

PYTHONPATH=. python3 benches/single_prompt_audit.py \
  --prompt "deposit 4697 USDC into uniswap on base" \
  --phrases_json benches/phrases.json \
  --mapper_backend sbert \
  --model_path .artifacts/defi_mapper.joblib \
  --map_threshold 0.7 \
  --encoder sbert \
  --require_mapper_pass 1 \
  --tau_ood 0.45 --tau_span 0.55 --alpha 0.6 --beta 0.2 --T 720

PYTHONPATH=. python3 benches/single_prompt_audit.py \
  --prompt "supply 7.0245 SOL to maker" \
  --phrases_json benches/phrases.json \
  --mapper_backend sbert \
  --model_path .artifacts/defi_mapper.joblib \
  --map_threshold 0.7 \
  --encoder sbert \
  --require_mapper_pass 1 \
  --tau_ood 0.45 --tau_span 0.55 --alpha 0.6 --beta 0.2 --T 720
  

Notes:
- This uses the same placeholder embedding + auditor logic as audit_bench.py.
- Replace PlaceholderEmbedding with your real encoder when ready.
"""

import argparse, sys, os, math, hashlib, json
import numpy as np
try:
    from micro_lm.core.mapper.base import load_backend as _load_mapper_backend
except Exception:
    _load_mapper_backend = None
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_OK = True
except Exception:
    _SBERT_OK = False

PRIMS_DEFAULT = ["deposit_asset","withdraw_asset","borrow_asset","repay_asset","swap_asset","stake_asset","unstake_asset","claim_rewards"]

# ---------- Placeholder Embedding (replace with your encoder) ----------
class PlaceholderEmbedding:
    def __init__(self, dim=256):
        self.dim = dim
    def _vec(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.lower().encode("utf-8")).digest()
        arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        vec = np.tile(arr, int(np.ceil(self.dim/len(arr))))[:self.dim]
        vec = (vec - vec.mean()) / (vec.std() + 1e-6)
        return vec
    def encode_one(self, text: str) -> np.ndarray:
        return self._vec(text)

class SbertEmbedding:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if not _SBERT_OK:
            raise RuntimeError("sentence-transformers not available; pip install sentence-transformers")
        self.m = SentenceTransformer(model_name)
    def encode_one(self, text: str) -> np.ndarray:
        v = self.m.encode([text], normalize_embeddings=True)[0]
        return v.astype(np.float32)

def cosine(a, b):
    an = a / (np.linalg.norm(a) + 1e-8)
    bn = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(an, bn))

def build_prototypes(emb, phrases_or_thresholds: dict):
    """
    Accepts either:
      A) phrases: {prim: [(phrase, score), (phrase, score), ...]}  or  {prim: [phrase, phrase, ...]}
      B) thresholds: {prim: float}  (per-class threshold file)
    For (B), we fall back to embedding the primitive name itself.
    """
    proto = {}
    for p, val in phrases_or_thresholds.items():
        vecs = []
        if isinstance(val, (int, float)):
            # thresholds file -> fallback: use primitive token as prototype
            vecs = [emb.encode_one(p)]
        else:
            # phrases file: list of phrases or (phrase,score)
            if isinstance(val, list) and len(val) > 0:
                for item in val:
                    if isinstance(item, (list, tuple)) and len(item) > 0:
                        phrase = item[0]
                    else:
                        phrase = item
                    vecs.append(emb.encode_one(str(phrase)))
            else:
                # empty -> fallback
                vecs = [emb.encode_one(p)]
        proto[p] = np.stack(vecs, axis=0).mean(axis=0)
    return proto


# ---------- Spans ----------
def spans_from_prompt(prompt, prototypes, emb, tau_span=0.55):
    toks = prompt.strip().split()
    spans = []
    for n in range(1, min(6, len(toks))+1):
        for i in range(0, len(toks)-n+1):
            s = " ".join(toks[i:i+n])
            e = emb.encode_one(s)
            for k, v in prototypes.items():
                sc = max(0.0, cosine(e, v))
                if sc >= tau_span:
                    t_center = (i + n/2.0) / max(1.0, len(toks))  # normalized [0,1]
                    spans.append({"primitive": k, "term": s, "score": round(sc,4), "t_center": round(t_center,4)})
    # keep top 3 spans per primitive
    by_prim = {}
    for sp in spans:
        by_prim.setdefault(sp["primitive"], []).append(sp)
    for k in by_prim:
        by_prim[k] = sorted(by_prim[k], key=lambda x: x["score"], reverse=True)[:3]
    return by_prim

# ---------- Auditor: text -> latents ----------
def rng_for(prompt):
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
        # OOD hard gate: noise-only traces; caller should ABSTAIN
        return traces, sims, s_max

    q = q_template(width=min(160, T//4))
    for k, s in sims.items():
        if s <= 0.0:
            continue
        span_max = 0.0
        t_center = int(0.35 * T)
        if span_map and k in span_map and span_map[k]:
            best = max(span_map[k], key=lambda sp: sp["score"])
            span_max = best["score"]
            t_center = int(best["t_center"] * (T - len(q)))

        A = (alpha * s + beta * span_max)
        if A <= 0.0:
            continue

        start = max(0, min(T - len(q), t_center))
        traces[k][start:start+len(q)] += (A * q)

    return traces, sims, s_max

def load_phrases(path: str):
    # Expect {primitive: [phrases...]}
    default = {
        "deposit_asset": ["deposit","top up","add","add funds","fund","supply","put in"],
        "withdraw_asset": ["withdraw","cash out","take out","remove","pull"],
        "swap_asset": ["swap","convert","exchange","trade"],
        "borrow_asset": ["borrow","take loan","obtain credit","draw"],
        "repay_asset": ["repay","pay back","settle debt","return loan"],
        "stake_asset": ["stake","lock","bond","delegate"],
        "unstake_asset": ["unstake","unlock","withdraw stake","redeem stake"],
        "claim_rewards": ["claim","collect","harvest","receive rewards"]
    }
    if not path:
        return default
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            return data
    except Exception:
        pass
    return default

# ---------- Matched filter + parser ----------
def matched_filter_scores(traces, restrict_to=None):
    keys = list(traces.keys()) if restrict_to is None else [k for k in traces.keys() if k in restrict_to]
    q = q_template()
    scores = {}
    nulls  = {}
    for k in keys:
        x = traces[k]
        vals = np.correlate(x, q, mode="valid")
        peak = float(np.max(vals)) if len(vals)>0 else 0.0
        # simple null via circular shift
        shift = len(x)//3
        x_null = np.roll(x, shift)
        vals_null = np.correlate(x_null, q, mode="valid")
        null_floor = float(np.percentile(vals_null, 95)) if len(vals_null)>0 else 0.0
        scores[k] = peak
        nulls[k]  = null_floor
    return scores, nulls

def rails_audit(traces, restrict_to=None, tau_rel=0.60, tau_abs_q=0.93):
    scores, nulls = matched_filter_scores(traces, restrict_to)
    if not scores:
        return {"sequence": [], "peaks": {}, "scores": scores, "nulls": nulls, "decision": "ABSTAIN"}

    max_peak = max(scores.values())
    keep = {k: v for k, v in scores.items() if v >= tau_rel * max_peak}

    accept = []
    for k, v in keep.items():
        thr = nulls[k] + tau_abs_q * abs(nulls[k])
        if v > thr:
            accept.append((k, v))

    if not accept:
        return {"sequence": [], "peaks": {}, "scores": scores, "nulls": nulls, "decision": "ABSTAIN"}

    accept.sort(key=lambda x: x[1], reverse=True)
    sequence = [k for k,_ in accept]
    peaks = {k: {"score": round(s,4)} for k,s in accept}
    return {"sequence": sequence, "peaks": peaks, "scores": scores, "nulls": nulls, "decision": "PASS"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="Single prompt to audit")
    ap.add_argument("--phrases_json", required=True, help="phrases.json from audit_bench training")
    ap.add_argument("--tau_ood", type=float, default=0.45)
    ap.add_argument("--tau_span", type=float, default=0.55)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--mapper_backend", type=str, default="none")
    ap.add_argument("--model_path", type=str, default=".artifacts/defi_mapper.joblib")
    ap.add_argument("--map_threshold", type=float, default=0.7)
    ap.add_argument("--encoder", type=str, default="sbert", help="sbert|toy (default: sbert)")
    ap.add_argument("--require_mapper_pass", type=int, default=1, help="1=if mapper abstains, force ABSTAIN")
    ap.add_argument("--span_ood_override", type=float, default=0.60, help="min span score to override OOD")

    args = ap.parse_args()

    emb = SbertEmbedding() if args.encoder == "sbert" else PlaceholderEmbedding(dim=256)
    phrases = load_phrases(args.phrases_json)
    prototypes = build_prototypes(emb, phrases)

    # 1) primitive mapping (whole-sentence sim)
    e = emb.encode_one(args.prompt)
    primitive_mapping = {k: round(max(0.0, cosine(e, v)), 4) for k, v in prototypes.items()}

    # 2) primitive -> term mapping (span detection)
    span_map = spans_from_prompt(args.prompt, prototypes, emb, tau_span=args.tau_span)

    # 3) audit via matched filter in latent space
    traces, sims, s_max = text_to_latents(
        args.prompt, emb, prototypes, tau_ood=args.tau_ood,
        span_map=span_map, alpha=args.alpha, beta=args.beta, T=args.T
    )

    # Mapper gating
    restrict = None
    mapper_pass = False
    mapper_label = None
    mapper_conf = None
    if args.mapper_backend != "none" and _load_mapper_backend is not None:
        be = _load_mapper_backend(args.mapper_backend, confidence_threshold=args.map_threshold, model_path=args.model_path)
        lab, conf = be.predict([args.prompt])[0]
        mapper_label, mapper_conf = lab, float(conf) if conf is not None else None
        print(f"[mapper_decision] label={mapper_label} conf={mapper_conf}", file=sys.stderr)
        if lab is not None and conf is not None and conf >= args.map_threshold:
            restrict = {lab}
            mapper_pass = True
    if args.require_mapper_pass and not mapper_pass:
        # Explicitly abstain when mapper abstains
        print(json.dumps({
            "prompt": args.prompt,
            "primitive_mapping": primitive_mapping,
            "primitive_to_term_mapping": span_map,
            "audit": {"decision": "ABSTAIN", "sequence": [], "accepted_peaks": {}, "notes": {
                "reason": "mapper_abstained", "mapper_label": mapper_label, "mapper_conf": mapper_conf,
               "tau_ood": args.tau_ood, "tau_span": args.tau_span, "alpha": args.alpha, "beta": args.beta, "T": args.T, "s_max": round(float(s_max),4)
            }}
        }, indent=2))
        return

    # OOD hard gate with SPAN OVERRIDE: if best span is strong enough, treat as in-domain
    best_span = 0.0
    if span_map:
        for lst in span_map.values():
            if lst:
                best_span = max(best_span, max(sp["score"] for sp in lst))
    if s_max < args.tau_ood and best_span < args.span_ood_override:
        print(json.dumps({
            "prompt": args.prompt,
            "primitive_mapping": primitive_mapping,
            "primitive_to_term_mapping": span_map,
            "audit": {"decision": "ABSTAIN", "sequence": [], "accepted_peaks": {}, "notes": {
              "reason": "ood", "tau_ood": args.tau_ood, "tau_span": args.tau_span, "alpha": args.alpha, "beta": args.beta,
               "T": args.T, "s_max": round(float(s_max),4), "best_span": round(float(best_span),4), "span_ood_override": args.span_ood_override
            }}
        }, indent=2))
        return

    audit = rails_audit(traces, restrict_to=restrict)

    out = {
        "prompt": args.prompt,
        "primitive_mapping": primitive_mapping,
        "primitive_to_term_mapping": span_map,
        "audit": {
            "decision": audit["decision"],
            "sequence": audit["sequence"],
            "accepted_peaks": audit["peaks"],
            "notes": {
                "tau_ood": args.tau_ood, "tau_span": args.tau_span, "alpha": args.alpha, "beta": args.beta, "T": args.T,
                "s_max": round(float(s_max),4), "best_span": round(float(best_span),4), "span_ood_override": args.span_ood_override,
                "mapper_label": mapper_label, "mapper_conf": mapper_conf
            }
        }
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
