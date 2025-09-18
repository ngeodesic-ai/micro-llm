from __future__ import annotations
import argparse, json, csv, sys, os, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
import numpy as np
import json, math, hashlib
import numpy as np

# --- sklearn bits
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

"""
  python3 src/micro_lm/domains/defi/benches/audit_bench_v3.py \
      --model_path .artifacts/defi_mapper.joblib \
      --prompts_jsonl tests/fixtures/defi/tests/fixtures/defi/defi_mapper_5k_prompts.json \
      --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv \
      --sbert sentence-transformers/all-MiniLM-L6-v2 \
      --out_dir .artifacts/defi/audit_bench 
"""

PRIMITIVE = "claim_rewards"

class SbertEmbedding:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.m = SentenceTransformer(model_name)
        
    def encode_one(self, text: str) -> np.ndarray:
        v = self.m.encode([text], normalize_embeddings=True)[0]
        return v.astype(np.float32)

def load_phrases():
    default = {
        "deposit_asset":  [
            "deposit","supply","provide","add liquidity","provide liquidity","top up","put in",
            "add funds","fund","allocate","contribute","add position","add to pool","supply to pool",
            "add into","supply into"
        ],
        "withdraw_asset": [
            "withdraw","redeem","remove liquidity","pull out","take out","cash out","exit",
            "remove position","remove from pool","take from","pull from"
        ],
        "swap_asset": ["swap","convert","trade","exchange","convert into","swap to","swap into","bridge","wrap","unwrap","swap for"],
        "borrow_asset": ["borrow","draw","open a loan for","open debt","draw down","take a loan","borrow against"],
        "repay_asset":  ["repay","pay back","close out the loan for","settle loan","pay debt","repay debt","close loan"],
        "stake_asset":  [
            "stake","lock","bond","delegate","lock up","stake into","stake to","stake on",
            "restake",               
            "redelegate"       
        ],
        "unstake_asset": [
            "unstake","unlock","unbond","undelegate","release","unstake from","unstake out",
            "unstow",             
            "withdraw staked"    
        ],
        "claim_rewards": [
            "claim","harvest","collect rewards","claim rewards","collect staking rewards",
            "collect yield","claim yield","harvest rewards",
            "collect incentives"   
        ],
    }
    return default

def get_train_prompts(args, N_TRAIN = 1000):

    labeled_train_prompts_read = _read_labels_csv(args.train_labels_csv)
    labeled_train_prompts = {}
    train_prompts = []
    train_labels = []
    for k, g in enumerate(labeled_train_prompts_read):
        labeled_train_prompts[k] = g
        train_labels.append(labeled_train_prompts_read[g])
        train_prompts.append(g)
        
        if (len(labeled_train_prompts) == N_TRAIN):
            break;

    return train_prompts, train_labels

def get_test_prompts(args, N_TEST = 25):
    # get test prompts
    labeled_test_prompts_read = _read_labels_csv(args.labels_csv_pred)
    labeled_test_prompts = {}
    test_prompts = []
    test_labels = []
    for k, g in enumerate(labeled_test_prompts_read):
        labeled_test_prompts[k] = g
        test_labels.append(labeled_test_prompts_read[g])
        test_prompts.append(g)

        if (len(labeled_test_prompts) == N_TEST):
            break;
            
    return test_prompts, test_labels


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
    
def cosine(a, b):
    an = a / (np.linalg.norm(a) + 1e-8)
    bn = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(an, bn))

def spans_from_prompt(prompt, prototypes, emb, tau_span=0.55):
    toks = prompt.strip().split()
    spans = []
    NGRAM = 3
    for n in range(1, min(NGRAM, len(toks))+1):
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

# ---------- utilities ----------
def kaiser_window(L=160, beta=8.6):
    # smooth, MF-friendly lobe (unit norm)
    n = np.arange(L)
    w = np.i0(beta * np.sqrt(1 - ((2*n)/(L-1) - 1)**2))
    w = w / (np.linalg.norm(w) + 1e-9)
    return w.astype("float32")

def matched_filter_scores(traces, q):
    scores, nulls, peaks = {}, {}, {}
    L = len(q)
    for k, x in traces.items():
        if len(x) < L:  # pad if needed
            x = np.pad(x, (0, L - len(x)))
        # convolution as correlation (flip q)
        r = np.convolve(x, q[::-1], mode="valid")
        peak = float(r.max()) if r.size else 0.0
        scores[k] = peak
        nulls[k]  = float(np.sqrt(np.sum(x**2)) * (np.linalg.norm(q))) / max(len(x),1)  # crude noise floor
        peaks[k]  = {"score": peak, "t_idx": int(np.argmax(r)) if r.size else 0}
    return scores, nulls, peaks

def decide(scores, nulls, tau_rel=0.60, tau_abs=0.93):
    accepted, seq = {}, []
    for k in scores:
        s, n = scores[k], nulls[k] + 1e-9
        rel = s / n
        if (rel >= tau_rel) and (s >= tau_abs):
            accepted[k] = {"score": round(s, 4), "rel": round(rel, 3), "null": round(n, 4)}
            seq.append((k, s))
    seq.sort(key=lambda z: -z[1])
    return [k for k, _ in seq], accepted

# ---------- core: audit from spans ----------
def audit_from_span_map(prompt:str,
                        primitive_to_term_mapping:dict,
                        T:int=720,
                        tau_span:float=0.55,
                        tau_abs:float=0.93,
                        tau_rel:float=0.60,
                        sigma:float=0.02,
                        fuse_per_primitive:bool=False):

    # 1) init noise traces
    seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
    rng  = np.random.default_rng(seed)
    primitives = list(primitive_to_term_mapping.keys())
    traces = {p: rng.normal(0.0, sigma, size=T).astype("float32") for p in primitives}

    # 2) build a canonical lobe shape
    q = kaiser_window(L=min(160, max(64, T//8)))

    # 3) inject lobes from strong spans
    span_hits = 0
    for p, hits in primitive_to_term_mapping.items():
        if not hits: 
            continue
        # keep only strong spans
        strong = [h for h in hits if float(h.get("score",0.0)) >= tau_span]
        if not strong:
            continue

        if fuse_per_primitive:
            # one fused lobe per primitive (amplitude = max span score)
            A = max(float(h["score"]) for h in strong)
            tc = np.mean([float(h.get("t_center", 0.5)) for h in strong])
            start = max(0, min(T - len(q), int(tc * (T - len(q)))))
            traces[p][start:start+len(q)] += A * q
            span_hits += 1
        else:
            # one lobe per span
            for h in strong:
                A  = float(h["score"])
                tc = float(h.get("t_center", 0.5))
                start = max(0, min(T - len(q), int(tc * (T - len(q)))))
                traces[p][start:start+len(q)] += A * q
                span_hits += 1

    if span_hits == 0:
        return {
            "decision": "ABSTAIN",
            "sequence": [],
            "accepted_peaks": {},
            "notes": {"reason": "no_span_evidence", "tau_span": tau_span, "T": T}
        }

    # 4) matched filter + parser
    scores, nulls, peaks = matched_filter_scores(traces, q)
    sequence, accepted = decide(scores, nulls, tau_rel=tau_rel, tau_abs=tau_abs)

    return {
        "decision": "PASS" if sequence else "ABSTAIN",
        "sequence": sequence,
        "accepted_peaks": accepted,
        "peaks": peaks,  # optional: raw peak info
        "notes": {
            "tau_span": tau_span, "tau_rel": tau_rel, "tau_abs": tau_abs,
            "sigma": sigma, "T": T, "fused": fuse_per_primitive
        }
    }

def _read_labels_csv(path: str) -> Dict[str, str]:
    gold: Dict[str, str] = {}
    df = pd.read_csv(path)
    if not {"prompt", "label"}.issubset(df.columns):
        raise ValueError(f"labels_csv must have columns ['prompt','label'], got {df.columns.tolist()}" )
    for _, row in df.iterrows():
        p = str(row["prompt"]).strip()
        y = str(row["label"]).strip()
        if p:
            gold[p] = y
    return gold

# ---------- demo ----------
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend",       default="sbert", help="wordmap|sbert")
    ap.add_argument("--model_path",    default=".artifacts/defi_mapper.joblib")
    ap.add_argument("--prompts_jsonl", default="tests/fixtures/defi/defi_mapper_5k_prompts.json")
    ap.add_argument("--labels_csv_pred", default="tests/fixtures/defi/defi_mapper_labeled_5k.csv")
    ap.add_argument("--train_labels_csv", default="tests/fixtures/defi/defi_mapper_labeled_large.csv")
    ap.add_argument("--thresholds",    default="0.5,0.55,0.6,0.65,0.7")
    ap.add_argument("--max_iter",    default="2000")
    ap.add_argument("--C",    default="8")
    ap.add_argument("--calibrate",    default="True")
    ap.add_argument("--calibration_method", choices=["auto","isotonic","sigmoid"], default="auto")
    ap.add_argument("--calibration_cv", type=int, default=3)
    ap.add_argument("--sbert_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--out_path", default="defi_mapper_embed.joblib")
    ap.add_argument("--out_dir",       default="")
    ap.add_argument("--out_rows_csv", default=".artifacts/m8_rows_simple.csv")
    ap.add_argument("--min_overall_acc", default=None)
    ap.add_argument("--verbose", default=False)
    args = ap.parse_args()
    
    tau_span = 0.55
    N_TESTS = 100
    
    test_prompts, test_labels = get_test_prompts(args, N_TESTS)

    emb = SbertEmbedding()
    phrases = load_phrases()
    prototypes = build_prototypes(emb, phrases)
    
    pass_audit = 0
    
    for k, test_prompt in enumerate(test_prompts):
        span_map = spans_from_prompt(test_prompt, prototypes, emb, tau_span=tau_span)
        primitive = test_labels[k]
        if(primitive in span_map):    
            primitive_to_term_mapping = {
                "deposit_asset":  [],       
                "withdraw_asset": [],
                "swap_asset": [],
                "borrow_asset": [],
                "repay_asset": [],
                "stake_asset": [],
                "unstake_asset": [],
                "claim_rewards": []
            }

            primitive_to_term_mapping[primitive] = span_map[primitive]
            
            audit = audit_from_span_map(
                test_prompt,
                primitive_to_term_mapping,
                T=720, tau_span=0.55, tau_abs=0.50, tau_rel=0.60, sigma=0.02,
                fuse_per_primitive=False
            )

            is_passed = audit['decision']
            if(args.verbose): print(f'{k} {is_passed} / prompt: {test_prompt} / primitives: {list(span_map.keys())}')
            if is_passed == "PASS":
                pass_audit += 1
        else:
            print(f'{k} ABSTAIN* / prompt: {test_prompt} / primitives: {list(span_map.keys())} / truth: {primitive}')

    print(f'Total PASS {pass_audit} / {N_TESTS}')
    
        