from __future__ import annotations
import json, math
from typing import List, Dict, Tuple, Any
import numpy as np

# --- SBERT import (lazy/guarded) ---
try:
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
except Exception:
    SentenceTransformer = None  # resolved at runtime in Emb.__init__


TERM_BANK: Dict[str, List[str]] = {
    "deposit_asset":  [
        "deposit","supply","provide","add liquidity","provide liquidity","top up","put in",
        "add funds","fund","allocate","contribute","add position","add to pool","supply to pool",
        "add into","supply into","top-up","topup"
    ],
    "withdraw_asset": [
        "withdraw","redeem","remove liquidity","pull out","take out","cash out","exit",
        "remove position","remove from pool","take from","pull from"
    ],
    "swap_asset":     ["swap","convert","trade","exchange","convert into","swap to","swap into","bridge","wrap","unwrap","swap for"],
    "borrow_asset":   ["borrow","draw","open a loan for","open debt","draw down","take a loan","borrow against"],
    "repay_asset":    ["repay","pay back","close out the loan for","settle loan","pay debt","repay debt","close loan"],
    "stake_asset":    ["stake","lock","bond","delegate","lock up","stake into","stake to","stake on","restake","redelegate"],
    "unstake_asset":  ["unstake","unlock","unbond","undelegate","release","unstake from","unstake out","unstow","withdraw staked"],
    "claim_rewards":  ["claim","harvest","collect rewards","claim rewards","collect staking rewards","collect yield","claim yield","harvest rewards","collect incentives"],
}
PRIMS = list(TERM_BANK.keys())


class Emb:
    def __init__(self, model_name: str, batch_size: int = 64, normalize: bool = True):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not available. `pip install sentence-transformers`.")
        self.m = SentenceTransformer(model_name)
        self.batch = batch_size
        self.norm = normalize
    def transform(self, X: List[str]) -> np.ndarray:
        V = self.m.encode(list(X), batch_size=self.batch, normalize_embeddings=self.norm, show_progress_bar=False)
        return np.asarray(V)
    def encode_one(self, s: str) -> np.ndarray:
        return self.transform([s])[0]

def build_term_vectors_and_protos(term_bank: Dict[str, List[str]], emb: Emb):
    term_vectors = {k: emb.transform(v) for k, v in term_bank.items()}
    prototypes   = {k: vecs.mean(axis=0) for k, vecs in term_vectors.items()}
    return term_vectors, prototypes

def span_similarity_max(phrase_vec: np.ndarray, prim: str, term_vectors: Dict[str, np.ndarray], prototypes: Dict[str, np.ndarray]) -> float:
    # Try all term vectors for that primitive; fall back to class mean
    best = 0.0
    vecs = term_vectors.get(prim, [])
    if len(vecs):
        na = float(np.linalg.norm(phrase_vec))
        if na > 0:
            for v in vecs:
                nb = float(np.linalg.norm(v))
                if nb > 0:
                    best = max(best, float(np.dot(phrase_vec, v) / (na * nb)))
    if best == 0.0:
        pv = prototypes[prim]
        na = float(np.linalg.norm(phrase_vec)); nb = float(np.linalg.norm(pv))
        if na > 0 and nb > 0:
            best = float(np.dot(phrase_vec, pv) / (na * nb))
    return best


def build_term_vectors_and_protos(term_bank: Dict[str, List[str]], emb: Emb):
    term_vectors = {k: emb.transform(v) for k, v in term_bank.items()}
    prototypes   = {k: vecs.mean(axis=0) for k, vecs in term_vectors.items()}
    return term_vectors, prototypes



def span_similarity_max(phrase_vec: np.ndarray, prim: str, term_vectors: Dict[str, np.ndarray], prototypes: Dict[str, np.ndarray]) -> float:
    # Try all term vectors for that primitive; fall back to class mean
    best = 0.0
    vecs = term_vectors.get(prim, [])
    if len(vecs):
        na = float(np.linalg.norm(phrase_vec))
        if na > 0:
            for v in vecs:
                nb = float(np.linalg.norm(v))
                if nb > 0:
                    best = max(best, float(np.dot(phrase_vec, v) / (na * nb)))
    if best == 0.0:
        pv = prototypes[prim]
        na = float(np.linalg.norm(phrase_vec)); nb = float(np.linalg.norm(pv))
        if na > 0 and nb > 0:
            best = float(np.dot(phrase_vec, pv) / (na * nb))
    return best



def _norm_tokens(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return [t for t in s.split() if t]

def spans_all(prompt: str, term_vectors: dict, prototypes: dict, emb: Emb, tau_span: float = 0.50, n_max: int = 4, topk_per_prim: int = 3):
    toks = _norm_tokens(prompt)
    by_prim = {k: [] for k in TERM_BANK.keys()}
    for n in range(1, min(n_max, len(toks)) + 1):
        for i in range(0, len(toks) - n + 1):
            phrase = " ".join(toks[i:i+n])
            e = emb.encode_one(phrase)
            for prim in TERM_BANK.keys():
                sc = span_similarity_max(e, prim, term_vectors, prototypes)
                if sc >= tau_span:
                    t_center = (i + n/2.0) / max(1.0, len(toks))
                    by_prim[prim].append({"primitive": prim, "term": phrase, "score": float(sc), "t_center": float(t_center)})
    for prim in by_prim:
        by_prim[prim].sort(key=lambda z: -z["score"])
        by_prim[prim] = by_prim[prim][:topk_per_prim]
    return by_prim



def spans_all(prompt: str, term_vectors: dict, prototypes: dict, emb: Emb, tau_span: float = 0.50, n_max: int = 4, topk_per_prim: int = 3):
    toks = _norm_tokens(prompt)
    by_prim = {k: [] for k in TERM_BANK.keys()}
    for n in range(1, min(n_max, len(toks)) + 1):
        for i in range(0, len(toks) - n + 1):
            phrase = " ".join(toks[i:i+n])
            e = emb.encode_one(phrase)
            for prim in TERM_BANK.keys():
                sc = span_similarity_max(e, prim, term_vectors, prototypes)
                if sc >= tau_span:
                    t_center = (i + n/2.0) / max(1.0, len(toks))
                    by_prim[prim].append({"primitive": prim, "term": phrase, "score": float(sc), "t_center": float(t_center)})
    for prim in by_prim:
        by_prim[prim].sort(key=lambda z: -z["score"])
        by_prim[prim] = by_prim[prim][:topk_per_prim]
    return by_prim



def spans_gold_only(prompt: str, gold_prim: str, term_vectors: dict, prototypes: dict, emb: Emb, tau_span: float = 0.50, n_max: int = 4, topk: int = 3):
    toks = _norm_tokens(prompt)
    out = []
    for n in range(1, min(n_max, len(toks)) + 1):
        for i in range(0, len(toks) - n + 1):
            phrase = " ".join(toks[i:i+n])
            e = emb.encode_one(phrase)
            sc = span_similarity_max(e, gold_prim, term_vectors, prototypes)
            if sc >= tau_span:
                t_center = (i + n/2.0) / max(1.0, len(toks))
                out.append({"primitive": gold_prim, "term": phrase, "score": float(sc), "t_center": float(t_center)})
    out.sort(key=lambda z: -z["score"])
    return out[:topk]



def kaiser_window(L=160, beta=8.6) -> np.ndarray:
    n = np.arange(L, dtype="float32")
    w = np.i0(beta * np.sqrt(1 - ((2*n)/(L-1) - 1)**2))
    w = w / (np.linalg.norm(w) + 1e-9)
    return w.astype("float32")

def matched_filter_score_for_prim(spans: List[dict], q: np.ndarray, T: int, L: int, sigma: float = 0.0, seed: int = 0):
    x = np.zeros(T, dtype="float32")
    if sigma > 0.0:
        rng = np.random.default_rng(seed)
        x += rng.normal(0.0, sigma, size=T).astype("float32")
    for sp in spans:
        center = int(float(sp["t_center"]) * T)
        start = max(0, center - L//2)
        end   = min(T, start + L)
        x[start:end] += q[:end-start] * float(sp["score"])
    r = np.convolve(x, q[::-1], mode="valid")
    peak = float(r.max()) if r.size else 0.0
    null = float(np.linalg.norm(x) * np.linalg.norm(q))
    rel  = peak / (null + 1e-9)
    return peak, rel, null



def matched_filter_score_for_prim(spans: List[dict], q: np.ndarray, T: int, L: int, sigma: float = 0.0, seed: int = 0):
    x = np.zeros(T, dtype="float32")
    if sigma > 0.0:
        rng = np.random.default_rng(seed)
        x += rng.normal(0.0, sigma, size=T).astype("float32")
    for sp in spans:
        center = int(float(sp["t_center"]) * T)
        start = max(0, center - L//2)
        end   = min(T, start + L)
        x[start:end] += q[:end-start] * float(sp["score"])
    r = np.convolve(x, q[::-1], mode="valid")
    peak = float(r.max()) if r.size else 0.0
    null = float(np.linalg.norm(x) * np.linalg.norm(q))
    rel  = peak / (null + 1e-9)
    return peak, rel, null



def matched_filter_scores_all(span_map: Dict[str, List[dict]], q: np.ndarray, T: int, L: int, sigma: float = 0.0, seed: int = 0):
    scores, nulls = {}, {}
    for prim in PRIMS:
        x = np.zeros(T, dtype="float32")
        if sigma > 0.0:
            rng = np.random.default_rng(seed + hash(prim) % 65536)
            x += rng.normal(0.0, sigma, size=T).astype("float32")
        spans = span_map.get(prim, [])
        for sp in spans:
            center = int(float(sp["t_center"]) * T)
            start = max(0, center - L//2)
            end   = min(T, start + L)
            x[start:end] += q[:end-start] * float(sp["score"])
        r = np.convolve(x, q[::-1], mode="valid")
        peak = float(r.max()) if r.size else 0.0
        null = float(np.linalg.norm(x) * np.linalg.norm(q))
        scores[prim] = peak
        nulls[prim]  = null
    return scores, nulls



def decide_from_scores(scores: Dict[str, float], nulls: Dict[str, float], tau_rel: float, tau_abs: float):
    seq = []
    for prim in PRIMS:
        s = scores.get(prim, 0.0)
        n = nulls.get(prim, 0.0) + 1e-9
        rel = s / n
        if (s >= tau_abs) and (rel >= tau_rel):
            seq.append((prim, s))
    seq.sort(key=lambda z: -z[1])
    return [k for k,_ in seq]




def run_audit(prompts: List[str],
              gold_labels: List[str],
              sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
              n_max: int = 4, topk_per_prim: int = 3,
              tau_span: float = 0.50, tau_rel: float = 0.60, tau_abs: float = 0.93,
              L: int = 160, beta: float = 8.6, sigma: float = 0.0,
              competitive_eval: bool = False) -> Dict[str, Any]:
    '''
    Returns dict with: rows (per-example audit), metrics (aggregate), and aux (term vectors, etc.).
    Mirrors audit_bench_metrics.py decisions.
    '''
    assert len(prompts) == len(gold_labels)
    emb = Emb(sbert_model)
    term_vectors, prototypes = build_term_vectors_and_protos(TERM_BANK, emb)
    q = kaiser_window(L=L, beta=beta)
    rows = []
    n_span_total = 0
    n_span_success = 0
    null_peaks = []
    null_count = 0

    # optional competitive eval accumulators
    halluc_count = 0
    multi_accept = 0
    accept_counts = []

    for p, g in zip(prompts, gold_labels):
        spans_g, spans_all_map, norm = spans_gold_only(
            prompt=p, gold_prim=g, n_max=n_max, topk_per_prim=topk_per_prim,
            term_vectors=term_vectors, prototypes=prototypes, tau=tau_span, sigma=sigma
        )
        n_span_total += sum(len(v) for v in spans_all_map.values())
        n_span_success += sum(1 for s in spans_g)

        # scores for the gold class
        s_gold = matched_filter_score_for_prim(
            prim=g, spans_map=spans_all_map, q=q, term_vectors=term_vectors, prototypes=prototypes
        )
        # scores across all (for competitive pass / null)
        s_all = matched_filter_scores_all(
            spans_map=spans_all_map, q=q, term_vectors=term_vectors, prototypes=prototypes
        )

        # Build null dist from other-class peaks
        other_peaks = [float(v["peak"]) for k,v in s_all.items() if k != g and v is not None]
        null_peaks.extend(other_peaks); null_count += len(other_peaks)

        # Decide accept/abstain for gold (using same rule)
        decision = decide_from_scores(s_all, gold=g, tau_rel=tau_rel, tau_abs=tau_abs)

        # Competitive eval metrics
        if competitive_eval:
            # hallucination: some other class passed while gold failed
            accepts = [k for k,v in s_all.items() if v and (v["peak"] >= tau_abs and (v["peak"] / (max([vv["peak"] for kk,vv in s_all.items() if vv] + [1e-9]))) >= tau_rel)]
            accept_counts.append(len(accepts))
            if g not in accepts and len(accepts) > 0:
                halluc_count += 1
            if len(accepts) > 1:
                multi_accept += 1

        rows.append({
            "prompt": p,
            "gold": g,
            "norm": norm,
            "spans_gold": [list(s) for s in spans_g],
            "scores_gold": s_gold,
            "scores_all": s_all,
            "decision": decision,
        })

    coverage = (n_span_success / max(1, len(prompts)))
    span_yield_rate = (n_span_success / max(1, n_span_total)) if n_span_total else 0.0
    abstain_count = sum(1 for r in rows if not (r["decision"] or {}).get("accept"))
    metrics = {
        "coverage": round(coverage, 6),
        "abstain_rate": round(abstain_count / max(1, len(rows)), 6),
        "span_yield_rate": round(span_yield_rate, 6),
        "abstain_no_span_rate": round(sum(1 for r in rows if (not (r["spans_gold"])) and not (r["decision"] or {}).get("accept")) / max(1, len(rows)), 6),
        "abstain_with_span_rate": round(sum(1 for r in rows if (r["spans_gold"]) and not (r["decision"] or {}).get("accept")) / max(1, len(rows)), 6),
    }
    if competitive_eval:
        metrics.update({
            "hallucination_rate": round(halluc_count / max(1, len(rows)), 6),
            "multi_accept_rate": round(sum(1 for c in accept_counts if c>1) / max(1, len(rows)), 6),
        })

    aux = {
        "null_peaks_count": null_count,
        "null_peaks_mean": float(np.mean(null_peaks)) if null_peaks else 0.0,
        "null_peaks_std": float(np.std(null_peaks)) if null_peaks else 0.0,
        "params": {
            "n_max": n_max, "topk_per_prim": topk_per_prim,
            "tau_span": tau_span, "tau_rel": tau_rel, "tau_abs": tau_abs,
            "L": L, "beta": beta, "sigma": sigma,
        }
    }
    return {"rows": rows, "metrics": metrics, "aux": aux}
