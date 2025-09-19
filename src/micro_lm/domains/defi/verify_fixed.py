from __future__ import annotations
import re, math, json
from typing import List, Dict, Tuple, Any
import numpy as np

# --- SBERT import (lazy/guarded) ---
try:
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
except Exception:
    SentenceTransformer = None  # resolved at runtime in Emb.__init__

# ------------------ Lexicon ------------------
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

# ------------------ SBERT wrapper ------------------
class Emb:
    def __init__(self, model_name: str, batch_size: int = 64, normalize: bool = True):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        self.m = SentenceTransformer(model_name)
        self.batch = batch_size
        self.norm = normalize
    def transform(self, X: List[str]) -> np.ndarray:
        V = self.m.encode(list(X), batch_size=self.batch, normalize_embeddings=self.norm, show_progress_bar=False)
        return np.asarray(V)
    def encode_one(self, s: str) -> np.ndarray:
        return self.transform([s])[0]

# ------------------ Span mining helpers ------------------
def build_term_vectors_and_protos(term_bank: Dict[str, List[str]], emb: Emb):
    term_vectors = {k: emb.transform(v) for k, v in term_bank.items()}
    prototypes   = {k: vecs.mean(axis=0) for k, vecs in term_vectors.items()}
    return term_vectors, prototypes

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def span_similarity_max(phrase_vec: np.ndarray, prim: str, term_vectors: Dict[str, np.ndarray], prototypes: Dict[str, np.ndarray]) -> float:
    vecs = term_vectors.get(prim, [])
    best = 0.0
    for v in vecs:
        best = max(best, _cos(phrase_vec, v))
    if best == 0.0:
        best = _cos(phrase_vec, prototypes[prim])
    return best

def _norm_tokens(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return [t for t in s.split() if t]

def spans_all(prompt: str, term_vectors: dict, prototypes: dict, emb: Emb,
              tau_span: float = 0.50, n_max: int = 4, topk_per_prim: int = 3) -> Dict[str, List[dict]]:
    toks = _norm_tokens(prompt)
    by_prim: Dict[str, List[dict]] = {k: [] for k in TERM_BANK.keys()}
    for n in range(1, min(n_max, len(toks)) + 1):
        for i in range(0, len(toks) - n + 1):
            phrase = " ".join(toks[i:i+n])
            e = emb.encode_one(phrase)
            for prim in TERM_BANK.keys():
                sc = span_similarity_max(e, prim, term_vectors, prototypes)
                if sc >= tau_span:
                    t_center = (i + n/2.0) / max(1.0, len(toks))
                    by_prim[prim].append({"primitive": prim, "term": phrase, "score": float(sc), "t_center": float(t_center)})
    # keep topk per primitive
    for prim in by_prim:
        by_prim[prim].sort(key=lambda z: -z["score"])
        by_prim[prim] = by_prim[prim][:topk_per_prim]
    return by_prim

def spans_gold_only(prompt: str, gold_prim: str, term_vectors: dict, prototypes: dict, emb: Emb,
                    tau_span: float = 0.50, n_max: int = 4, topk_per_prim: int = 3) -> Tuple[List[dict], Dict[str, List[dict]], Dict[str, Any]]:
    """Return (gold_spans, spans_all_map, norm_info) for this prompt."""
    span_map = spans_all(prompt, term_vectors, prototypes, emb, tau_span=tau_span, n_max=n_max, topk_per_prim=topk_per_prim)
    gold_spans = list(span_map.get(gold_prim, []))
    norm = {"n_tokens": len(_norm_tokens(prompt))}
    return gold_spans, span_map, norm

# ------------------ Matched filter ------------------
def kaiser_window(L: int = 160, beta: float = 8.6) -> np.ndarray:
    n = np.arange(L, dtype="float32")
    w = np.i0(beta * np.sqrt(1 - ((2*n)/(L-1) - 1)**2))
    w = w / (np.linalg.norm(w) + 1e-9)
    return w.astype("float32")

def _render_channel(spans: List[dict], q: np.ndarray, T: int, L: int) -> np.ndarray:
    x = np.zeros(T, dtype="float32")
    for sp in spans:
        center = int(float(sp["t_center"]) * T)
        start = max(0, center - L//2)
        end   = min(T, start + L)
        x[start:end] += q[:end-start] * float(sp["score"])
    return x

def matched_filter_scores_all(span_map: Dict[str, List[dict]], q: np.ndarray, T: int, L: int,
                              sigma: float = 0.0, seed: int = 0) -> Tuple[Dict[str, float], Dict[str, float]]:
    scores, nulls = {}, {}
    rng = np.random.default_rng(seed) if sigma > 0.0 else None
    for prim in PRIMS:
        x = _render_channel(span_map.get(prim, []), q, T, L)
        if rng is not None:
            x = x + rng.normal(0.0, sigma, size=x.shape).astype("float32")
        r = np.convolve(x, q[::-1], mode="valid")
        peak = float(r.max()) if r.size else 0.0
        null = float(np.linalg.norm(x) * np.linalg.norm(q))
        scores[prim] = peak
        nulls[prim]  = null
    return scores, nulls

def decide_from_scores(scores: Dict[str, float], nulls: Dict[str, float], tau_rel: float, tau_abs: float) -> List[str]:
    """Return list of accepted primitives (passing absolute and relative gates)."""
    best = max([scores.get(k, 0.0) for k in PRIMS] + [1e-9])
    accepts = []
    for prim in PRIMS:
        s = scores.get(prim, 0.0)
        n = nulls.get(prim, 0.0) + 1e-9
        rel = s / max(best, 1e-9)
        absn = s / n
        if (s >= tau_abs) and (absn >= tau_rel):
            accepts.append((prim, s))
    accepts.sort(key=lambda z: -z[1])
    return [k for k,_ in accepts]

# ------------------ Driver ------------------
def run_audit(prompts: List[str],
              gold_labels: List[str],
              sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
              n_max: int = 4, topk_per_prim: int = 3,
              tau_span: float = 0.50, tau_rel: float = 0.60, tau_abs: float = 0.93,
              L: int = 160, beta: float = 8.6, sigma: float = 0.0,
              competitive_eval: bool = False) -> Dict[str, Any]:
    """
    Returns dict with: rows (per-example audit), metrics (aggregate), and aux (term vectors, etc.).
    Mirrors the intended audit_bench_metrics.py behavior, sans mapper (tautology-free).
    """
    assert len(prompts) == len(gold_labels)
    emb = Emb(sbert_model)
    term_vectors, prototypes = build_term_vectors_and_protos(TERM_BANK, emb)
    q = kaiser_window(L=L, beta=beta)

    rows = []
    n_span_total = 0
    n_span_success = 0
    halluc_count = 0
    accept_counts = []

    for p, g in zip(prompts, gold_labels):
        spans_g, span_map, norm = spans_gold_only(
            prompt=p, gold_prim=g, term_vectors=term_vectors, prototypes=prototypes, emb=emb,
            tau_span=tau_span, n_max=n_max, topk_per_prim=topk_per_prim
        )
        # span accounting
        n_span_total += sum(len(v) for v in span_map.values())
        n_span_success += 1 if len(spans_g) > 0 else 0

        # MF scores
        scores, nulls = matched_filter_scores_all(span_map, q=q, T=256, L=L, sigma=sigma, seed=0)
        accepts = decide_from_scores(scores, nulls, tau_rel=tau_rel, tau_abs=tau_abs)

        # decisions & competitive metrics
        gold_accepted = (g in accepts)
        if competitive_eval:
            accept_counts.append(len(accepts))
            if (not gold_accepted) and any(k != g for k in accepts):
                halluc_count += 1

        rows.append({
            "prompt": p,
            "gold": g,
            "pred": accepts[0] if accepts else "",
            "score": round(scores.get(accepts[0], 0.0) if accepts else 0.0, 6),
            "ok": bool(gold_accepted),
            "reason": "" if gold_accepted else "abstain_or_wrong",
            "spans": spans_g,
            "tags": {"accepts": accepts, "scores": scores, "nulls": nulls, "norm": norm},
        })

    # aggregate metrics
    coverage = (n_span_success / max(1, len(prompts)))
    span_yield_rate = (n_span_success / max(1, n_span_total)) if n_span_total else 0.0
    abstain_count = sum(1 for r in rows if not r["ok"])

    metrics: Dict[str, Any] = {
        "coverage": round(coverage, 6),
        "abstain_rate": round(abstain_count / max(1, len(rows)), 6),
        "span_yield_rate": round(span_yield_rate, 6),
        "abstain_no_span_rate": round(sum(1 for r in rows if (len(r["spans"])==0 and not r["ok"])) / max(1, len(rows)), 6),
        "abstain_with_span_rate": round(sum(1 for r in rows if (len(r["spans"])>0 and not r["ok"])) / max(1, len(rows)), 6),
        "params": {"n_max": n_max, "topk_per_prim": topk_per_prim, "tau_span": tau_span, "tau_rel": tau_rel, "tau_abs": tau_abs, "L": L, "beta": beta, "sigma": sigma},
    }
    if competitive_eval:
        metrics.update({
            "hallucination_rate": round(halluc_count / max(1, len(rows)), 6),
            "multi_accept_rate": round(sum(1 for c in accept_counts if c>1) / max(1, len(rows)), 6),
        })

    aux = {"term_vectors_dim": int(next(iter(term_vectors.values())).shape[1]) if term_vectors else 0}
    return {"rows": rows, "metrics": metrics, "aux": aux}
