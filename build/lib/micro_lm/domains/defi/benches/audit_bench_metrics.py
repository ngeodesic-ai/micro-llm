"""
# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Copyright 2025 Ian C. Moore (Provisional Patents #63/864,726, #63/865,437, #63/871,647 and #63/872,334)
# Email: ngeodesic@gmail.com
# Part of Noetic Geodesic Framework (NGF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from pathlib import Path

"""
audit_bench_metrics.py — Solid audit bench with rich metrics (gold-only + optional competitive eval)

What it does
------------
- Gold-only audit (default): for each prompt, build spans ONLY for the gold class, stamp a Kaiser
  window, matched-filter, and decide PASS/ABSTAIN for the gold class.
- Optional competitive pass (--competitive_eval): also audit across ALL classes (spans for all),
  then report hallucination and multi-accept rates (doesn't affect gold-only PASS/ABSTAIN).

Key metrics written to metrics_audit.json
-----------------------------------------
- coverage (pass_rate): fraction approved (PASS) by the gold-only audit.
- abstain_rate: 1 - coverage.
- span_yield_rate: fraction with ANY spans for the gold class (before MF gate).
- abstain_no_span_rate: fraction ABSTAIN due to no spans for gold (lexicon issue).
- abstain_with_span_rate: fraction ABSTAIN with spans present but MF gate failed (threshold issue).
- peak/rel means split by PASS/ABSTAIN (sanity trends).
- per_class coverage & span_yield_rate.
- (optional) hallucination_rate: in competitive pass, audit accepted non-gold class(es) and NOT gold.
- (optional) multi_accept_rate: in competitive pass, audit accepted more than one class.

Outputs
-------
- rows_audit.csv: per-example details (gold-only decision and optional competitive sequence)
- metrics_audit.json: metrics summary

Example
-------
PYTHONWARNINGS="ignore::FutureWarning" \
python3 src/micro_lm/domains/defi/benches/audit_bench_metrics.py \
  --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl \
  --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv \
  --sbert sentence-transformers/all-MiniLM-L6-v2 \
  --n_max 4 --tau_span 0.50 --tau_rel 0.60 --tau_abs 0.93 \
  --out_dir .artifacts/defi/audit_bench \
  --competitive_eval

PYTHONWARNINGS="ignore::FutureWarning" \
python3 src/micro_lm/domains/defi/benches/audit_bench_metrics.py \
  --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl \
  --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv \
  --sbert sentence-transformers/all-MiniLM-L6-v2 \
  --n_max 4 --tau_span 0.50 --tau_rel 0.60 --tau_abs 0.93 \
  --out_dir .artifacts/defi/audit_bench \
  --competitive_eval
  
"""

import argparse, csv, json, hashlib, re, os
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None


# ---------------- I/O ----------------
def read_prompts(fp: Path) -> List[str]:
    txt = fp.read_text().strip()
    # Support both json array and jsonl
    if txt.startswith('['):
        arr = json.loads(txt)
        out = []
        for row in arr:
            if isinstance(row, dict) and "prompt" in row:
                out.append(row["prompt"])
            else:
                out.append(str(row))
        return out
    # jsonl
    outs = []
    for line in txt.splitlines():
        if not line.strip():
            continue
        J = json.loads(line)
        outs.append(J["prompt"] if isinstance(J, dict) and "prompt" in J else line.strip())
    return outs

def read_labels_csv(fp: Path) -> List[str]:
    out = []
    with fp.open() as f:
        R = csv.DictReader(f)
        key = "label" if "label" in R.fieldnames else R.fieldnames[-1]
        for r in R:
            out.append(r[key])
    return out


# ------------- TERM BANK + Embeddings -------------
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


# ------------- Spans + Matched Filter -------------
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


# ------------- Main run -------------
def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir or ".artifacts/defi/audit_bench")
    out_dir.mkdir(parents=True, exist_ok=True)

    NTEST = 200
    prompts = read_prompts(Path(args.prompts_jsonl))
    gold = read_labels_csv(Path(args.labels_csv))
    gold = gold[0:NTEST]
    prompts = prompts[0:NTEST]
    assert len(prompts) == len(gold), "prompts and labels must align"

    emb = Emb(args.sbert)
    term_vectors, prototypes = build_term_vectors_and_protos(TERM_BANK, emb)
    q = kaiser_window(L=args.L, beta=args.beta)

    rows = []
    N = len(prompts)
    pass_cnt = 0
    span_yield_cnt = 0
    abstain_no_span = 0
    abstain_with_span = 0

    for idx, (p, g) in enumerate(zip(prompts, gold)):
        seed = int(hashlib.sha256(p.encode("utf-8")).hexdigest()[:8], 16)

        # Gold-only spans and decision
        spans_g = spans_gold_only(p, g, term_vectors, prototypes, emb, tau_span=args.tau_span, n_max=args.n_max, topk=args.topk_per_prim)
        had_spans = len(spans_g) > 0
        if had_spans:
            span_yield_cnt += 1

        peak, rel, null = matched_filter_score_for_prim(spans_g, q, T=args.T, L=args.L, sigma=args.sigma, seed=seed)
        decision = "PASS" if (peak >= args.tau_abs and rel >= args.tau_rel) else "ABSTAIN"

        if decision == "PASS":
            pass_cnt += 1
        else:
            if not had_spans:
                abstain_no_span += 1
            else:
                abstain_with_span += 1

        row = {
            "idx": idx,
            "prompt": p,
            "gold": g,
            "had_spans": bool(had_spans),
            "decision": decision,
            "reason": ("pass" if decision=="PASS" else ("no_span" if not had_spans else "below_threshold")),
            "peak": round(peak, 4),
            "rel": round(rel, 4),
            "null": round(null, 4),
            "span_terms": "|".join([sp["term"] for sp in spans_g]) if had_spans else "",
        }

        # Optional competitive evaluation (for hallucination & multi-accept metrics)
        if args.competitive_eval:
            span_map_all = spans_all(p, term_vectors, prototypes, emb, tau_span=args.tau_span, n_max=args.n_max, topk_per_prim=args.topk_per_prim)
            scores_all, nulls_all = matched_filter_scores_all(span_map_all, q, T=args.T, L=args.L, sigma=args.sigma, seed=seed)
            comp_seq = decide_from_scores(scores_all, nulls_all, tau_rel=args.tau_rel, tau_abs=args.tau_abs)
            row["comp_seq"] = "|".join(comp_seq)
            row["hallucinated_non_gold"] = bool(comp_seq and (g not in comp_seq))
            row["multi_accept"] = (len(comp_seq) > 1)

        rows.append(row)

    # ----- Metrics -----
    coverage = pass_cnt / N if N else 0.0
    abstain_rate = 1.0 - coverage
    span_yield_rate = span_yield_cnt / N if N else 0.0
    abstain_no_span_rate = abstain_no_span / N if N else 0.0
    abstain_with_span_rate = abstain_with_span / N if N else 0.0

    # Per-class
    per_class = defaultdict(lambda: {"n":0, "pass":0, "span_yield":0})
    for r in rows:
        c = r["gold"]
        per_class[c]["n"] += 1
        per_class[c]["pass"] += int(r["decision"]=="PASS")
        per_class[c]["span_yield"] += int(r["had_spans"]==True)
    per_class_out = {
        k: {
            "n": v["n"],
            "coverage": (v["pass"]/v["n"] if v["n"] else 0.0),
            "span_yield_rate": (v["span_yield"]/v["n"] if v["n"] else 0.0)
        } for k,v in per_class.items()
    }

    metrics = {
        "total": N,
        "coverage": coverage,
        "abstain_rate": abstain_rate,
        "span_yield_rate": span_yield_rate,
        "abstain_no_span_rate": abstain_no_span_rate,
        "abstain_with_span_rate": abstain_with_span_rate,
        "peak_mean_pass": float(np.mean([r["peak"] for r in rows if r["decision"]=="PASS"])) if pass_cnt else 0.0,
        "peak_mean_abstain": float(np.mean([r["peak"] for r in rows if r["decision"]=="ABSTAIN"])) if (N-pass_cnt) else 0.0,
        "rel_mean_pass": float(np.mean([r["rel"] for r in rows if r["decision"]=="PASS"])) if pass_cnt else 0.0,
        "rel_mean_abstain": float(np.mean([r["rel"] for r in rows if r["decision"]=="ABSTAIN"])) if (N-pass_cnt) else 0.0,
        "per_class": per_class_out,
        "params": {
            "sbert": args.sbert,
            "tau_span": args.tau_span, "tau_rel": args.tau_rel, "tau_abs": args.tau_abs,
            "n_max": args.n_max, "topk_per_prim": args.topk_per_prim,
            "T": args.T, "L": args.L, "beta": args.beta, "sigma": args.sigma
        }
    }

    if args.competitive_eval:
        halluc = sum(1 for r in rows if r.get("hallucinated_non_gold"))
        multi  = sum(1 for r in rows if r.get("multi_accept"))
        metrics.update({
            "hallucination_rate": (halluc / N if N else 0.0),
            "multi_accept_rate":  (multi / N if N else 0.0)
        })

    # ----- Write outputs -----
    rows_fp = out_dir / "rows_audit.csv"
    with rows_fp.open("w", newline="") as f:
        W = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        W.writeheader()
        for r in rows:
            W.writerow(r)

    (out_dir / "metrics_audit.json").write_text(json.dumps(metrics, indent=2))

    # Console summary
    print(json.dumps(metrics, indent=2))
    print(f"Total PASS {pass_cnt} / {N}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--n_max", type=int, default=4, help="max n-gram length to scan (1..n_max)")
    ap.add_argument("--topk_per_prim", type=int, default=3, help="keep top-K spans per class")
    ap.add_argument("--tau_span", type=float, default=0.50, help="span acceptance threshold vs class terms/prototype")
    ap.add_argument("--tau_rel", type=float, default=0.60, help="rel gate: peak / null")
    ap.add_argument("--tau_abs", type=float, default=0.93, help="abs gate: peak")
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--L", type=int, default=160)
    ap.add_argument("--beta", type=float, default=8.6)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--out_dir", default=".artifacts/defi/audit_bench")
    ap.add_argument("--competitive_eval", action="store_true", help="also run a competitive audit to measure hallucination & multi-accept rates")
    args = ap.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
