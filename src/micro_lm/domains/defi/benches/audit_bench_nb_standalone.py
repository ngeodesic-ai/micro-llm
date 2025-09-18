#!/usr/bin/env python3
"""
audit_bench_nb_standalone.py — self-contained "notebook-style" auditor (no notebook dependency)

Implements the matched-filter span auditor you described:
- Build SBERT prototypes per primitive from a TERM_BANK.
- Collect 1..N-gram spans from each prompt with SBERT scores.
- Stamp a Kaiser window at each span location to build per-primitive traces.
- Matched-filter the traces; PASS if both absolute and relative thresholds are met.
- By default, approval == PASS (no mapper confidence gating or opposite veto).

Outputs:
  - rows_nb.csv (prompt, gold, mapper_top/conf, audit sequence, PASS/ABSTAIN, approved)
  - metrics.csv, summary.json

Example:
  python3 audit_bench_nb_standalone.py \
    --model_path .artifacts/defi_mapper.joblib \
    --prompts_jsonl .artifacts/defi/synth/defi_prompts_120.jsonl \
    --labels_csv    .artifacts/defi/synth/defi_labels_120.csv \
    --sbert sentence-transformers/all-MiniLM-L6-v2 \
    --n_max 6 --tau_span 0.55 --tau_rel 0.60 --tau_abs 0.93 \
    --T 720 --L 160 --beta 8.6 --sigma 0.0 \
    --out_dir .artifacts/defi/audit_bench_nb
"""
from __future__ import annotations
import argparse, json, csv, sys, re, hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import joblib

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ---------------- I/O ----------------
def read_prompts_jsonl(fp: Path) -> List[str]:
    rows = []
    with fp.open() as f:
        for line in f:
            J = json.loads(line)
            rows.append(J["prompt"] if isinstance(J, dict) and "prompt" in J else line.strip())
    return rows

def read_labels_csv(fp: Path) -> List[str]:
    out = []
    with fp.open() as f:
        R = csv.DictReader(f)
        lab_key = "label" if "label" in R.fieldnames else R.fieldnames[-1]
        for r in R:
            out.append(r[lab_key])
    return out

def load_mapper(path: str):
    return joblib.load(path)

def mapper_top1_label(mapper, prompt: str) -> Tuple[str | None, float]:
    if hasattr(mapper, "predict_proba"):
        probs = mapper.predict_proba([prompt])[0]
        classes = list(getattr(mapper, "classes_", []))
        top_idx = int(np.argmax(probs))
        top = classes[top_idx] if classes else None
        return top, float(probs[top_idx])
    elif hasattr(mapper, "predict"):
        lab = mapper.predict([prompt])[0]
        return str(lab), 1.0
    return None, 0.0

# ------------- TERM BANK + prototypes -------------
TERM_BANK: Dict[str, list[str]] = {
    "deposit_asset":  ["deposit","supply","provide","add liquidity","provide liquidity","top up","put in","add funds","fund","allocate","contribute","add position","add to pool","supply to pool","add into","supply into"],
    "withdraw_asset": ["withdraw","redeem","unstake","remove liquidity","pull out","take out","cash out","exit","remove position","remove from pool","take from","pull from"],
    "swap_asset":     ["swap","convert","trade","exchange","convert into","swap to","swap into","bridge","wrap","unwrap","swap for"],
    "borrow_asset":   ["borrow","draw","open a loan for","open debt","draw down","take a loan","borrow against"],
    "repay_asset":    ["repay","pay back","close out the loan for","settle loan","pay debt","repay debt","close loan"],
    "stake_asset":    ["stake","lock","bond","delegate","lock up","stake into","stake to","stake on"],
    "unstake_asset":  ["unstake","unlock","unbond","undelegate","release","unstake from","unstake out"],
    "claim_rewards":  ["claim","harvest","collect rewards","claim rewards","collect staking rewards","collect yield","claim yield","harvest rewards"],
}
PRIMS = list(TERM_BANK.keys())

class Emb:
    def __init__(self, model_name: str, batch_size: int = 64, normalize: bool = True):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not available. `pip install sentence-transformers`.")
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._m = SentenceTransformer(model_name)
    def transform(self, X):
        V = self._m.encode(list(X), batch_size=self.batch_size, normalize_embeddings=self.normalize, show_progress_bar=False)
        return np.asarray(V)
    def encode_one(self, s: str):
        return self.transform([s])[0]

def build_prototypes(term_bank: Dict[str, list[str]], emb: Emb) -> Dict[str, np.ndarray]:
    protos = {}
    for prim, terms in term_bank.items():
        V = emb.transform(terms)
        protos[prim] = V.mean(axis=0)
    return protos

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

# ------------- spans + matched filter -------------
def tokenize_raw(s: str) -> List[str]:
    # Simple, notebook-like tokenization: split on whitespace, keep case/punct as-is
    return s.strip().split()

def spans_from_prompt(prompt: str, prototypes: Dict[str, np.ndarray], emb: Emb, tau_span: float = 0.55, n_max: int = 6, topk_per_prim: int = 3):
    toks = tokenize_raw(prompt)
    spans = []
    for n in range(1, min(n_max, len(toks)) + 1):
        for i in range(0, len(toks) - n + 1):
            phrase = " ".join(toks[i:i+n])
            e = emb.encode_one(phrase)
            for k, proto in prototypes.items():
                sc = max(0.0, cosine(e, proto))
                if sc >= tau_span:
                    t_center = (i + n/2.0) / max(1.0, len(toks))
                    spans.append({"primitive": k, "term": phrase, "score": float(sc), "t_center": float(t_center)})
    # keep top-K per primitive
    by_prim = {k: [] for k in prototypes.keys()}
    for sp in spans:
        by_prim[sp["primitive"]].append(sp)
    for k in by_prim:
        by_prim[k].sort(key=lambda z: -z["score"])
        by_prim[k] = by_prim[k][:topk_per_prim]
    return by_prim

def kaiser_window(L=160, beta=8.6):
    n = np.arange(L, dtype="float32")
    w = np.i0(beta * np.sqrt(1 - ((2*n)/(L-1) - 1)**2))
    w = w / (np.linalg.norm(w) + 1e-9)  # unit-norm
    return w.astype("float32")

def matched_filter_scores(traces: Dict[str, np.ndarray], q: np.ndarray):
    scores, nulls, peaks = {}, {}, {}
    q_rev = q[::-1]
    nq = float(np.linalg.norm(q))
    for k, x in traces.items():
        # conv valid
        r = np.convolve(x, q_rev, mode="valid")
        peak = float(r.max()) if r.size else 0.0
        scores[k] = peak
        nulls[k]  = float(np.linalg.norm(x) * nq)  # simple null proportional to energy
        peaks[k]  = {"score": peak, "t_idx": int(np.argmax(r)) if r.size else 0}
    return scores, nulls, peaks

def decide(scores: Dict[str, float], nulls: Dict[str, float], tau_rel: float = 0.60, tau_abs: float = 0.93):
    accepted, seq = {}, []
    for k in scores:
        s = scores[k]; n = nulls[k] + 1e-9
        rel = s / n
        if (s >= tau_abs) and (rel >= tau_rel):
            accepted[k] = {"score": round(s,4), "rel": round(rel,3), "null": round(n,4)}
            seq.append((k, s))
    seq.sort(key=lambda z: -z[1])
    return [k for k,_ in seq], accepted

# ------------- metrics + output -------------
def compute_metrics(gold: List[str] | None, outs: List[dict]) -> Dict[str, Any]:
    total = len(outs)
    approved = sum(1 for o in outs if o["decision"] == "approve")
    rejected = total - approved
    coverage = approved / total if total else 0.0
    abstain_rate = rejected / total if total else 0.0

    acc_on_approved = -1.0
    overall_acc = -1.0
    if gold is not None and len(gold) == total:
        correct_on_approved = 0
        for g, o in zip(gold, outs):
            if o["decision"] == "approve" and o["mapper"]["top"] == g:
                correct_on_approved += 1
        acc_on_approved = (correct_on_approved / approved) if approved else 0.0
        overall_acc = (correct_on_approved / total) if total else 0.0

    return {
        "total": total,
        "approved": approved,
        "rejected": rejected,
        "coverage": coverage,
        "abstain_rate": abstain_rate,
        "accuracy_on_approved": acc_on_approved,
        "overall_accuracy": overall_acc,
    }

def write_rows_csv(fp: Path, prompts: List[str], gold: List[str] | None, outs: List[dict]):
    with fp.open("w", newline="") as f:
        W = csv.writer(f)
        W.writerow(["prompt","gold_label","mapper_top","mapper_conf","audit_sequence","audit_decision","approved"])
        for i, p in enumerate(prompts):
            g = (gold[i] if gold is not None and i < len(gold) else "")
            o = outs[i]
            seq = "|".join(o["audit"].get("sequence", []))
            W.writerow([p, g, o["mapper"]["top"] or "", f'{o["mapper"]["conf"]:.4f}', seq, o["audit"]["decision"], str(o["decision"]=="approve")])

# ------------- main run -------------
def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir or ".artifacts/defi/audit_bench_nb")
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts_jsonl(Path(args.prompts_jsonl))
    gold = read_labels_csv(Path(args.labels_csv)) if args.labels_csv else None
    if gold is not None:
        assert len(prompts) == len(gold), "prompts and labels must align"

    mapper = load_mapper(args.model_path)
    emb = Emb(args.sbert)
    prototypes = build_prototypes(TERM_BANK, emb)

    q = kaiser_window(L=args.L, beta=args.beta)

    outs = []
    for p in prompts:
        # deterministic tiny noise if requested
        if args.sigma > 0.0:
            seed = int(hashlib.sha256(p.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)

        spans = spans_from_prompt(p, prototypes, emb, tau_span=args.tau_span, n_max=args.n_max)
        traces = {k: np.zeros(args.T, dtype="float32") for k in prototypes.keys()}
        # optional base noise
        if args.sigma > 0.0:
            for k in traces:
                traces[k] += rng.normal(0.0, args.sigma, size=args.T).astype("float32")

        for k, sps in spans.items():
            for sp in sps:
                center = int(float(sp["t_center"]) * args.T)
                start = max(0, center - args.L//2)
                end   = min(args.T, start + args.L)
                w = q[:end-start] * float(sp["score"])
                traces[k][start:end] += w

        scores, nulls, peaks = matched_filter_scores(traces, q)
        seq, accepted = decide(scores, nulls, tau_rel=args.tau_rel, tau_abs=args.tau_abs)

        m_top, m_conf = mapper_top1_label(mapper, p)

        approve = (len(seq) > 0)
        if args.strict_mapper_match and approve:
            approve = (m_top in set(seq))

        outs.append({
            "prompt": p,
            "decision": "approve" if approve else "reject",
            "mapper": {"top": m_top, "conf": m_conf},
            "audit": {"decision": "PASS" if approve else "ABSTAIN", "sequence": seq, "accepted_peaks": accepted, "peaks": peaks}
        })

    M = compute_metrics(gold, outs)
    write_rows_csv(out_dir / "rows_nb.csv", prompts, gold, outs)
    (out_dir / "metrics.csv").write_text(
        "total,approved,rejected,coverage,abstain_rate,accuracy_on_approved,overall_accuracy\n" +
        f'{M["total"]},{M["approved"]},{M["rejected"]},{M["coverage"]},{M["abstain_rate"]},{M["accuracy_on_approved"]},{M["overall_accuracy"]}\n'
    )
    (out_dir / "summary.json").write_text(json.dumps({"bench":"audit_bench_nb_standalone","params":vars(args),"metrics":M}, indent=2))
    print(f'[audit_bench_nb_standalone] DONE. overall_acc={M["overall_accuracy"]:.3f} coverage={M["coverage"]:.3f} acc_on_approved={M["accuracy_on_approved"]:.3f}')
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--labels_csv",    required=False)
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--n_max", type=int, default=6)
    ap.add_argument("--tau_span", type=float, default=0.55)
    ap.add_argument("--tau_rel", type=float, default=0.60)
    ap.add_argument("--tau_abs", type=float, default=0.93)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--L", type=int, default=160)
    ap.add_argument("--beta", type=float, default=8.6)
    ap.add_argument("--sigma", type=float, default=0.0, help="add small deterministic noise; default 0.0 (off)")
    ap.add_argument("--strict_mapper_match", action="store_true", help="require mapper label to appear in audit sequence to approve")
    ap.add_argument("--out_dir", default=".artifacts/defi/audit_bench_nb")
    args = ap.parse_args()
    raise SystemExit(run(args))

if __name__ == "__main__":
    main()
