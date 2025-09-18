#!/usr/bin/env python3
"""
audit_bench_nb.py — Notebook-parity auditor (matched filter on SBERT spans).

This duplicates the decision style in your notebook:
- Build per-primitive span "traces" from SBERT span hits (1..6-gram).
- Correlate with a Kaiser window (matched filter).
- Decide PASS vs ABSTAIN using relative (tau_rel) and absolute (tau_abs) thresholds.
- By default, **approval == PASS** (no mapper confidence gating, no opposite veto).

You can optionally require mapper label agreement with --strict_mapper_match.
Metrics are computed against the provided gold labels (CSV).

CLI example:
  python3 audit_bench_nb.py \
    --model_path .artifacts/defi_mapper.joblib \
    --prompts_jsonl .artifacts/defi/synth/defi_prompts_120.jsonl \
    --labels_csv    .artifacts/defi/synth/defi_labels_120.csv \
    --sbert sentence-transformers/all-MiniLM-L6-v2 \
    --tau_span 0.55 --tau_rel 0.60 --tau_abs 0.93 \
    --T 720 --L 160 --sigma 0.02 \
    --out_dir .artifacts/defi/audit_bench_nb \
    --strict_mapper_match  # optional
"""
from __future__ import annotations
import argparse, json, csv, time, sys, re, hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import joblib

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


# ---- Simple Emb wrapper (with encode_one) -----------------------------------
class Emb:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64, normalize: bool = True):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not available. `pip install sentence-transformers`.")
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._m = None

    def _ensure_model(self):
        if self._m is None:
            self._m = SentenceTransformer(self.model_name)

    def fit(self, X, y=None):
        self._ensure_model()
        return self

    def transform(self, X):
        self._ensure_model()
        V = self._m.encode(list(X), batch_size=self.batch_size, normalize_embeddings=self.normalize, show_progress_bar=False)
        return np.asarray(V)

    def encode_one(self, s: str):
        return self.transform([s])[0]


# ---- I/O --------------------------------------------------------------------
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

# ---- TERM BANK + Prototypes -------------------------------------------------
TERM_BANK: Dict[str, list[str]] = {
    "deposit_asset":  ["deposit","supply","provide","add liquidity","provide liquidity","top up","put in","add funds","fund","allocate","contribute","add position","add to pool","supply to pool","add into","supply into"],
    "withdraw_asset": ["withdraw","redeem","unstake","remove liquidity","pull out","take out","cash out","exit","remove position","remove from pool","take from","pull from"],
    "swap_asset":     ["swap","convert","trade","exchange","convert into","swap to","swap into","bridge","wrap","unwrap","swap for"],
    "borrow_asset":   ["borrow","draw","open a loan for","open debt","draw down","take a loan","borrow against"],
    "repay_asset":    ["repay","pay back","close out the loan for","settle loan","pay debt","repay debt","close loan"],
    "stake_asset":    ["stake","lock","bond","delegate","lock up","stake into","stake to","stake on","restake"],
    "unstake_asset":  ["unstake","unlock","unbond","undelegate","release","unstake from","unstake out","unstow"],
    "claim_rewards":  ["claim","harvest","collect rewards","claim rewards","collect staking rewards","collect yield","claim yield","harvest rewards","collect incentives"],
}
PRIMS = list(TERM_BANK.keys())

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

# ---- Spans (NB style) -------------------------------------------------------
def tokenize(s: str):
    return s.strip().split()

def spans_from_prompt(prompt, prototypes, emb, tau_span=0.55):
    toks = tokenize(prompt)
    spans = []
    for n in range(1, min(6, len(toks))+1):  # NB uses up to 6-gram
        for i in range(0, len(toks)-n+1):
            s = " ".join(toks[i:i+n])
            e = emb.encode_one(s)
            for k, v in prototypes.items():
                sc = max(0.0, cosine(e, v))
                if sc >= tau_span:
                    t_center = (i + n/2.0) / max(1.0, len(toks))  # normalized [0,1]
                    spans.append({"primitive": k, "term": s, "score": float(np.round(sc,4)), "t_center": float(np.round(t_center,4))})
    # keep top 3 per primitive
    by_prim = {k: [] for k in prototypes.keys()}
    for sp in spans:
        by_prim[sp["primitive"]].append(sp)
    for k in by_prim:
        by_prim[k].sort(key=lambda z: -z["score"])
        by_prim[k] = by_prim[k][:3]
    return by_prim

# ---- Matched filter bits (NB style) -----------------------------------------
def kaiser_window(L=160, beta=8.6):
    n = np.arange(L)
    w = np.i0(beta * np.sqrt(1 - ((2*n)/(L-1) - 1)**2))
    w = w / (np.linalg.norm(w) + 1e-9)
    return w.astype("float32")

def matched_filter_scores(traces, q):
    scores, nulls, peaks = {}, {}, {}
    L = len(q)
    for k, x in traces.items():
        if len(x) < L:
            x = np.pad(x, (0, L - len(x)))
        r = np.convolve(x, q[::-1], mode="valid")
        peak = float(r.max()) if r.size else 0.0
        scores[k] = peak
        nulls[k]  = float(np.sqrt(np.sum(x**2)) * (np.linalg.norm(q))) / max(len(x),1)
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

def audit_from_span_map(prompt: str,
                        primitive_to_term_mapping: dict,
                        T: int = 720,
                        tau_span: float = 0.55,
                        tau_abs: float = 0.93,
                        tau_rel: float = 0.60,
                        sigma: float = 0.02,
                        L: int = 160,
                        fuse_per_primitive: bool = False):
    # 1) init deterministic noise per prompt
    seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    traces = {k: (rng.normal(0.0, sigma, size=T).astype("float32")) for k in primitive_to_term_mapping.keys()}

    # 2) stamp span kernels into traces
    q = kaiser_window(L=L)  # unit-norm
    for k, spans in primitive_to_term_mapping.items():
        if not spans: 
            continue
        # Optionally fuse multiple spans by summing kernels; otherwise stamp each
        if fuse_per_primitive:
            x = traces[k]
            for sp in spans:
                center = int(sp["t_center"] * T)
                start = max(0, center - L//2)
                end   = min(T, start + L)
                w = q[:end-start] * float(sp["score"])
                x[start:end] += w
        else:
            x = traces[k]
            for sp in spans:
                center = int(sp["t_center"] * T)
                start = max(0, center - L//2)
                end   = min(T, start + L)
                w = q[:end-start] * float(sp["score"])
                x[start:end] += w
        traces[k] = x

    # 3) matched filter + decision
    scores, nulls, peaks = matched_filter_scores(traces, q)
    sequence, accepted = decide(scores, nulls, tau_rel=tau_rel, tau_abs=tau_abs)

    return {
        "decision": "PASS" if sequence else "ABSTAIN",
        "sequence": sequence,
        "accepted_peaks": accepted,
        "peaks": peaks,
        "notes": {"tau_span": tau_span, "tau_rel": tau_rel, "tau_abs": tau_abs, "sigma": sigma, "T": T, "L": L, "fused": fuse_per_primitive},
    }

# ---- Mapper interop & metrics -----------------------------------------------
def load_mapper(path: str):
    return joblib.load(path)

def mapper_top1_label(mapper, prompt: str) -> Tuple[str | None, float]:
    if hasattr(mapper, "predict_proba"):
        probs = mapper.predict_proba([prompt])[0]
        classes = list(getattr(mapper, "classes_", PRIMS))
        top_idx = int(np.argmax(probs))
        return classes[top_idx], float(probs[top_idx])
    elif hasattr(mapper, "predict"):
        lab = mapper.predict([prompt])[0]
        return str(lab), 1.0
    return None, 0.0

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
            W.writerow([p, g, o["mapper"]["top"] or "", f'{o["mapper"]["conf"]:.4f}', "|".join(o["audit"]["sequence"]), o["audit"]["decision"], str(o["decision"]=="approve")])

# ---- Runner -----------------------------------------------------------------
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

    outs = []
    for p in prompts:
        span_map = spans_from_prompt(p, prototypes, emb, tau_span=args.tau_span)
        audit = audit_from_span_map(p, span_map, T=args.T, tau_span=args.tau_span, tau_abs=args.tau_abs, tau_rel=args.tau_rel, sigma=args.sigma, L=args.L, fuse_per_primitive=args.fuse_per_primitive)

        m_top, m_conf = mapper_top1_label(mapper, p)

        # NB parity: approve == PASS
        approve = (audit["decision"] == "PASS")

        # optional stricter mode: require mapper label among accepted sequence (usually the first)
        if args.strict_mapper_match and approve:
            approve = (m_top in set(audit["sequence"]))

        outs.append({
            "prompt": p,
            "decision": "approve" if approve else "reject",
            "mapper": {"top": m_top, "conf": m_conf},
            "audit": audit,
        })

    # metrics
    M = compute_metrics(gold, outs)
    (out_dir / "metrics.csv").write_text(
        "total,approved,rejected,coverage,abstain_rate,accuracy_on_approved,overall_accuracy\n" +
        f'{M["total"]},{M["approved"]},{M["rejected"]},{M["coverage"]},{M["abstain_rate"]},{M["accuracy_on_approved"]},{M["overall_accuracy"]}\n'
    )
    write_rows_csv(out_dir / "rows_nb.csv", prompts, gold, outs)
    (out_dir / "summary.json").write_text(json.dumps({"bench":"audit_bench_nb","params":vars(args),"metrics":M}, indent=2))

    print(f'[audit_bench_nb] DONE. overall_acc={M["overall_accuracy"]:.3f} coverage={M["coverage"]:.3f} acc_on_approved={M["accuracy_on_approved"]:.3f}')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=".artifacts/defi_mapper.joblib")
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--labels_csv",    required=False)
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--tau_span", type=float, default=0.55)
    ap.add_argument("--tau_rel", type=float, default=0.60)
    ap.add_argument("--tau_abs", type=float, default=0.93)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--L", type=int, default=160, help="Kaiser window length")
    ap.add_argument("--fuse_per_primitive", action="store_true")
    ap.add_argument("--strict_mapper_match", action="store_true", help="require mapper label to appear in audit sequence to approve")
    ap.add_argument("--out_dir", default=".artifacts/defi/audit_bench_nb")
    args = ap.parse_args()
    raise SystemExit(run(args))

if __name__ == "__main__":
    main()
