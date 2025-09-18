# Create a "gold-only" auditor that mirrors the manual behavior and also writes metrics/rows.
from pathlib import Path

"""
audit_bench_gold_only.py — gold-only auditor (matches the "manual" style)

Behavior:
- For each prompt, build SBERT spans **only** for its gold label class (others left empty).
- Convert spans to per-primitive traces via a Kaiser window and run a matched filter.
- PASS iff (score >= tau_abs) AND (score/null >= tau_rel) for the gold class.
- Approval == PASS (no mapper-confidence gate, no opposite veto).
- Prints per-example lines like the manual script and a final "Total PASS X / N".
- Also writes rows.csv and metrics.json under --out_dir.

CLI example:
  python3 audit_bench_gold_only.py \
    --model_path .artifacts/defi_mapper.joblib \
    --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.json \
    --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv \
    --sbert sentence-transformers/all-MiniLM-L6-v2 \
    --n_max 6 --tau_span 0.55 --tau_rel 0.60 --tau_abs 0.93 \
    --T 720 --L 160 --beta 8.6 --sigma 0.0 \
    --out_dir .artifacts/defi/audit_bench
"""
from __future__ import annotations
import argparse, csv, json, hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

# ------------- I/O -------------
def read_prompts(fp: Path) -> List[str]:
    txt = fp.read_text().strip()
    # Support both jsonl and json array
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

def load_mapper(path: str):
    return joblib.load(path)

def mapper_top1_label(mapper, prompt: str) -> Tuple[str|None,float]:
    if hasattr(mapper, "predict_proba"):
        probs = mapper.predict_proba([prompt])[0]
        classes = list(getattr(mapper, "classes_", []))
        top = int(np.argmax(probs))
        return (classes[top] if classes else None), float(probs[top])
    elif hasattr(mapper, "predict"):
        lab = mapper.predict([prompt])[0]
        return str(lab), 1.0
    return None, 0.0

# ------------- TERM BANK + prototypes -------------
TERM_BANK: Dict[str, List[str]] = {
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

class Emb:
    def __init__(self, name: str, batch_size: int = 64, normalize: bool = True):
        self.m = SentenceTransformer(name)
        self.batch = batch_size
        self.norm = normalize
    def transform(self, X: List[str]) -> np.ndarray:
        V = self.m.encode(list(X), batch_size=self.batch, normalize_embeddings=self.norm, show_progress_bar=False)
        return np.asarray(V)
    def encode_one(self, s: str) -> np.ndarray:
        return self.transform([s])[0]

def build_prototypes(term_bank: Dict[str, List[str]], emb: Emb) -> Dict[str, np.ndarray]:
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
def tokenize_ws(s: str) -> List[str]:
    return s.strip().split()

def spans_from_prompt(prompt: str, prim: str, protos: Dict[str,np.ndarray], emb: Emb, tau_span: float = 0.55, n_max: int = 6, topk: int = 3):
    # gold-only spans: compute only for 'prim'
    toks = tokenize_ws(prompt)
    spans = []
    for n in range(1, min(n_max, len(toks)) + 1):
        for i in range(0, len(toks) - n + 1):
            phrase = " ".join(toks[i:i+n])
            e = emb.encode_one(phrase)
            sc = max(0.0, cosine(e, protos[prim]))
            if sc >= tau_span:
                t_center = (i + n/2.0) / max(1.0, len(toks))
                spans.append({"primitive": prim, "term": phrase, "score": float(sc), "t_center": float(t_center)})
    spans.sort(key=lambda z: -z["score"])
    return spans[:topk]

def kaiser_window(L=160, beta=8.6) -> np.ndarray:
    n = np.arange(L, dtype="float32")
    w = np.i0(beta * np.sqrt(1 - ((2*n)/(L-1) - 1)**2))
    w = w / (np.linalg.norm(w) + 1e-9)  # unit-norm
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

# ------------- run -------------
def run(args: argparse.Namespace) -> int:
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    prompts = read_prompts(Path(args.prompts_jsonl))
    gold = read_labels_csv(Path(args.labels_csv))
    assert len(prompts) == len(gold), "prompts and labels must align"

    emb = Emb(args.sbert)
    protos = build_prototypes(TERM_BANK, emb)
    q = kaiser_window(L=args.L, beta=args.beta)

    N = len(prompts)
    print(f"N: {N}")
    pass_cnt = 0

    # Outputs
    rows = []

    for k, (test_prompt, gold_prim) in enumerate(zip(prompts, gold)):
        # gold-only spans
        spans = spans_from_prompt(test_prompt, gold_prim, protos, emb, tau_span=args.tau_span, n_max=args.n_max)

        # matched filter against gold class only
        seed = int(hashlib.sha256(test_prompt.encode("utf-8")).hexdigest()[:8], 16)
        peak, rel, null = matched_filter_score_for_prim(spans, q, T=args.T, L=args.L, sigma=args.sigma, seed=seed)

        is_pass = ("PASS" if (peak >= args.tau_abs and rel >= args.tau_rel) else "ABSTAIN")
        print(f"{k} {is_pass} / prompt: {test_prompt} / primitives: {[gold_prim if spans else '']}")

        if is_pass == "PASS":
            pass_cnt += 1

        rows.append({
            "idx": k,
            "prompt": test_prompt,
            "gold": gold_prim,
            "span_terms": "|".join([sp["term"] for sp in spans]) if spans else "",
            "peak": round(peak, 4),
            "rel": round(rel, 4),
            "null": round(null, 4),
            "decision": is_pass
        })

    print(f"Total PASS {pass_cnt} / {N}")

    # Write rows + metrics
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "rows_gold_only.csv", index=False)

    metrics = {
        "total": N,
        "pass": pass_cnt,
        "pass_rate": pass_cnt / N if N else 0.0,
        "tau_span": args.tau_span,
        "tau_rel": args.tau_rel,
        "tau_abs": args.tau_abs,
        "n_max": args.n_max,
        "L": args.L, "T": args.T, "beta": args.beta, "sigma": args.sigma,
    }
    (out / "metrics_gold_only.json").write_text(json.dumps(metrics, indent=2))
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=False, help="kept for API parity (unused)")
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--n_max", type=int, default=6)
    ap.add_argument("--tau_span", type=float, default=0.55)
    ap.add_argument("--tau_rel", type=float, default=0.60)
    ap.add_argument("--tau_abs", type=float, default=0.93)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--L", type=int, default=160)
    ap.add_argument("--beta", type=float, default=8.6)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--out_dir", default=".artifacts/defi/audit_bench")
    args = ap.parse_args()
    raise SystemExit(run(args))

if __name__ == "__main__":
    main()

