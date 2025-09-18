#!/usr/bin/env python3
"""
Audit Bench — Rails-as-Auditor (Tier‑1 fix)
- Uses *independent* SBERT latents + lexical span evidence to audit the mapper.
- Run mapper_bench.py (SBERT backend) first to train a mapper .joblib.
- Then run this to *audit* mapper outputs using spans + prototype similarity.

Outputs (under --out_dir, default: .artifacts/defi/audit_bench):
  - rows_thr_<conf>.csv     : per-prompt decisions and details
  - metrics.csv             : coverage, abstain rate, accuracy on approved
  - report.md               : human-readable summary
  - summary.json            : chosen config + timestamp

Example:
  python3 src/micro_lm/domains/defi/benches/audit_bench.py \
    --model_path .artifacts/defi_mapper.joblib \
    --prompts_jsonl tests/fixtures/defi/mapper_smoke_prompts.jsonl \
    --labels_csv    tests/fixtures/defi/mapper_smoke_labels.csv \
    --sbert sentence-transformers/all-MiniLM-L6-v2 \
    --conf_thr 0.70 --tau_span 0.55 --rel_margin 0.06 \
    --out_dir .artifacts/defi/audit_bench

PYTHONWARNINGS="ignore::FutureWarning" python3 src/micro_lm/domains/defi/benches/audit_bench.py
python3 src/micro_lm/domains/defi/benches/audit_bench.py \
  --model_path .artifacts/defi_mapper.joblib \
  --prompts_jsonl tests/fixtures/defi/mapper_smoke_prompts.jsonl \
  --labels_csv    tests/fixtures/defi/mapper_smoke_labels.csv \
  --sbert sentence-transformers/all-MiniLM-L6-v2 \
  --conf_thr 0.70 --tau_span 0.55 --rel_margin 0.06 \
  --out_dir .artifacts/defi/audit_bench

python3 src/micro_lm/domains/defi/benches/audit_bench.py \
  --model_path .artifacts/defi_mapper.joblib \
  --prompts_jsonl tests/fixtures/defi/mapper_smoke_prompts.jsonl \
  --labels_csv    tests/fixtures/defi/mapper_smoke_labels.csv \
  --sbert sentence-transformers/all-MiniLM-L6-v2 \
  --conf_thr 0.70 --tau_span 0.55 --rel_margin 0.06 \
  --out_dir .artifacts/defi/audit_bench \
  --min_overall_acc 0.80

PYTHONWARNINGS="ignore::FutureWarning" \
    python3 src/micro_lm/domains/defi/benches/audit_bench.py \
      --model_path .artifacts/defi_mapper.joblib \
      --prompts_jsonl tests/fixtures/defi/mapper_smoke_prompts.jsonl \
      --labels_csv    tests/fixtures/defi/mapper_smoke_labels.csv \
      --sbert sentence-transformers/all-MiniLM-L6-v2 \
      --conf_thr 0.70 --tau_span 0.55 --rel_margin 0.06 \
      --out_dir .artifacts/defi/audit_bench \
      --min_overall_acc 0.80

PYTHONWARNINGS="ignore::FutureWarning" \
    python3 src/micro_lm/domains/defi/benches/audit_bench.py \
      --model_path .artifacts/defi_mapper.joblib \
      --prompts_jsonl .artifacts/defi/synth/defi_prompts_120.jsonl \
      --labels_csv    .artifacts/defi/synth/defi_labels_120.csv \
      --sbert sentence-transformers/all-MiniLM-L6-v2 \
      --conf_thr 0.70 --tau_span 0.55 --rel_margin 0.06 \
      --out_dir .artifacts/defi/audit_bench_120 \
      --min_overall_acc 0.80

Key idea (from "Rails Tautology Fix"):
  Rails must *not* fabricate latents from mapper labels. They must audit using
  text-derived evidence (spans + SBERT prototypes). Mapper narrows candidates;
  rails decide from evidence and can veto/abstain.
"""
from __future__ import annotations
import argparse, json, csv, time, sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import joblib

# Light dependency: sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None  # allow import even if not installed; runtime will fail gracefully


# ----- Domain term bank & opposites (simple, tight seed) ---------------------
TERM_BANK: Dict[str, list[str]] = {
    "deposit_asset":  ["deposit", "supply", "provide", "add liquidity"],
    "withdraw_asset": ["withdraw", "redeem", "unstake", "remove liquidity"],
    "swap_asset":     ["swap", "convert", "trade", "exchange"],
    "borrow_asset":   ["borrow", "draw"],
    "repay_asset":    ["repay", "pay back"],
    "stake_asset":    ["stake", "lock", "bond"],
    "unstake_asset":  ["unstake", "unlock", "unbond"],
    "claim_rewards":  ["claim", "harvest", "collect rewards"],
}
PRIMS = list(TERM_BANK.keys())

OPPOSITE = {
    "deposit_asset": "withdraw_asset",
    "withdraw_asset": "deposit_asset",
    "stake_asset": "unstake_asset",
    "unstake_asset": "stake_asset",
    "borrow_asset": "repay_asset",
    "repay_asset": "borrow_asset",
    # claim_rewards has no strict opposite here
}


# ----- SBERT embedding wrapper ----------------------------------------------
class Emb:
    def __init__(self, model_name: str):
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers not available. Please `pip install sentence-transformers`."
            )
        self._m = SentenceTransformer(model_name)

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size: int = 32):
        return self._m.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
        )


# ----- I/O helpers (align with mapper_bench.py) ------------------------------
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
        for r in R:
            out.append(r["label"])
    return out


# ----- Text / span utilities -------------------------------------------------
def _norm_tokens(s: str) -> List[str]:
    import re
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [t for t in s.split() if t]
    return toks


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_prototypes(term_bank: Dict[str, list[str]], emb: Emb) -> Dict[str, np.ndarray]:
    protos = {}
    for prim, terms in term_bank.items():
        V = np.asarray(emb.encode(terms, normalize_embeddings=True))
        protos[prim] = V.mean(axis=0)
    return protos


def spans_from_prompt(prompt: str,
                      prototypes: Dict[str, np.ndarray],
                      emb: Emb,
                      tau_span: float = 0.55,
                      n_max: int = 5,
                      topk_per_prim: int = 3):
    """
    Returns: dict primitive -> [ {primitive, term, score, t_center, start, len}, ... ]
    """
    toks = _norm_tokens(prompt)
    if not toks:
        return {}

    grams, meta = [], []   # meta holds (start, n, t_center)
    for n in range(1, min(n_max, len(toks)) + 1):
        for i in range(0, len(toks) - n + 1):
            s = " ".join(toks[i:i+n])
            t_center = (i + n/2.0) / max(1.0, len(toks))
            grams.append(s)
            meta.append((i, n, t_center))

    V = np.asarray(emb.encode(grams, normalize_embeddings=True))  # [M, D]

    by_prim: Dict[str, list] = {k: [] for k in prototypes.keys()}
    for m, (i, n, t_center) in enumerate(meta):
        v = V[m]
        for prim, proto in prototypes.items():
            sc = max(0.0, _cosine(v, proto))
            if sc >= tau_span:
                by_prim[prim].append({
                    "primitive": prim,
                    "term": grams[m],
                    "score": sc,
                    "t_center": float(t_center),
                    "start": i,
                    "len": n,
                })

    # per-primitive topk
    for prim, arr in by_prim.items():
        arr.sort(key=lambda x: x["score"], reverse=True)
        by_prim[prim] = arr[:topk_per_prim]
    return by_prim


def audit_prompt_with_spans(prompt: str,
                            prototypes: Dict[str, np.ndarray],
                            emb: Emb,
                            tau_span: float = 0.55,
                            rel_margin: float = 0.06):
    """
    Produces:
      - best_primitive: lexical 'winner' (requires tau + small relative margin)
      - scores: top-score per primitive (0 if none)
      - spans: raw span map
      - rel_margin: best - second best score
    """
    span_map = spans_from_prompt(prompt, prototypes, emb, tau_span=tau_span)

    # best score per primitive
    scores = {k: (arr[0]["score"] if arr else 0.0) for k, arr in span_map.items()}
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    if not ordered:
        return {"best_primitive": None, "scores": scores, "spans": span_map, "rel_margin": 0.0}

    best_prim, best_sc = ordered[0]
    second_sc = ordered[1][1] if len(ordered) > 1 else 0.0
    winner = best_prim if (best_sc >= tau_span and (best_sc - second_sc) >= rel_margin) else None

    return {
        "best_primitive": winner,
        "scores": scores,
        "spans": span_map,
        "rel_margin": best_sc - second_sc,
        "params": {"tau_span": tau_span, "rel_margin": rel_margin},
    }


def load_mapper(path: str):
    return joblib.load(path)


def mapper_top1_label(mapper, prompt: str) -> Tuple[str | None, float]:
    # generic scikit-like pipeline
    import numpy as np
    if hasattr(mapper, "predict_proba"):
        probs = mapper.predict_proba([prompt])[0]
        classes = list(getattr(mapper, "classes_", PRIMS))
        top_idx = int(np.argmax(probs))
        return classes[top_idx], float(probs[top_idx])
    elif hasattr(mapper, "predict"):
        lab = mapper.predict([prompt])[0]
        return str(lab), 1.0
    return None, 0.0


def should_veto(mapper_top1: str | None, audit_best: str | None) -> bool:
    if not mapper_top1 or not audit_best:
        return False
    return OPPOSITE.get(mapper_top1) == audit_best


def fuse_decision(prompt: str,
                  mapper,
                  prototypes: Dict[str, np.ndarray],
                  emb: Emb,
                  conf_thr: float = 0.70,
                  tau_span: float = 0.55,
                  rel_margin: float = 0.06):
    """
    Decision policy:
      - If mapper confidence < conf_thr => reject (abstain)
      - Else compute audit_best via spans. If audit_best is None => reject
      - If audit_best is opposite of mapper_top1 => reject (hard veto)
      - If audit_best != mapper_top1 => reject (mismatch)
      - Else approve
    """
    m_top, m_conf = mapper_top1_label(mapper, prompt)
    fired = bool(m_conf >= conf_thr)

    audit = audit_prompt_with_spans(prompt, prototypes, emb, tau_span=tau_span, rel_margin=rel_margin)
    a_best = audit["best_primitive"]

    if not fired:
        return {
            "prompt": prompt,
            "decision": "reject",
            "reason": "low_confidence",
            "mapper": {"top": m_top, "conf": m_conf},
            "audit": audit,
        }
    if a_best is None:
        return {
            "prompt": prompt,
            "decision": "reject",
            "reason": "no_span_evidence",
            "mapper": {"top": m_top, "conf": m_conf},
            "audit": audit,
        }
    if should_veto(m_top, a_best):
        return {
            "prompt": prompt,
            "decision": "reject",
            "reason": f"veto_opposite:{m_top}_vs_{a_best}",
            "mapper": {"top": m_top, "conf": m_conf},
            "audit": audit,
        }
    if a_best != m_top:
        return {
            "prompt": prompt,
            "decision": "reject",
            "reason": f"audit_mismatch:{m_top}_vs_{a_best}",
            "mapper": {"top": m_top, "conf": m_conf},
            "audit": audit,
        }

    return {
        "prompt": prompt,
        "decision": "approve",
        "mapper": {"top": m_top, "conf": m_conf},
        "audit": audit,
    }


# ----- Bench core ------------------------------------------------------------
def write_rows_csv(fp: Path, prompts: List[str], gold: List[str] | None, outs: List[dict], conf_thr: float):
    with fp.open("w", newline="") as f:
        W = csv.writer(f)
        W.writerow(["prompt","gold_label","mapper_top","mapper_conf","audit_best","decision","reason","approved","threshold"])
        for i, p in enumerate(prompts):
            g = (gold[i] if gold is not None and i < len(gold) else "")
            o = outs[i]
            m_top = o["mapper"]["top"]
            m_conf = o["mapper"]["conf"]
            a_best = o["audit"]["best_primitive"]
            dec = o["decision"]
            rsn = o.get("reason","")
            W.writerow([p, g, m_top or "", f"{m_conf:.4f}", a_best or "", dec, rsn, str(dec=="approve"), f"{conf_thr:.4f}"])


def compute_metrics(gold: List[str] | None, outs: List[dict]) -> Dict[str, Any]:
    total = len(outs)
    approved = sum(1 for o in outs if o["decision"] == "approve")
    rejected = total - approved
    coverage = approved / total if total else 0.0
    abstain_rate = rejected / total if total else 0.0

    acc_on_approved = None
    overall_acc = None

    if gold is not None and len(gold) == total:
        correct_on_approved = 0
        for g, o in zip(gold, outs):
            if o["decision"] == "approve" and o["mapper"]["top"] == g:
                correct_on_approved += 1
        acc_on_approved = (correct_on_approved / approved) if approved else 0.0

        # Define overall accuracy as (# approved & correct) / total
        overall_acc = (correct_on_approved / total) if total else 0.0

    return {
        "total": total,
        "approved": approved,
        "rejected": rejected,
        "coverage": coverage,
        "abstain_rate": abstain_rate,
        "accuracy_on_approved": acc_on_approved if acc_on_approved is not None else -1.0,
        "overall_accuracy": overall_acc if overall_acc is not None else -1.0,
    }
    
def _parse_list(val):
    if isinstance(val, (float, int)):
        return [float(val)]
    s = str(val)
    if "," in s:
        return [float(x.strip()) for x in s.split(",") if x.strip()]
    return [float(s.strip())]

def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir or ".artifacts/defi/audit_bench")
    out_dir.mkdir(parents=True, exist_ok=True)

    conf_list = _parse_list(args.conf_thr)
    tau_list  = _parse_list(args.tau_span)
    rel_list  = _parse_list(args.rel_margin)

    # Load data
    prompts = read_prompts_jsonl(Path(args.prompts_jsonl))
    gold = read_labels_csv(Path(args.labels_csv)) if args.labels_csv else None
    if gold is not None:
        assert len(prompts) == len(gold), "prompts and labels must align"

    # Load mapper (trained via mapper_bench.py / train_mapper_embed.py)
    mapper = load_mapper(args.model_path)

    # Prepare SBERT + prototypes
    emb = Emb(args.sbert)
    prototypes = build_prototypes(TERM_BANK, emb)

    # Evaluate at one or multiple confidence thresholds (allow comma list for convenience)
    thrs = [float(x) for x in (args.conf_thr.split(",") if isinstance(args.conf_thr, str) and "," in args.conf_thr else [args.conf_thr])]

    metrics_rows = []
    chosen = None
    best_util = -1e9

    for thr in thrs:
        outs = [
            fuse_decision(p, mapper, prototypes, emb,
                          conf_thr=thr, tau_span=args.tau_span, rel_margin=args.rel_margin)
            for p in prompts
        ]

        # Per-threshold rows
        write_rows_csv(out_dir / f"rows_thr_{thr:.2f}.csv", prompts, gold, outs, thr)

        M = compute_metrics(gold, outs)
        M.update({"conf_thr": thr, "tau_span": args.tau_span, "rel_margin": args.rel_margin})
        metrics_rows.append(M)

        util = (M["overall_accuracy"] if M["overall_accuracy"] >= 0 else 0.0) * 100.0 \
               - (M["abstain_rate"] * 10.0)
        if util > best_util:
            best_util = util
            chosen = {
                "conf_thr": thr,
                "tau_span": args.tau_span,
                "rel_margin": args.rel_margin,
                "coverage": M["coverage"],
                "abstain_rate": M["abstain_rate"],
                "accuracy_on_approved": M["accuracy_on_approved"],
                "overall_accuracy": M["overall_accuracy"],
            }

    # metrics.csv
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        W = csv.DictWriter(f, fieldnames=[
            "conf_thr","tau_span","rel_margin","total","approved","rejected",
            "coverage","abstain_rate","accuracy_on_approved","overall_accuracy"
        ])
        W.writeheader()
        for r in metrics_rows:
            W.writerow(r)

    # summary.json
    summary = {
        "bench": "audit_bench",
        "sbert": args.sbert,
        "thresholds": {"conf_thr": thrs, "tau_span": args.tau_span, "rel_margin": args.rel_margin},
        "chosen": chosen,
        "timestamp": int(time.time())
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # report.md
    lines = [
        "# Audit Bench — Report",
        "",
        f"SBERT: {args.sbert}",
        "",
    ]
    for r in metrics_rows:
        lines.append(
            f"- conf_thr={r['conf_thr']:.2f} tau_span={r['tau_span']:.2f} rel_margin={r['rel_margin']:.2f} "
            f"overall_acc={r['overall_accuracy']:.4f} abstain_rate={r['abstain_rate']:.4f} coverage={r['coverage']:.4f}"
        )
    (out_dir / "report.md").write_text("\n".join(lines))

    # Optional gate
    if args.min_overall_acc is not None and chosen is not None:
        if chosen["overall_accuracy"] is not None and chosen["overall_accuracy"] < float(args.min_overall_acc):
            print(f"[audit_bench] FAIL gate: overall_acc={chosen['overall_accuracy']:.3f} < {args.min_overall_acc}", file=sys.stderr)
            return 2

    print(f"[audit_bench] PASS. chosen={json.dumps(chosen)}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=".artifacts/defi_mapper.joblib")
    ap.add_argument("--prompts_jsonl", default="tests/fixtures/defi/mapper_smoke_prompts.jsonl")
    ap.add_argument("--labels_csv",    default="tests/fixtures/defi/mapper_smoke_labels.csv")
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="SentenceTransformer model name to use for audit spans")
    ap.add_argument("--conf_thr", type=float, default=0.70,
                    help="Mapper confidence threshold for approval (comma-separated allowed)")
    ap.add_argument("--tau_span", type=float, default=0.55,
                    help="Span/prototype similarity threshold")
    ap.add_argument("--rel_margin", type=float, default=0.06,
                    help="Required margin between best and second-best span scores")
    ap.add_argument("--out_dir", default=".artifacts/defi/audit_bench")
    ap.add_argument("--min_overall_acc", default=None)
    ap.add_argument("--conf_thr", default="0.70",
                    help="Mapper confidence threshold(s), comma-separated allowed")
    ap.add_argument("--tau_span", default="0.55",
                    help="Span/prototype similarity threshold(s), comma-separated allowed")
    ap.add_argument("--rel_margin", default="0.06",
                    help="Required margin(s), comma-separated allowed")
    args = ap.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
