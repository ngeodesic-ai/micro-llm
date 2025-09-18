#!/usr/bin/env python3
"""
Audit Bench — Rails-as-Auditor (Tier‑1 fix) with sweeps, failure dumps, and span override.

- Uses independent SBERT span evidence vs per-primitive prototypes.
- Supports grid sweeps for conf/tau/margin.
- Can dump failures and a confusion matrix.
- New: --n_max for span length, and optional strong-span override for low-confidence mapper hits.

Outputs under --out_dir:
  rows_thr_<conf>_tau_<tau>_rel_<rel>.csv
  metrics.csv
  summary.json
  report.md
  [optional] failures_thr_...jsonl, confusion_approved_...csv
"""
from __future__ import annotations
import argparse, json, csv, time, sys, re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import joblib

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


# ----- Domain term bank & opposites -----------------------------------------
TERM_BANK: Dict[str, list[str]] = {
    "deposit_asset": [
        "deposit","supply","provide","add liquidity","provide liquidity","top up","put in","add funds",
        "fund","allocate","contribute","add position","add to pool","supply to pool","add into","supply into"
    ],
    "withdraw_asset": [
        "withdraw","redeem","unstake","remove liquidity","pull out","take out","cash out","exit",
        "remove position","remove from pool","take from","pull from"
    ],
    "swap_asset": [
        "swap","convert","trade","exchange","convert into","swap to","swap into",
        "bridge","wrap","unwrap","swap for"
    ],
    "borrow_asset": [
        "borrow","draw","open a loan for","open debt","draw down","take a loan","borrow against"
    ],
    "repay_asset": [
        "repay","pay back","close out the loan for","settle loan","pay debt","repay debt","close loan"
    ],
    "stake_asset": [
        "stake","lock","bond","delegate","lock up","stake into","stake to","stake on"
    ],
    "unstake_asset": [
        "unstake","unlock","unbond","undelegate","release","unstake from","unstake out"
    ],
    "claim_rewards": [
        "claim","harvest","collect rewards","claim rewards","collect staking rewards",
        "collect yield","claim yield","harvest rewards"
    ],
}
PRIMS = list(TERM_BANK.keys())

OPPOSITE = {
    "deposit_asset": "withdraw_asset",
    "withdraw_asset": "deposit_asset",
    "stake_asset": "unstake_asset",
    "unstake_asset": "stake_asset",
    "borrow_asset": "repay_asset",
    "repay_asset": "borrow_asset",
}


# ----- Embedding wrapper -----------------------------------------------------
class Emb:
    def __init__(self, model_name: str):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not available. `pip install sentence-transformers`.")
        self._m = SentenceTransformer(model_name)

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size: int = 32):
        return self._m.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
        )


# ----- I/O helpers -----------------------------------------------------------
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


# ----- Text / span utilities -------------------------------------------------
def _norm_tokens(s: str) -> List[str]:
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
    toks = _norm_tokens(prompt)
    if not toks:
        return {}

    grams, meta = [], []
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

    for prim, arr in by_prim.items():
        arr.sort(key=lambda x: x["score"], reverse=True)
        by_prim[prim] = arr[:topk_per_prim]
    return by_prim


def audit_prompt_with_spans(prompt: str,
                            prototypes: Dict[str, np.ndarray],
                            emb: Emb,
                            tau_span: float = 0.55,
                            rel_margin: float = 0.06,
                            n_max: int = 5):
    span_map = spans_from_prompt(prompt, prototypes, emb, tau_span=tau_span, n_max=n_max)

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
        "params": {"tau_span": tau_span, "rel_margin": rel_margin, "n_max": n_max},
    }


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
                  rel_margin: float = 0.06,
                  n_max: int = 5,
                  allow_span_override: bool = False,
                  override_tau: float = 0.62,
                  override_margin: float = 0.10):
    m_top, m_conf = mapper_top1_label(mapper, prompt)

    audit = audit_prompt_with_spans(prompt, prototypes, emb,
                                    tau_span=tau_span, rel_margin=rel_margin, n_max=n_max)
    a_best = audit["best_primitive"]

    # Optional span override for low confidence
    if m_conf < conf_thr and allow_span_override:
        a_scores = audit["scores"]
        ordered = sorted(a_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_prim, best_sc = ordered[0] if ordered else (None, 0.0)
        second_sc = ordered[1][1] if len(ordered) > 1 else 0.0
        strong = (best_sc >= override_tau) and ((best_sc - second_sc) >= override_margin)
        if strong and audit["best_primitive"] == m_top:
            return {
                "prompt": prompt,
                "decision": "approve",
                "mapper": {"top": m_top, "conf": m_conf},
                "audit": audit,
                "note": "strong_span_override"
            }

    # Decision policy
    if m_conf < conf_thr:
        return {"prompt": prompt, "decision": "reject", "reason": "low_confidence",
                "mapper": {"top": m_top, "conf": m_conf}, "audit": audit}
    if a_best is None:
        return {"prompt": prompt, "decision": "reject", "reason": "no_span_evidence",
                "mapper": {"top": m_top, "conf": m_conf}, "audit": audit}
    if should_veto(m_top, a_best):
        return {"prompt": prompt, "decision": "reject", "reason": f"veto_opposite:{m_top}_vs_{a_best}",
                "mapper": {"top": m_top, "conf": m_conf}, "audit": audit}
    if a_best != m_top:
        return {"prompt": prompt, "decision": "reject", "reason": f"audit_mismatch:{m_top}_vs_{a_best}",
                "mapper": {"top": m_top, "conf": m_conf}, "audit": audit}
    return {"prompt": prompt, "decision": "approve",
            "mapper": {"top": m_top, "conf": m_conf}, "audit": audit}


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


def _parse_list(val):
    if isinstance(val, (float, int)):
        return [float(val)]
    s = str(val)
    if "," in s:
        return [float(x.strip()) for x in s.split(",") if x.strip()]
    return [float(s.strip())]


def _collect_failures(prompts: List[str], gold: List[str] | None, outs: List[dict]) -> List[dict]:
    fails = []
    N = len(outs)
    for i in range(N):
        o = outs[i]
        g = (gold[i] if gold is not None and i < len(gold) else None)
        m_top = o["mapper"]["top"]
        dec = o["decision"]
        reason = o.get("reason","")
        a_best = o["audit"]["best_primitive"]
        fail_type = None
        if g is not None:
            if dec == "approve" and m_top != g:
                fail_type = "wrong_approval"
            elif dec == "reject" and m_top == g:
                if reason.startswith("veto_opposite"):
                    fail_type = "false_reject_opposite_veto"
                elif reason.startswith("audit_mismatch"):
                    fail_type = "false_reject_audit_mismatch"
                elif reason == "no_span_evidence":
                    fail_type = "false_reject_no_span"
                elif reason == "low_confidence":
                    fail_type = "false_reject_low_conf"
                else:
                    fail_type = "false_reject_other"
            elif g != m_top and dec == "reject":
                fail_type = "correct_reject_mapper_wrong"
        else:
            if dec == "reject":
                fail_type = "reject_no_gold"
        row = {
            "idx": i,
            "prompt": prompts[i],
            "gold": g,
            "mapper_top": m_top,
            "mapper_conf": o["mapper"]["conf"],
            "audit_best": a_best,
            "decision": dec,
            "reason": reason,
            "failure_type": fail_type,
        }
        sc = o["audit"]["scores"]
        top2 = sorted(sc.items(), key=lambda kv: kv[1], reverse=True)[:2]
        row["audit_scores_top2"] = top2
        row["audit_params"] = o["audit"].get("params", {})
        if "spans" in o["audit"]:
            row["spans"] = o["audit"]["spans"]
        fails.append(row)
    return fails


def _write_failures_jsonl(fp: Path, failures: List[dict], only_problematic: bool = True):
    with fp.open("w") as f:
        for r in failures:
            if (not only_problematic) or (r.get("failure_type") in (
                "wrong_approval",
                "false_reject_opposite_veto",
                "false_reject_audit_mismatch",
                "false_reject_no_span",
                "false_reject_low_conf",
                "false_reject_other",
            )):
                f.write(json.dumps(r) + "\n")


def _write_confusion_csv(fp: Path, gold: List[str] | None, outs: List[dict]):
    if gold is None or len(gold) != len(outs):
        return
    labels = sorted(set(gold) | set([o["mapper"]["top"] for o in outs if o["mapper"]["top"]]))
    import csv as _csv
    mat = {g:{p:0 for p in labels} for g in labels}
    for g,o in zip(gold, outs):
        if o["decision"] == "approve":
            p = o["mapper"]["top"]
            if g in mat and p in mat[g]:
                mat[g][p] += 1
    with fp.open("w", newline="") as f:
        W = _csv.writer(f)
        W.writerow(["gold\\pred"] + labels)
        for g in labels:
            W.writerow([g] + [mat[g][p] for p in labels])


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir or ".artifacts/defi/audit_bench")
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts_jsonl(Path(args.prompts_jsonl))
    gold = read_labels_csv(Path(args.labels_csv)) if args.labels_csv else None
    if gold is not None:
        assert len(prompts) == len(gold), "prompts and labels must align"

    mapper = load_mapper(args.model_path)

    emb = Emb(args.sbert)
    prototypes = build_prototypes(TERM_BANK, emb)

    conf_list = _parse_list(args.conf_thr)
    tau_list  = _parse_list(args.tau_span)
    rel_list  = _parse_list(args.rel_margin)

    metrics_rows = []
    chosen = None
    best_util = -1e9

    for thr in conf_list:
        for tau in tau_list:
            for rel in rel_list:
                outs = [
                    fuse_decision(p, mapper, prototypes, emb,
                                  conf_thr=thr, tau_span=tau, rel_margin=rel,
                                  n_max=args.n_max,
                                  allow_span_override=args.allow_span_override,
                                  override_tau=args.override_tau,
                                  override_margin=args.override_margin)
                    for p in prompts
                ]
                write_rows_csv(out_dir / f"rows_thr_{thr:.2f}_tau_{tau:.2f}_rel_{rel:.2f}.csv",
                               prompts, gold, outs, thr)

                M = compute_metrics(gold, outs)
                M.update({"conf_thr": thr, "tau_span": tau, "rel_margin": rel})
                metrics_rows.append(M)

                util = (M["overall_accuracy"] if M["overall_accuracy"] >= 0 else 0.0) * 100.0 \
                       - (M["abstain_rate"] * 10.0)
                if util > best_util:
                    best_util = util
                    chosen = {
                        "conf_thr": thr,
                        "tau_span": tau,
                        "rel_margin": rel,
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
        "thresholds": {
            "conf_thr": conf_list,
            "tau_span": tau_list,
            "rel_margin": rel_list
        },
        "chosen": chosen,
        "timestamp": int(time.time())
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # report.md
    lines = [
        "# Audit Bench — Report",
        "",
        f"SBERT: {args.sbert}",
        f"grid: conf={conf_list} tau={tau_list} rel={rel_list}",
        f"n_max: {args.n_max}, allow_span_override: {args.allow_span_override}, override_tau: {args.override_tau}, override_margin: {args.override_margin}",
        "",
    ]
    for r in metrics_rows:
        lines.append(
            f"- conf={r['conf_thr']:.2f} tau={r['tau_span']:.2f} rel={r['rel_margin']:.2f} "
            f"overall_acc={r['overall_accuracy']:.4f} abstain={r['abstain_rate']:.4f} coverage={r['coverage']:.4f}"
        )
    (out_dir / "report.md").write_text("\n".join(lines))

    # Optional outputs
    if args.dump_failures:
        fails = _collect_failures(prompts, gold, [
            fuse_decision(p, mapper, prototypes, emb,
                          conf_thr=chosen['conf_thr'], tau_span=chosen['tau_span'], rel_margin=chosen['rel_margin'],
                          n_max=args.n_max, allow_span_override=args.allow_span_override,
                          override_tau=args.override_tau, override_margin=args.override_margin)
            for p in prompts
        ])
        _write_failures_jsonl(out_dir / f"failures_thr_{chosen['conf_thr']:.2f}_tau_{chosen['tau_span']:.2f}_rel_{chosen['rel_margin']:.2f}.jsonl",
                              fails, only_problematic=True)
    if args.dump_confusion:
        outs_best = [
            fuse_decision(p, mapper, prototypes, emb,
                          conf_thr=chosen['conf_thr'], tau_span=chosen['tau_span'], rel_margin=chosen['rel_margin'],
                          n_max=args.n_max, allow_span_override=args.allow_span_override,
                          override_tau=args.override_tau, override_margin=args.override_margin)
            for p in prompts
        ]
        _write_confusion_csv(out_dir / f"confusion_approved_thr_{chosen['conf_thr']:.2f}_tau_{chosen['tau_span']:.2f}_rel_{chosen['rel_margin']:.2f}.csv",
                             gold, outs_best)

    # Gate
    if args.min_overall_acc is not None and chosen is not None:
        thr = float(args.min_overall_acc)
        oa = chosen["overall_accuracy"] if chosen["overall_accuracy"] is not None else -1.0
        if oa < thr:
            print(f"[audit_bench] FAIL GATE overall_acc={oa:.3f} < {thr:.3f} | "
                  f"conf={chosen['conf_thr']:.2f} tau={chosen['tau_span']:.2f} rel={chosen['rel_margin']:.2f}",
                  file=sys.stderr)
            return 2
        else:
            print(f"[audit_bench] PASS GATE overall_acc={oa:.3f} ≥ {thr:.3f} | "
                  f"conf={chosen['conf_thr']:.2f} tau={chosen['tau_span']:.2f} rel={chosen['rel_margin']:.2f}")
            return 0

    print(f"[audit_bench] PASS. chosen={json.dumps(chosen)}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=".artifacts/defi_mapper.joblib")
    ap.add_argument("--prompts_jsonl", default="tests/fixtures/defi/mapper_smoke_prompts.jsonl")
    ap.add_argument("--labels_csv",    default="tests/fixtures/defi/mapper_smoke_labels.csv")
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="SentenceTransformer model name to use for audit spans")
    # String defaults so comma lists are accepted
    ap.add_argument("--conf_thr", default="0.70",
                    help="Mapper confidence threshold(s), comma-separated allowed")
    ap.add_argument("--tau_span", default="0.55",
                    help="Span/prototype similarity threshold(s), comma-separated allowed")
    ap.add_argument("--rel_margin", default="0.06",
                    help="Required margin(s) between best and second-best span scores, comma-separated allowed")
    ap.add_argument("--n_max", type=int, default=5, help="max n-gram length for span scanning (try 6 or 7 for longer phrasing)")
    ap.add_argument("--allow_span_override", action="store_true",
                    help="allow strong span evidence to approve even if mapper conf < conf_thr")
    ap.add_argument("--override_tau", type=float, default=0.62, help="span score threshold for override")
    ap.add_argument("--override_margin", type=float, default=0.10, help="required margin for override")
    ap.add_argument("--out_dir", default=".artifacts/defi/audit_bench")
    ap.add_argument("--min_overall_acc", default=None)
    ap.add_argument("--dump_failures", action="store_true", help="Write failures_*.jsonl (problematic cases + spans)")
    ap.add_argument("--dump_confusion", action="store_true", help="Write confusion_approved_*.csv over approved decisions")
    args = ap.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
