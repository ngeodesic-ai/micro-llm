
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_bench.py — single-file experiment harness
------------------------------------------------
Run an "auditor-mode" rails trial over curated prompts.

Train:
  - (Optional) mapper prototypes (from train CSV)
  - Phrase harvesting (n-grams) -> per-primitive phrase lists
  - Per-primitive prototype vectors v_k (mean of phrase embeddings)
  - OOD threshold tau_ood from junk prompts

Test:
  - Mode A: auditor_only
  - Mode B: mapper_and_auditor (mapper only filters channels; never injects energy)
  - Text -> latents (independent of mapper) -> matched filter + parser
  - Report metrics (F1, hallucination rate, abstain on OOD)

CSV formats:
  train.csv: id,prompt,primitive
  test.csv : id,prompt,primitive
  junk.csv : id,prompt

Outputs:
  out_dir/metrics.json
  out_dir/report.md

python3 audit_bench.py \
  --train_csv train.csv \
  --test_csv test.csv \
  --junk_csv junk.csv \
  --out_dir results_auditor_only \
  --mode auditor_only

Replace the PlaceholderEmbedding with your encoder (e.g., mpnet).
Replace MapperHook.predict with your mapper (optional).
"""

import argparse, json, os, math, hashlib, random
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

# ---------- CONFIG DEFAULTS ----------
PRIMS_DEFAULT = ["deposit_asset","withdraw_asset","borrow_asset","repay_asset","swap_asset"]
T_DEFAULT = 720

# ---------- PLACEHOLDER EMBEDDING ----------
class PlaceholderEmbedding:
    """
    Deterministic sentence/phrase 'embedding' for prototyping.
    Replace with your real encoder:
       - def encode(self, texts: List[str]) -> np.ndarray [N, D]
       - def encode_one(self, text: str) -> np.ndarray [D]
    """
    def __init__(self, dim=256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.lower().encode("utf-8")).digest()
        # stretch bytes deterministically to dim
        arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        vec = np.tile(arr, int(math.ceil(self.dim/len(arr))))[:self.dim]
        vec = (vec - vec.mean()) / (vec.std() + 1e-6)
        return vec

    def encode_one(self, text: str) -> np.ndarray:
        return self._vec(text)

    def encode(self, texts):
        return np.stack([self._vec(t) for t in texts], axis=0)

def cosine(a, b):
    an = a / (np.linalg.norm(a) + 1e-8)
    bn = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(an, bn))

# ---------- MAPPER HOOK (OPTIONAL) ----------
class MapperHook:
    """
    Plug your mapper here. By default this uses nearest-prototype to simulate a mapper.
    It returns a list of (primitive, score) sorted by score desc.
    """
    def __init__(self, emb: PlaceholderEmbedding, proto: dict):
        self.emb = emb
        self.proto = proto  # {prim: vec}

    def predict(self, prompt: str):
        e = self.emb.encode_one(prompt)
        scores = [(k, max(0.0, cosine(e, v))) for k,v in self.proto.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

# ---------- PHRASE MINING ----------
def harvest_phrases(df_train, emb, prims, n_min=1, n_max=6, top_k=20, tau_span=0.4, seed_exemplars=3):
    """
    Very simple n-gram miner using cosine to seed exemplars per primitive.
    In practice, you'd curate and de-duplicate more carefully.
    """
    rng = random.Random(42)
    exemplars = {p: [] for p in prims}
    # seeds: take up to seed_exemplars prompts per primitive
    for p in prims:
        ex = df_train[df_train["primitive"]==p]["prompt"].tolist()
        rng.shuffle(ex)
        exemplars[p] = ex[:seed_exemplars] if ex else []

    proto_seed = {p: emb.encode(exemplars[p]).mean(axis=0) if exemplars[p] else emb.encode_one(p)
                  for p in prims}

    spans_per_prim = defaultdict(list)
    for _, row in df_train.iterrows():
        text = row["prompt"]
        gold = row["primitive"]
        toks = text.strip().split()
        for n in range(n_min, n_max+1):
            for i in range(0, len(toks)-n+1):
                span = " ".join(toks[i:i+n])
                e = emb.encode_one(span)
                for p in prims:
                    s = max(0.0, cosine(e, proto_seed[p]))
                    if s >= tau_span:
                        spans_per_prim[p].append((span, s))

    # keep top_k unique spans per prim
    phrases = {}
    for p, spans in spans_per_prim.items():
        spans.sort(key=lambda x: x[1], reverse=True)
        uniq = []
        seen = set()
        for s,score in spans:
            if s.lower() in seen: 
                continue
            uniq.append((s,score))
            seen.add(s.lower())
            if len(uniq) >= top_k:
                break
        phrases[p] = uniq
    return phrases

def build_prototypes(emb, phrases):
    proto = {}
    for p, pairs in phrases.items():
        if not pairs:
            proto[p] = emb.encode_one(p)
        else:
            proto[p] = emb.encode([s for s,_ in pairs]).mean(axis=0)
    return proto

# ---------- AUDITOR ENCODER (TEXT -> LATENTS) ----------
def rng_for(prompt):
    seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed)

def q_template(width=160):
    t = np.linspace(0, 1, width)
    q = np.sin(np.pi * t)
    return q / (np.linalg.norm(q) + 1e-8)

def spans_from_prompt(prompt, proto, emb, tau_span=0.5):
    toks = prompt.strip().split()
    spans = []
    for n in range(1, min(6, len(toks))+1):
        for i in range(0, len(toks)-n+1):
            s = " ".join(toks[i:i+n])
            e = emb.encode_one(s)
            for k, v in proto.items():
                sc = max(0.0, cosine(e, v))
                if sc >= tau_span:
                    t_center = (i + n/2.0) / max(1.0, len(toks))  # normalized [0,1]
                    spans.append({"prim": k, "score": sc, "t_center": t_center})
    return spans

def text_to_latents(prompt, emb, prototypes, tau_ood=0.45, spans=None, alpha=0.7, beta=0.3, sigma=0.02, T=720):
    e = emb.encode_one(prompt)
    sims = {k: max(0.0, cosine(e, prototypes[k])) for k in prototypes}
    s_max = max(sims.values()) if sims else 0.0

    rng = rng_for(prompt)
    traces = {k: rng.normal(0.0, sigma, size=T).astype("float32") for k in prototypes}

    if s_max < tau_ood:
        return traces

    q = q_template(width=min(160, T//4))
    for k, s in sims.items():
        if s <= 0.0:
            continue
        span_max = 0.0
        t_center = int(0.35 * T)
        if spans:
            ks = [sp for sp in spans if sp["prim"] == k]
            if ks:
                best = max(ks, key=lambda sp: sp["score"])
                span_max = best["score"]
                t_center = int(best["t_center"] * (T - len(q)))

        A = (alpha * s + beta * span_max)
        if A <= 0.0:
            continue

        start = max(0, min(T - len(q), t_center))
        traces[k][start:start+len(q)] += (A * q)

    return traces

# ---------- MATCHED FILTER + PARSER ----------
def matched_filter_scores(traces, restrict_to=None):
    keys = list(traces.keys()) if restrict_to is None else [k for k in traces.keys() if k in restrict_to]
    q = q_template()
    qlen = len(q)
    scores = {}
    nulls  = {}
    for k in keys:
        x = traces[k]
        # simple sliding dot-product
        vals = np.correlate(x, q, mode="valid")
        peak = float(np.max(vals)) if len(vals)>0 else 0.0
        # null via circular shift (one shift for speed; increase for robustness)
        shift = len(x)//3
        x_null = np.roll(x, shift)
        vals_null = np.correlate(x_null, q, mode="valid")
        null_floor = float(np.percentile(vals_null, 95)) if len(vals_null)>0 else 0.0
        scores[k] = peak
        nulls[k]  = null_floor
    return scores, nulls

def rails_detect_and_parse(traces, restrict_to=None, tau_rel=0.60, tau_abs_q=0.93):
    """
    tau_rel: relative gate as fraction of max channel (winner must be >= tau_rel * max)
    tau_abs_q: absolute gate: peak must exceed null + tau_abs_q * |null|
    """
    scores, nulls = matched_filter_scores(traces, restrict_to)
    if not scores:
        return {"sequence": [], "scores": scores, "nulls": nulls, "peaks": {}}

    # relative gate
    max_peak = max(scores.values()) if scores else 0.0
    keep = {k: v for k, v in scores.items() if v >= tau_rel * max_peak}

    # absolute gate vs null
    accept = []
    for k, v in keep.items():
        thr = nulls[k] + tau_abs_q * abs(nulls[k])
        if v > thr:
            accept.append((k, v))

    if not accept:
        return {"sequence": [], "scores": scores, "nulls": nulls, "peaks": {}}

    # fake temporal order: infer from span t_center if present in traces? (not stored here)
    # For this skeleton, order by score descending as a proxy.
    accept.sort(key=lambda x: x[1], reverse=True)
    sequence = [k for k,_ in accept]
    peaks = {k: {"score": s} for k,s in accept}
    return {"sequence": sequence, "scores": scores, "nulls": nulls, "peaks": peaks}

# ---------- METRICS ----------
def eval_sets(df, df_junk, emb, phrases, mode, tau_map, tau_span, tau_ood, alpha, beta, T, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Build prototypes from phrases
    prototypes = build_prototypes(emb, phrases)
    mapper = MapperHook(emb, prototypes)

    records = []
    for _, row in df.iterrows():
        pid, prompt, gold = row["id"], row["prompt"], row["primitive"]
        # mapper candidates
        candidates = set()
        if mode == "mapper_and_auditor":
            preds = mapper.predict(prompt)
            candidates = {k for k,p in preds if p >= tau_map}

        spans = spans_from_prompt(prompt, prototypes, emb, tau_span=tau_span)
        traces = text_to_latents(prompt, emb, prototypes, tau_ood=tau_ood, spans=spans, alpha=alpha, beta=beta, T=T)

        det = rails_detect_and_parse(traces, restrict_to=candidates if candidates else None)
        pred_seq = det["sequence"]
        pred = pred_seq[0] if pred_seq else ""

        ok = (pred == gold)
        records.append({"set":"test","id":pid,"prompt":prompt,"gold":gold,"pred":pred,"ok":ok})

    # junk set — should abstain
    for _, row in df_junk.iterrows():
        pid, prompt = row["id"], row["prompt"]
        candidates = set()
        if mode == "mapper_and_auditor":
            preds = mapper.predict(prompt)
            candidates = {k for k,p in preds if p >= tau_map}

        spans = spans_from_prompt(prompt, prototypes, emb, tau_span=tau_span)
        traces = text_to_latents(prompt, emb, prototypes, tau_ood=tau_ood, spans=spans, alpha=alpha, beta=beta, T=T)
        det = rails_detect_and_parse(traces, restrict_to=candidates if candidates else None)
        pred_seq = det["sequence"]
        pred = pred_seq[0] if pred_seq else ""
        records.append({"set":"junk","id":pid,"prompt":prompt,"gold":"","pred":pred,"ok": (pred == "")})

    df_out = pd.DataFrame.from_records(records)
    df_out.to_csv(os.path.join(out_dir, "preds.csv"), index=False)

    # metrics
    df_test = df_out[df_out["set"]=="test"]
    df_jk   = df_out[df_out["set"]=="junk"]

    tp = sum((df_test["gold"] == df_test["pred"]) & (df_test["gold"] != ""))
    fp = sum((df_test["gold"] != df_test["pred"]) & (df_test["pred"] != ""))
    fn = sum((df_test["gold"] != "") & (df_test["pred"] == ""))
    prec = tp / max(1, tp+fp)
    rec  = tp / max(1, tp+fn)
    f1   = 2*prec*rec / max(1e-8, (prec+rec))

    halluc_rate = sum(df_jk["pred"] != "") / max(1, len(df_jk))
    abstain_rate_junk = sum(df_jk["pred"] == "") / max(1, len(df_jk))

    metrics = {
        "precision": round(prec,4),
        "recall": round(rec,4),
        "f1": round(f1,4),
        "hallucination_rate_on_junk": round(halluc_rate,4),
        "abstain_rate_on_junk": round(abstain_rate_junk,4),
        "n_test": int(len(df_test)),
        "n_junk": int(len(df_jk))
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # brief report
    lines = []
    lines.append("# Auditor Bench Report")
    lines.append("")
    lines.append("## Metrics")
    lines.append("```json")
    lines.append(json.dumps(metrics, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Errors (first 10)")
    err = df_test[(df_test["gold"]!="") & (df_test["gold"]!=df_test["pred"])].head(10)
    for _,r in err.iterrows():
        lines.append(f"- id={r['id']} gold={r['gold']} pred={r['pred']} :: {r['prompt']}")
    lines.append("")
    lines.append("## Junk that should abstain (first 10 that failed)")
    bad = df_jk[df_jk["pred"]!=""].head(10)
    for _,r in bad.iterrows():
        lines.append(f"- id={r['id']} pred={r['pred']} :: {r['prompt']}")

    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--junk_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["auditor_only","mapper_and_auditor"], default="auditor_only")
    ap.add_argument("--tau_map", type=float, default=0.7)
    ap.add_argument("--tau_span", type=float, default=0.55)
    ap.add_argument("--tau_ood", type=float, default=0.45)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--lobe_width", type=int, default=160)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--top_k_phrases", type=int, default=20)
    ap.add_argument("--prims", type=str, default=",".join(PRIMS_DEFAULT))
    args = ap.parse_args()

    prims = [p.strip() for p in args.prims.split(",") if p.strip()]

    df_train = pd.read_csv(args.train_csv)
    df_test  = pd.read_csv(args.test_csv)
    df_junk  = pd.read_csv(args.junk_csv)

    emb = PlaceholderEmbedding(dim=256)

    phrases = harvest_phrases(df_train, emb, prims,
                              n_min=1, n_max=6,
                              top_k=args.top_k_phrases,
                              tau_span=args.tau_span)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "phrases.json"), "w") as f:
        json.dump({k: v for k,v in phrases.items()}, f, indent=2)

    eval_sets(df_test, df_junk, emb, phrases,
              mode=args.mode,
              tau_map=args.tau_map,
              tau_span=args.tau_span,
              tau_ood=args.tau_ood,
              alpha=args.alpha,
              beta=args.beta,
              T=args.T,
              out_dir=args.out_dir)

if __name__ == "__main__":
    main()
