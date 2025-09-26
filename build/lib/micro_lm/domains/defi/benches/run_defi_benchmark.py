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

# warp-detect
python3 src/micro_lm/domains/defi/benches/run_defi_benchmark.py \
    --samples 200 --seed 42 --T 720 --sigma 9 --proto_width 160 \
    --out_plot manifold_pca3_mesh_warped.png \
    --out_plot_fit manifold_pca3_mesh_warped_fit.png \
    --out_csv stage11_metrics.csv \
    --out_json stage11_summary.json

# warp-detect-denoise
python3 benchmarks/defi/run_defi_benchmark.py \
  --samples 200 --seed 42 \
  --denoise_mode hybrid --ema_decay 0.85 --median_k 3 \
  --probe_k 5 --probe_eps 0.02 --conf_gate 0.65 --noise_floor 0.03 \
  --seed_jitter 2 \
  --latent_arc --latent_arc_noise 0.05 \
  --out_csv latent_arc_denoise.csv \
  --out_json latent_arc_denoise.json

"""
from __future__ import annotations

import argparse, csv, json, os, warnings, math, logging as pylog
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

from ngeodesic.core.parser import geodesic_parse_report, stock_parse
# from ngeodesic.synth.arc_like import make_synthetic_traces_stage11
from ngeodesic.bench.metrics import set_metrics
from ngeodesic.core.matched_filter import half_sine_proto, nxcorr, null_threshold
from ngeodesic.core.denoise import TemporalDenoiser, snr_db
from ngeodesic.synth import gaussian_bump
from ngeodesic.core.denoise import make_denoiser
from ngeodesic.bench.io import write_rows_csv, write_json

from ngeodesic.stage11.runner import Runner 
from ngeodesic.stage11.hooks import ModelHooks
from ngeodesic.stage11.guard import phantom_guard

# ============================================================
# Stage-11 synthetic ARC-like generator (RNG-first, hard mode)
# ============================================================

PRIMS = [
  "deposit", "withdraw", "borrow", "repay",
  "swap", "add_liquidity", "remove_liquidity", "claim_rewards"
]

def make_synthetic_traces_stage11(rng, T=720, noise=0.02, cm_amp=0.02, overlap=0.5,
                          amp_jitter=0.4, distractor_prob=0.25,
                          tasks_k=(1,3)):
    k = int(rng.integers(tasks_k[0], tasks_k[1]+1))
    tasks = list(rng.choice(PRIMS, size=k, replace=False))
    rng.shuffle(tasks)
    base = np.linspace(0.15, 0.85, num=3) * T
    centers = ((1.0 - overlap) * base + overlap * (T * 0.50)).astype(int)
    width = int(max(12, T * 0.08))
    t = np.arange(T)

    # slow “market/chain” common-mode (e.g., gas or volatility drift)
    cm = cm_amp * (1.0 + 0.2 * np.sin(2*np.pi * t / max(30, T//6)))

    traces = {p: np.zeros(T, float) for p in PRIMS}

    # generate true DeFi actions as Gaussian bumps
    for i, prim in enumerate(tasks):
        c = centers[i % len(centers)]
        amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
        c_jit = int(np.clip(c + rng.integers(-width//5, width//5 + 1), 0, T-1))
        traces[prim] += gaussian_bump(T, c_jit, width, amp=amp)

        # optional coupling (e.g., borrow→swap or add_liquidity→claim_rewards)
        if prim in ("borrow","add_liquidity") and rng.random() < 0.5:
            buddy = "swap" if prim == "borrow" else "claim_rewards"
            amp2 = 0.6 * amp
            c2 = int(np.clip(c_jit + rng.integers(width//6, width//3), 0, T-1))
            traces[buddy] += gaussian_bump(T, c2, width, amp=amp2)

    # distractors (spurious parser pressure) on non-selected primitives
    for p in PRIMS:
        if p not in tasks and rng.random() < distractor_prob:
            c = int(rng.uniform(T*0.15, T*0.85))
            amp = max(0.2, 0.8 + rng.normal(0, 0.25))
            traces[p] += gaussian_bump(T, c, width, amp=amp)

    # add common mode + noise and clip
    for p in PRIMS:
        traces[p] = np.clip(traces[p] + cm, 0, None)
        traces[p] = traces[p] + rng.normal(0, noise, size=T)
        traces[p] = np.clip(traces[p], 0, None)

    return traces, tasks

def prefix_exact(true_list: List[str], pred_list: List[str]) -> bool:
    return list(true_list) == list(pred_list)

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage-11 report benchmark (package-based, compat)")
    # data
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=int, default=720)
    p.add_argument("--sigma", type=int, default=9, help="smoother window for residual energy")
    p.add_argument("--proto_width", type=int, default=160)
    # generator knobs (Stage-11 hard mode)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--cm_amp", type=float, default=0.02)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--amp_jitter", type=float, default=0.4)
    p.add_argument("--distractor_prob", type=float, default=0.4)
    p.add_argument("--min_tasks", type=int, default=1)
    p.add_argument("--max_tasks", type=int, default=3)
    # outputs
    p.add_argument("--out_plot", type=str, default="manifold_pca3_mesh_warped.png")
    p.add_argument("--out_plot_fit", type=str, default="manifold_pca3_mesh_warped_fit.png")
    p.add_argument("--out_csv", type=str, default="stage11_metrics.csv")
    p.add_argument("--out_json", type=str, default="stage11_summary.json")
    # viz toggle
    p.add_argument("--render_well", action="store_true", help="Render PCA well/funnel proxies (if viz extras available)")
    # DENOISE & GUARDS
    p.add_argument("--denoise_mode", type=str, default="off", choices=["off","ema","median","hybrid"])
    p.add_argument("--ema_decay", type=float, default=0.85)
    p.add_argument("--median_k", type=int, default=3)
    p.add_argument("--probe_k", type=int, default=5)
    p.add_argument("--probe_eps", type=float, default=0.02)
    p.add_argument("--conf_gate", type=float, default=0.65)
    p.add_argument("--noise_floor", type=float, default=0.03)
    p.add_argument("--seed_jitter", type=int, default=2)
    p.add_argument("--log_snr", type=int, default=1)
    # Latent ARC
    p.add_argument("--latent_arc", action="store_true")
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--latent_arc_noise", type=float, default=0.05)
    # Logging
    p.add_argument("--log", type=str, default="INFO")
    return p

def main():
    args = build_argparser().parse_args()
    pylog.basicConfig(level=getattr(pylog, args.log.upper(), pylog.INFO),
                      format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rng = np.random.default_rng(args.seed)

    rows: List[Dict[str, object]] = []
    agg_geo = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)
    agg_stock = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)

    for i in range(1, args.samples + 1):
        traces, true_order = make_synthetic_traces_stage11(
            rng,
            T=args.T,
            noise=args.noise,
            cm_amp=args.cm_amp,
            overlap=args.overlap,
            amp_jitter=args.amp_jitter,
            distractor_prob=args.distractor_prob,
            tasks_k=(args.min_tasks, args.max_tasks),
        )

        keep_g, order_g = geodesic_parse_report(traces, sigma=args.sigma, proto_width=args.proto_width)
        keep_s, order_s = stock_parse(traces, sigma=args.sigma, proto_width=args.proto_width)

        acc_g = int(prefix_exact(true_order, order_g))
        acc_s = int(prefix_exact(true_order, order_s))

        sm_g = set_metrics(true_order, keep_g)
        sm_s = set_metrics(true_order, keep_s)

        for k, v in sm_g.items():
            key = {"precision":"P","recall":"R","f1":"F1","jaccard":"J","hallucination_rate":"H","omission_rate":"O"}[k]
            agg_geo[key] = agg_geo.get(key, 0) + v
        for k, v in sm_s.items():
            key = {"precision":"P","recall":"R","f1":"F1","jaccard":"J","hallucination_rate":"H","omission_rate":"O"}[k]
            agg_stock[key] = agg_stock.get(key, 0) + v

        agg_geo["acc"] += acc_g
        agg_stock["acc"] += acc_s

        rows.append(dict(
            sample=i,
            true="|".join(true_order),
            geodesic_tasks="|".join(keep_g), geodesic_order="|".join(order_g), geodesic_ok=acc_g,
            stock_tasks="|".join(keep_s), stock_order="|".join(order_s), stock_ok=acc_s,
            geodesic_precision=sm_g["precision"], geodesic_recall=sm_g["recall"], geodesic_f1=sm_g["f1"],
            geodesic_jaccard=sm_g["jaccard"], geodesic_hallucination=sm_g["hallucination_rate"], geodesic_omission=sm_g["omission_rate"],
            stock_precision=sm_s["precision"], stock_recall=sm_s["recall"], stock_f1=sm_s["f1"],
            stock_jaccard=sm_s["jaccard"], stock_hallucination=sm_s["hallucination_rate"], stock_omission=sm_s["omission_rate"],
        ))

    n = float(args.samples)
    Sg = dict(
        accuracy_exact=agg_geo["acc"]/n, precision=agg_geo["P"]/n, recall=agg_geo["R"]/n, f1=agg_geo["F1"]/n,
        jaccard=agg_geo["J"]/n, hallucination_rate=agg_geo["H"]/n, omission_rate=agg_geo["O"]/n
    )
    Ss = dict(
        accuracy_exact=agg_stock["acc"]/n, precision=agg_stock["P"]/n, recall=agg_stock["R"]/n, f1=agg_stock["F1"]/n,
        jaccard=agg_stock["J"]/n, hallucination_rate=agg_stock["H"]/n, omission_rate=agg_stock["O"]/n
    )

    # Optional rendering (lazy import; safe failure)
    if args.render_well:
        try:
            from ngeodesic.viz import collect_HE, render_pca_well  # NOTE: may fail if viz extras not installed
            H, E = collect_HE(
                samples=min(max(int(n), 100), 2000),
                rng=np.random.default_rng(args.seed + 777),
                T=args.T, sigma=args.sigma,
                noise=args.noise, cm_amp=args.cm_amp, overlap=args.overlap,
                amp_jitter=args.amp_jitter, distractor_prob=args.distractor_prob,
                tasks_k=(args.min_tasks, args.max_tasks),
            )
            render_pca_well(args.out_plot, args.out_plot_fit, H, E)
        except Exception as e:
            warnings.warn(f"Render disabled (viz extras not available): {e}")

    if args.out_csv:
        write_rows_csv(args.out_csv, rows)

    summary = dict(
        samples=int(n), geodesic=Sg, stock=Ss,
        plot_raw=args.out_plot, plot_fitted=args.out_plot_fit, csv=args.out_csv
    )
    if args.out_json:
        write_json(args.out_json, summary)

    print("[SUMMARY] Geodesic:", {k: round(v, 3) for k, v in Sg.items()})
    print("[SUMMARY] Stock   :", {k: round(v, 3) for k, v in Ss.items()})
    print(f"[PLOT] RAW:     {args.out_plot}")
    print(f"[PLOT] FITTED:  {args.out_plot_fit}")
    print(f"[CSV ] {args.out_csv}")
    print(f"[JSON] {args.out_json}")

    # -------------------
    # Denoiser path (optional)
    # -------------------
    if args.denoise_mode != "off" and args.latent_arc:
        hooks = ModelHooks()
        runner = Runner(args, hooks, phantom_guard)  # pass the guard function in
        denoise_metrics = runner.run()

        
        if args.out_json:
            try:
                with open(args.out_json, "r") as f:
                    S = json.load(f)
            except Exception:
                S = {}
            S["denoise"] = denoise_metrics
            write_json(args.out_json, S)
        print("[DENOISE] latent-ARC metrics:", {k: (round(v,3) if isinstance(v,float) else v) for k,v in denoise_metrics.items()})


if __name__ == "__main__":
    main()