# src/micro_lm/verify/wdd_audit.py
from __future__ import annotations
from typing import Dict, Any
import joblib
from .wdd_helpers import (
    make_trace_encoder, load_warp, traces_from_text,
    ngf_pass_prior, ngf_pass_report
)

def verify_wdd(plan: Dict[str,Any], state: Dict[str,Any], policy: Dict[str,Any]) -> Dict[str,Any]:
    seq = (plan or {}).get("sequence") or []
    if not seq:
        return {"ok": False, "reason":"abstain_non_exec", "tags":["wdd","no_plan"]}

    label = seq[0]
    prompt = (state or {}).get("prompt","")
    if not prompt:
        return {"ok": False, "reason":"no_prompt", "tags":["wdd","context"]}

    cfg = policy.get("wdd", {})
    enc = make_trace_encoder(cfg.get("base","sentence-transformers/all-MiniLM-L6-v2"),
                             cfg.get("layer_offset",-4))
    warp = load_warp(cfg.get("warp_path",".artifacts/wdd_token_warp.joblib"))
    traces = traces_from_text(enc, warp, prompt)

    # choose prior vs report
    pri_path = cfg.get("priors_path")
    if pri_path:
        priors = joblib.load(pri_path)
        ok, info = ngf_pass_prior(traces, priors,
                                  z=cfg.get("z",2.2),
                                  rel_floor=cfg.get("rel_floor",0.70),
                                  alpha=cfg.get("alpha",0.08),
                                  beta_s=cfg.get("beta_s",0.35),
                                  q_s=cfg.get("q_s",2.0))
    else:
        ok, info = ngf_pass_report(traces, z_abs=cfg.get("z_abs",2.4))

    # domain policy checks still apply downstream (HF/LTV/oracle etc.)
    return {"ok": bool(ok), "reason": "ok" if ok else "below_wdd_gates", "stats": info, "tags":["wdd"]}
