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

# src/micro_lm/core/runner.py
from dataclasses import dataclass
from typing import Dict
from pathlib import Path
import os, time, json

from .mapper_api import MapperAPI
from .rails_shim import Rails
from .bench_io import ArtifactWriter

# Small-interface shim pieces
from micro_lm.adapters.simple_context import SimpleContextAdapter
from micro_lm.mappers.joblib_mapper import JoblibMapper, JoblibMapperConfig
from micro_lm.planners.rule_planner import RulePlanner
from micro_lm.core.audit import get_audit_backend, wdd_audit, apply_pca_prior, load_pca_prior
from micro_lm.domains.defi.wdd_harness import detect as wdd_detect


def _write_profile(domain: str, profile: list, topline: dict):
    # ensure directory
    base = Path(".artifacts") / domain / "profile"
    base.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = base / ts
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "profile.json").write_text(json.dumps(profile, indent=2))
    (outdir / "topline.json").write_text(json.dumps(topline, indent=2))


def _intent_to_primitive(intent: str) -> str:
    t = (intent or "").strip().lower()
    table = {
        "swap": "swap_asset",
        "deposit": "deposit_asset",
        "withdraw": "withdraw_asset",
        "stake": "stake_asset",
        "unstake": "unstake_asset",
        "borrow": "borrow_asset",
        "repay": "repay_asset",
        "claim_rewards": "claim_rewards_asset",
    }
    return table.get(t, t)



def _latent_triplet_if_available(mapper, prompt: str, *, context: dict, rails: str):
    emb = None
    bank = None
    anchors = None
    # try nested impl APIs defensively
    impl = getattr(mapper, "impl", None)
    if impl is not None:
        encode = getattr(impl, "encode", None)
        if callable(encode):
            try:
                emb = encode(prompt, context=context, rails=rails)
            except Exception:
                emb = None
        get_bank = getattr(impl, "get_bank", None)
        if callable(get_bank):
            try:
                bank = get_bank(domain=getattr(mapper, "domain", ""))
            except Exception:
                bank = None
        get_anchors = getattr(impl, "get_anchors", None)
        if callable(get_anchors):
            try:
                anchors = get_anchors(domain=getattr(mapper, "domain", ""))
            except Exception:
                anchors = None
    return emb, bank, anchors


def _map_prompt_any(mapper, prompt: str):
    # Prefer modern API, fall back to older ones
    fn = (
        getattr(mapper, "map_prompt", None)
        or getattr(mapper, "map", None)
        or getattr(getattr(mapper, "impl", object()), "map", None)
    )
    if fn is None:
        raise AttributeError("No mapping function found on MapperAPI (expected map_prompt/map).")
    return fn(prompt)


def _audit_selector(domain: str, backend_name: str):
    """
    Dynamically load the requested audit backend and return a callable
    `audit(domain, policy)` → function that will be used during rails/verify.
    """
    mod = get_audit_backend(backend_name)
    if hasattr(mod, "audit"):
        return mod.audit  # expected shape
    # Fallbacks if your backend exposes a different entrypoint
    if hasattr(mod, "make"):
        return mod.make
    if hasattr(mod, "get_backend"):
        return mod.get_backend
    raise RuntimeError(f"Audit backend '{backend_name}' does not expose an audit/make/get_backend factory")



def _shim_map_and_plan(user_text: str, *, context: dict, policy: dict) -> dict:
    adapter = SimpleContextAdapter()
    model_path = policy.get("mapper", {}).get("model_path", ".artifacts/defi_mapper.joblib")

    if not os.path.exists(model_path):
        ctx = adapter.normalize(context)
        return {
            "label": "abstain",
            "score": 0.0,
            "reason": "shim:model_missing",
            "artifacts": {"shim": {"model_path": model_path, "ctx": ctx.raw}},
        }

    mapper = JoblibMapper(
        JoblibMapperConfig(
            model_path=model_path,
            confidence_threshold=policy.get("mapper", {}).get("confidence_threshold", 0.7),
        )
    )
    planner = RulePlanner()

    ctx = adapter.normalize(context)
    mres = mapper.infer(user_text, context=ctx.raw)

    if not getattr(mres, "intent", None):
        return {
            "label": "abstain",
            "score": float(getattr(mres, "score", 0.0) or 0.0),
            "reason": "low_confidence",
            "artifacts": {"mapper": mres.__dict__},
        }

    plan = planner.plan(intent=mres.intent, text=user_text, context=ctx.raw)
    artifacts = {"mapper": mres.__dict__, "plan": getattr(plan, "__dict__", {})}

    return {
        "label": mres.intent,
        "score": float(getattr(mres, "score", 1.0) or 1.0),
        "reason": "shim:mapper",
        "artifacts": artifacts,
    }


@dataclass(frozen=True)
class RunInputs:
    domain: str
    prompt: str
    context: dict
    policy: dict
    rails: str
    T: int
    backend: str = "sbert"  # Tier-1 default; Tier-0 "wordmap" available


def run_micro(
    domain: str,
    prompt: str,
    *,
    context: dict,
    policy: dict,
    rails: str,
    T: int,
    backend: str = "sbert",
) -> dict:
    """
    PUBLIC API (stable).
    Returns a dict with: ok, label, score, reason, artifacts.
    """
    # --- lightweight profiling scaffold (only if enabled) ---
    t0 = time.perf_counter()
    prof_enabled = bool((policy or {}).get("audit", {}).get("profile")) or os.getenv("MICRO_LM_PROFILE") == "1"
    prof = []        # profile.json must be a LIST
    profile_dir = None

    def _mark(phase: str, **extra):
        if prof_enabled:
            prof.append({"phase": phase, "t": time.perf_counter() - t0, **extra})

    # If profiling, create the timestamped dir up front so both files land together.
    if prof_enabled:
        base = Path(".artifacts") / domain / "profile"
        base.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        profile_dir = (base / ts)
        profile_dir.mkdir(parents=True, exist_ok=True)
        # Smoke test expects at least one entry with phase == "parse"
        _mark("parse", rails=rails, backend=backend, T=T)

    # 1) Map prompt -> (label, score, aux) via selected backend (support multiple surfaces)
    mapper = MapperAPI(backend=backend, domain=domain, policy=policy)
    map_fn = (
        getattr(mapper, "map_prompt", None)
        or getattr(mapper, "map", None)
        or getattr(getattr(mapper, "impl", object()), "map_prompt", None)
        or getattr(getattr(mapper, "impl", object()), "map", None)
    )
    if map_fn is None:
        raise AttributeError("No mapping function found on MapperAPI (expected map_prompt/map).")
    res = map_fn(prompt)
    
    if isinstance(res, tuple):
        label, score, aux = res[0], float(res[1]), (res[2] or {})
    elif isinstance(res, dict):
        label = res.get("label", "abstain")
        score = float(res.get("score", 0.0) or 0.0)
        aux = res.get("aux", {}) or {}
    else:
        label, score, aux = "abstain", 0.0, {}

    _mark("map", label=label, score=score)

    # 1b) Optional shim fallback (skip for Tier-0 "wordmap" to keep tests hermetic)
    use_shim_default = backend != "wordmap"
    use_shim = policy.get("mapper", {}).get("use_shim_fallback", use_shim_default)
    if label == "abstain" and use_shim and backend != "wordmap":
        shim_out = _shim_map_and_plan(prompt, context=context, policy=policy)
        label, score = shim_out["label"], float(shim_out.get("score", score) or score)
        aux = {
            "reason": shim_out.get("reason", aux.get("reason", "shim")),
            "artifacts": {**aux.get("artifacts", {}), **shim_out.get("artifacts", {})},
        }

    # 2) If abstain or rails disabled, finalize early (still honor profiling)
    if label == "abstain" or not rails:
        out = {
            "ok": label != "abstain",
            "label": label,
            "score": score,
            "reason": aux.get("reason", "abstain" if label == "abstain" else "mapped"),
            "artifacts": aux.get("artifacts", {}),
        }
        if prof_enabled and profile_dir is not None:
            # profile.json must be a LIST containing an entry with phase == "parse"
            (profile_dir / "profile.json").write_text(json.dumps(prof, indent=2))
            # topline.json must include latency_ms and n_keep
            topline = {
                "latency_ms": int(1000 * (time.perf_counter() - t0)),
                "n_keep": len(prof),
                "label": label,
                "score": score,
                "ok": out["ok"],
                "reason": out["reason"],
            }
            (profile_dir / "topline.json").write_text(json.dumps(topline, indent=2))
        return out

    # 3) Execute rails (shim executor for now)
    rails_exec = Rails(rails=rails, T=T)
    verify = rails_exec.verify(domain=domain, label=label, context=context, policy=policy)
    _mark("rails", ok=bool(verify.get("ok", False)), reason=verify.get("reason", "verified"))

    # 4) Package artifacts consistently (nice for --debug and reports)
    writer = ArtifactWriter()
    artifacts = writer.collect(label=label, mapper={"score": score, **aux}, verify=verify)
    
    out = {
        "ok": bool(verify.get("ok", False)),
        "label": label,
        "score": score,
        "reason": verify.get("reason", "verified"),
        "artifacts": artifacts,
    }
    
    # --- Add verify block for quickstart contract ---
    out["verify"] = {"ok": bool(verify.get("ok", False)), "reason": str(verify.get("reason", "verified"))}
 
    # --- WDD detector (only when audit backend == "wdd") ---

    # Canonical top-1 primitive for downstream use
    seq_canon = []
    if label and label != "abstain":
        seq_canon = [_intent_to_primitive(label)]
    
    audit_cfg = (policy or {}).get("audit") or {}
    if str(audit_cfg.get("backend", "")).lower() == "wdd":
        try:

            # --- Minimal plan for downstream consumers (quickstart expects plan.sequence) ---            
            wdd = wdd_detect(
                prompt=prompt,
                sequence=seq_canon,
                policy=policy,
                context=context,
                pca_prior=audit_cfg.get("pca_prior"),
                debug=bool(audit_cfg.get("debug")),   # <-- add this
            ) or {}
        except Exception as e:
            wdd = {"error": str(e)}
    
        stage11 = out.setdefault("aux", {}).setdefault("stage11", {})
        stage11["wdd"] = {
            "decision":    wdd.get("decision"),
            "sigma":       wdd.get("sigma"),
            "proto_w":     wdd.get("proto_w"),
            "which_prior": wdd.get("which_prior"),
            "mf_peak":     wdd.get("mf_peak"),
            "keep":        wdd.get("keep"),
        }

    # 5) Finish profiling (both files in same timestamped dir)
    if prof_enabled and profile_dir is not None:
        (profile_dir / "profile.json").write_text(json.dumps(prof, indent=2))
        topline = {
            "latency_ms": int(1000 * (time.perf_counter() - t0)),
            "n_keep": len(prof),
            "label": label,
            "score": score,
            "ok": out["ok"],
            "reason": out["reason"],
        }
        (profile_dir / "topline.json").write_text(json.dumps(topline, indent=2))

    return out
