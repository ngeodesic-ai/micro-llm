from dataclasses import dataclass
from typing import Any, Dict
import os

from .mapper_api import MapperAPI
from .rails_shim import Rails
from .bench_io import ArtifactWriter

# NEW (top of file, with the other imports)
from micro_lm.adapters.simple_context import SimpleContextAdapter
from micro_lm.mappers.joblib_mapper import JoblibMapper, JoblibMapperConfig
from micro_lm.planners.rule_planner import RulePlanner

@dataclass(frozen=True)
class RunInputs:
    domain: str
    prompt: str
    context: dict
    policy: dict
    rails: str
    T: int
    backend: str = "sbert"  # Tier-1 default; Tier-0 wordmap available

def _shim_map_and_plan(user_text: str, *, context: dict, policy: dict) -> dict:
    """
    Local shim that uses the small interfaces:
    SimpleContextAdapter -> JoblibMapper -> RulePlanner
    Returns a dict with (label, score, reason, artifacts) in the same shape
    expected by run_micro downstream.
    """
    adapter = SimpleContextAdapter()
    model_path = policy.get("mapper", {}).get("model_path", ".artifacts/defi_mapper.joblib")
    # If the model isn't available (e.g. unit tests), quietly abstain.
    if not os.path.exists(model_path):
        ctx = adapter.normalize(context)
        return {
            "label": "abstain",
            "score": 0.0,
            "reason": "shim:model_missing",
            "artifacts": {"shim": {"model_path": model_path, "ctx": ctx.raw}},
        }
    mapper = JoblibMapper(JoblibMapperConfig(
        model_path=model_path,
        confidence_threshold=policy.get("mapper", {}).get("confidence_threshold", 0.7),
    ))
    planner = RulePlanner()

    ctx = adapter.normalize(context)
    mres = mapper.infer(user_text, context=ctx.raw)   # -> has fields: intent, score, topk, etc.

    if not getattr(mres, "intent", None):
        return {
            "label": "abstain",
            "score": getattr(mres, "score", 0.0),
            "reason": "low_confidence",
            "artifacts": {"mapper": mres.__dict__},
        }

    plan = planner.plan(intent=mres.intent, text=user_text, context=ctx.raw)
    artifacts = {"mapper": mres.__dict__,  "plan": getattr(plan, "__dict__", {})}

    return {
        "label": mres.intent,
        "score": getattr(mres, "score", 1.0),
        "reason": "shim:mapper",
        "artifacts": artifacts,
    }


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
    PUBLIC API (frozen at Stage-1).
    Returns a standardized dict with keys: 'ok', 'label', 'reason', 'artifacts'.
    """
    # 1) Map prompt -> label (or 'abstain') using selected backend
    mapper = MapperAPI(backend=backend, domain=domain, policy=policy)
    label, score, aux = mapper.map_prompt(prompt)

    # 1b) Optional shim fallback (use if Tier-1 abstains or if explicitly enabled)
    # 1b) Optional shim fallback (skip for Tier-0 'wordmap' to keep tests hermetic)
    use_shim = policy.get("mapper", {}).get("use_shim_fallback", backend != "wordmap")
    if (label == "abstain") and use_shim and backend != "wordmap":
        shim_out = _shim_map_and_plan(prompt, context=context, policy=policy)
        label, score = shim_out["label"], shim_out.get("score", score)
        # merge artifacts+reason so downstream packaging sees them
        aux = {
            "reason": shim_out.get("reason", aux.get("reason", "shim")),
            "artifacts": {**aux.get("artifacts", {}), **shim_out.get("artifacts", {})},
        }

    # 2) If abstain or no rails requested, return early
    if label == "abstain" or not rails:
        return {
            "ok": label != "abstain",
            "label": label,
            "score": score,
            "reason": aux.get("reason", "abstain" if label == "abstain" else "mapped"),
            "artifacts": aux.get("artifacts", {}),
        }

    # 3) Execute rails (delegated via shim; Stage-2 wires real ngeodesic)
    rails_exec = Rails(rails=rails, T=T)
    verify = rails_exec.verify(domain=domain, label=label, context=context, policy=policy)

    # 4) Package artifacts in a consistent shape
    writer = ArtifactWriter()
    artifacts = writer.collect(label=label, mapper={"score": score, **aux}, verify=verify)

    return {
        "ok": bool(verify.get("ok", False)),
        "label": label,
        "score": score,
        "reason": verify.get("reason", "verified"),
        "artifacts": artifacts,
    }
