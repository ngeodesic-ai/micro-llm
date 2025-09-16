from dataclasses import dataclass
from typing import Any, Dict

from .mapper_api import MapperAPI
from .rails_shim import Rails
from .bench_io import ArtifactWriter


@dataclass(frozen=True)
class RunInputs:
    domain: str
    prompt: str
    context: dict
    policy: dict
    rails: str
    T: int
    backend: str = "sbert"  # Tier-1 default; Tier-0 wordmap available


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
