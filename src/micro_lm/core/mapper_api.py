from __future__ import annotations
from typing import Any, Dict, Tuple
from .backends import sbert, wordmap


class MapperAPI:
    """
    Thin façade over interchangeable backends.
    Enforces abstain-first safety with a confidence gate (from policy).
    """

    def __init__(self, *, backend: str, domain: str, policy: Dict[str, Any]):
        self.backend = backend
        self.domain = domain
        self.policy = policy
        thr = policy.get("mapper", {}).get("confidence_threshold", 0.5)
        self.threshold = float(thr)

        if backend == "sbert":
            self.impl = sbert.SBertMapper(domain=domain, policy=policy)
        elif backend == "wordmap":
            self.impl = wordmap.WordMapMapper(domain=domain, policy=policy)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def map_prompt(self, prompt: str) -> Tuple[str, float, Dict[str, Any]]:
        label, score, aux = self.impl.map_prompt(prompt)
        if score < self.threshold:
            return "abstain", score, {"reason": "low_confidence", "aux": aux}
        return label, score, {"reason": aux.get("reason", "mapped"), "aux": aux}
