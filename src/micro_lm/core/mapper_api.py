from __future__ import annotations
from typing import Any, Dict, Tuple
from .backends import sbert, wordmap


class MapperAPI:
    def __init__(self, *, backend: str, domain: str, policy: Dict[str, Any]):
        self.backend = backend
        self.domain = domain
        self.policy = policy
        self.debug = {}

        thr = policy.get("mapper", {}).get("confidence_threshold", 0.5)
        self.threshold = float(thr)

        if backend == "sbert":
            try:
                self.impl = sbert.SBertMapper(domain=domain, policy=policy)
            except Exception as e:
                # degrade to wordmap
                self.impl = wordmap.WordMapMapper(domain=domain, policy=policy)
                self.backend = "wordmap"
                self.debug.update({"degraded": True, "reason": str(e)})
        elif backend == "wordmap":
            self.impl = wordmap.WordMapMapper(domain=domain, policy=policy)
        else:
            raise ValueError(f"Unknown backend: {backend}")

