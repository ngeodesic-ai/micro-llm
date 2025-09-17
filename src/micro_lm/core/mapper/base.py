# Mapper backend interface (Stage 7) — patched
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any, Protocol

class MapperBackend(Protocol):
    name: str
    def predict(self, prompts: List[str]) -> List[Tuple[Optional[str], float]]: ...

def load_backend(kind: str, **kwargs) -> MapperBackend:
    kind = (kind or "wordmap").lower()
    if kind == "wordmap":
        from micro_lm.domains.defi.mapper_backends.wordmap_backend import WordmapBackend
        return WordmapBackend(**kwargs)
    if kind == "sbert":
        from micro_lm.domains.defi.mapper_backends.sbert_backend import SbertBackend
        return SbertBackend(**kwargs)
    raise ValueError(f"Unknown mapper backend: {kind}")
