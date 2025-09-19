from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

@dataclass
class IntentResult:
    intent: Optional[str]
    score: float
    topk: List[Tuple[str, float]]
    reason: Optional[str] = None
    aux: Optional[Dict[str, Any]] = None

class IntentMapper(Protocol):
    def infer(self, text: str, *, context: Dict[str, Any] | None = None) -> IntentResult:
        """Return an intent distribution (may be low-confidence)."""
        ...
