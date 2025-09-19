from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Protocol

@dataclass
class Context:
    raw: Dict[str, Any]
    oracle_age_sec: Optional[int] = None
    oracle_max_age_sec: Optional[int] = None
    account_balances: Optional[Dict[str, float]] = None
    venues: Optional[list[str]] = None

class ContextAdapter(Protocol):
    def normalize(self, context: Dict[str, Any]) -> Context:
        """Parse/validate incoming context into a normalized Context."""
        ...

    def augment(self, ctx: Context) -> Context:
        """Optionally enrich context (defaults: passthrough)."""
        return ctx
