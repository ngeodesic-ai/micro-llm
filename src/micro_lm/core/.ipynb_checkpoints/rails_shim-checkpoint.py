from __future__ import annotations
from typing import Any, Dict


class Rails:
    """
    Thin adapter to the ngeodesic Stage-10/11 rails. Kept separate by design.
    """
    def __init__(self, *, rails: str, T: int):
        self.rails = rails
        self.T = T

    def verify(self, *, domain: str, label: str, context: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
        # Lazy import to keep micro_lm import light and optional
        try:
            import ngeodesic  # noqa: F401
        except Exception:
            # In Stage-1 we only need shape correctness; Stage-2 wires real calls.
            return {"ok": True, "reason": "shim:noop"}

        # TODO(Stage-2): call into ngeodesic with stable schema
        return {"ok": True, "reason": "shim:placeholder"}
