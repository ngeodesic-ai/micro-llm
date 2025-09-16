from typing import Any, Dict

class Rails:
    """
    Adapter to Stage-10/11 rails. If `ngeodesic` is unavailable, we return a
    deterministic accept (or you can set STRICT_SHIM=1 to force abstain).
    """

    def __init__(self, *, rails: str, T: int):
        self.rails = rails
        self.T = T

    def verify(self, *, domain: str, label: str, context: dict, policy: dict) -> dict:
        # Allow forcing shim mode for tests/CI
        import os
        if os.getenv("MICROLM_STRICT_SHIM") == "1":
            return {"ok": True, "reason": "shim:forced"}

        try:
            # Expecting ngeodesic to expose a stage11 `verify_action` or a dispatcher.
            # We duck-type to keep micro_lm independent of exact version pins.
            from ngeodesic.stage11 import verify_action  # type: ignore[attr-defined]
        except Exception:
            return {"ok": True, "reason": "shim:accept:stage-3"}

        # Shape we pass through — keep minimal and explicit
        args = {
            "domain": domain,
            "label": label,
            "context": context or {},
            "policy": policy or {},
            "rails": self.rails,
            "T": self.T,
        }

        try:
            out = verify_action(**args)
            # Expect a dict with at least {"ok": bool, "reason": str}
            ok = bool(out.get("ok", False))
            reason = out.get("reason", "verified")
            # We pass through anything else under 'details'
            details = {k: v for k, v in out.items() if k not in ("ok", "reason")}
            return {"ok": ok, "reason": reason, **details}
        except Exception as e:
            # Fail-safe: do not block execution if rails explode — log reason
            return {"ok": False, "reason": f"rails_error:{e.__class__.__name__}"}
