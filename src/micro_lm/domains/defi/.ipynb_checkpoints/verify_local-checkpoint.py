from typing import Dict, Any
from .schema import DEFI_LABELS, REASONS
from .guards import check_oracle, check_ltv, check_hf

def verify_action_local(*, label: str, context: dict, policy: dict) -> Dict[str, Any]:
    # Only handle our known labels locally; otherwise abstain to shim/rails.
    if label not in DEFI_LABELS:
        return {"ok": False, "reason": REASONS["abstain"]}

    # Oracle freshness
    o = check_oracle(context)
    if not o["ok"]:
        return {"ok": False, "reason": REASONS["oracle"]}

    # Basic policy checks (apply only to exec-like labels)
    if label in {"withdraw_asset", "borrow_asset"}:
        ltvc = check_ltv(policy, context)
        if not ltvc["ok"]:
            return {"ok": False, "reason": REASONS["ltv"]}

    if label in {"borrow_asset"}:
        hfc = check_hf(policy, context)
        if not hfc["ok"]:
            return {"ok": False, "reason": REASONS["hf"]}

    # If we get here, it’s locally “safe”; the rails shim can still refine.
    return {"ok": True, "reason": REASONS["ok"]}
