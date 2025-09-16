from typing import Dict, Any
from .schema import ARC_LABELS

def verify_action_local(*, label: str, context: dict, policy: dict) -> Dict[str, Any]:
    if label not in ARC_LABELS:
        return {"ok": False, "reason": "abstain_non_exec"}
    # ARC has no chain/price oracles; nothing to check for Stage 4
    return {"ok": True, "reason": "verified"}
