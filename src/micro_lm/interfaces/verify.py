# interfaces/verify.py
from typing import Protocol, Dict, Any, List, TypedDict, Optional

class VerifyResult(TypedDict, total=False):
    ok: bool
    reason: str
    tags: List[str]
    aux: Dict[str, Any]

class Verifier(Protocol):
    def __call__(
        self,
        prompt: str,
        plan: Dict[str, Any],
        *,
        context: Dict[str, Any],
        policy: Dict[str, Any]
    ) -> VerifyResult: ...

# example “no-ops are safe” verifier you can swap out
def safe_default_verifier(prompt, plan, *, context, policy) -> VerifyResult:
    # approve only if plan contains exactly one safe, known primitive
    seq = (plan or {}).get("sequence") or []
    known = {"deposit_asset", "swap_asset", "withdraw_asset", "borrow_asset", "repay_asset", "stake_asset", "unstake_asset", "claim_rewards"}
    if len(seq) == 1 and (seq[0] or {}).get("op") in known:
        return {"ok": True, "reason": "single_known_primitive"}
    return {"ok": False, "reason": "low_confidence_or_empty", "tags": ["abstain"]}
