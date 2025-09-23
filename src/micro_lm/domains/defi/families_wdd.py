# src/micro_lm/domains/defi/families_wdd.py
from typing import List, Dict, Any
from micro_lm.core.audit import FamilySpec

_NAME_TO_IDXS = {
    "swap_like": [0,1,2],
    "deposit_like": [3,4],
    "withdraw_like": [5],
    "stake_like": [6],
    "unstake_like": [7],
    "borrow_like": [8,9],
    "repay_like": [10],
    "claim_rewards_like": [11],
}
def _safe(idxs, K): return [i for i in idxs if 0 <= i < K]

def defi_family_registry(K: int,
                         defaults: Dict[str, Any] = None,
                         overrides: Dict[str, Dict[str, Any]] = None) -> List[FamilySpec]:
    d = {"template_width": 64, "z_abs": 0.55, "keep_frac": 0.70}
    if defaults: d.update(defaults)
    overrides = overrides or {}
    fams: List[FamilySpec] = []
    for name, idxs in _NAME_TO_IDXS.items():
        spec = dict(d); spec.update(overrides.get(name, {}))
        fams.append(FamilySpec(
            name=name,
            idxs=_safe(idxs, K),
            z_abs=float(spec["z_abs"]),
            keep_frac=float(spec["keep_frac"]),
            template_width=int(spec["template_width"]),
        ))
    return fams
