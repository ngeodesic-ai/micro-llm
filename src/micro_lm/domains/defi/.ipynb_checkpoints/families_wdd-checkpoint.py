from __future__ import annotations
from typing import List
from micro_lm.core.audit import FamilySpec

def _safe(idxs, K):  # keep only valid indices
    return [i for i in idxs if 0 <= i < K]

def defi_family_registry(K: int) -> List[FamilySpec]:
    return [
        FamilySpec(name="swap_like",    idxs=_safe([0,1,2], K), z_abs=0.5, keep_frac=0.7,  template_width=64),
        FamilySpec(name="deposit_like", idxs=_safe([3,4],   K), z_abs=0.5, keep_frac=0.7,  template_width=64),
        FamilySpec(name="borrow_like",  idxs=_safe([5,6],   K), z_abs=0.6, keep_frac=0.75, template_width=64),
    ]
