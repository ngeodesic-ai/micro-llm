from __future__ import annotations
from typing import List
from micro_lm.core.audit import FamilySpec

def arc_family_registry(K: int) -> List[FamilySpec]:
    # Example: rotate (4), flips (2)
    # idx mapping assumed external; adjust to your class→index map
    fams = [
        FamilySpec(name="rotate", idxs=[0,1,2,3], z_abs=0.6, keep_frac=0.75, template_width=64),
        FamilySpec(name="flip",   idxs=[4,5],     z_abs=0.6, keep_frac=0.75, template_width=64),
    ]
    return fams