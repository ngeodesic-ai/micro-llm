# src/micro_lm/domains/arc/families_wdd.py
from typing import List, Dict, Any
from micro_lm.core.audit import FamilySpec

# hard-coded name→idxs map for ARC classes
_NAME_TO_IDXS = {
    "rotate": [0,1,2,3],
    "flip":   [4,5],
    "color_map": [6],
    "translate": [7],
    "crop_pad": [8],
    "tile": [9],
    "rot90": [10],
    "cc_mask_largest": [11],
}

def _safe(idxs, K): return [i for i in idxs if 0 <= i < K]

def arc_family_registry(K: int,
                        defaults: Dict[str, Any] = None,
                        overrides: Dict[str, Dict[str, Any]] = None) -> List[FamilySpec]:
    d = {"template_width": 64, "z_abs": 0.60, "keep_frac": 0.75}
    if defaults: d.update(defaults)
    overrides = overrides or {}
    fams: List[FamilySpec] = []
    for name, idxs in _NAME_TO_IDXS.items():
        spec = dict(d)
        spec.update(overrides.get(name, {}))
        fams.append(FamilySpec(
            name=name,
            idxs=_safe(idxs, K),
            z_abs=float(spec["z_abs"]),
            keep_frac=float(spec["keep_frac"]),
            template_width=int(spec["template_width"]),
        ))
    return fams
