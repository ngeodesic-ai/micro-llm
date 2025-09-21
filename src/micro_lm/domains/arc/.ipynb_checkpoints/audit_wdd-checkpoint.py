from __future__ import annotations
import numpy as np
from typing import Dict, Any
from micro_lm.core.audit import AuditRequest, Mode, run_wdd
from .families_wdd import arc_family_registry

# expected cfg keys: {"mode": "family", "T": 600, "seed": 0}

def wdd_arc_audit(emb: np.ndarray, prototypes: np.ndarray, anchors: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    mode = Mode(cfg.get("mode", "family"))
    T = int(cfg.get("T", 600)); seed = int(cfg.get("seed", 0))
    fams = arc_family_registry(K=prototypes.shape[0]) if mode != Mode.PURE else None
    req = AuditRequest(emb=emb, prototypes=prototypes, anchors=anchors, T=T, seed=seed, families=fams, mode=mode)
    res = run_wdd(req)
    return {"keep": res.keep, "order": res.order, "windows": res.windows, "zfloor": res.zfloor, "debug": res.debug}