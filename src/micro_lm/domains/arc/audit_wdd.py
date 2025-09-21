from __future__ import annotations
import numpy as np
from typing import Dict, Any
from micro_lm.core.audit import AuditRequest, Mode, run_wdd
from micro_lm.core.config import load_domain_config
from .families_wdd import arc_family_registry

def wdd_arc_audit(emb, prototypes, anchors, cfg):
    mode = cfg.get("mode", "family")
    T, seed = int(cfg.get("T", 600)), int(cfg.get("seed", 0))
    # read defaults/overrides if caller didn’t include them
    defaults = cfg.get("defaults")
    overrides = cfg.get("family_overrides")
    fams = arc_family_registry(K=prototypes.shape[0], defaults=defaults, overrides=overrides) if mode != "pure" else None
    req = AuditRequest(emb=emb, prototypes=prototypes, anchors=anchors, T=T, seed=seed, families=fams, mode=mode)
    res = run_wdd(req)
    return {"keep": res.keep, "order": res.order, "windows": res.windows, "zfloor": res.zfloor, "debug": res.debug}