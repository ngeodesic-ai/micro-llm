"""
# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Copyright 2025 Ian C. Moore (Provisional Patents #63/864,726, #63/865,437, #63/871,647 and #63/872,334)
# Email: ngeodesic@gmail.com
# Part of Noetic Geodesic Framework (NGF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from micro_lm.core.config import load_domain_config
from micro_lm.core.audit import AuditRequest, Mode, run_wdd
from .families_wdd import defi_family_registry
import numpy as np

def wdd_defi_audit(emb, prototypes, anchors, cfg):
    # Prefer explicit cfg, else file cfg, else PURE (no family gating)
    file_cfg = load_domain_config("defi")
    mode_str = cfg.get("mode", file_cfg.get("audit", {}).get("mode", "pure"))
    mode = Mode(mode_str)
    T, seed = int(cfg.get("T", 600)), int(cfg.get("seed", 0))
    defaults = cfg.get("defaults")
    overrides = cfg.get("family_overrides")

    if defaults is None or overrides is None:
        defaults = defaults or file_cfg.get("defaults")
        overrides = overrides or file_cfg.get("family_overrides")

    fams = None
    if mode != Mode.PURE:
        fams = defi_family_registry(
            K=prototypes.shape[0], defaults=defaults, overrides=overrides
        )

    req = AuditRequest(emb=emb, prototypes=prototypes, anchors=anchors,
                       T=T, seed=seed, families=fams, mode=mode)
    res = run_wdd(req)
    # Auto-fallback: if FAMILY/PATH yields nothing, rerun as PURE
    if not res.keep and mode != Mode.PURE:
        req_fallback = AuditRequest(emb=emb, prototypes=prototypes, anchors=anchors,
                                    T=T, seed=seed, families=None, mode=Mode.PURE)
        res = run_wdd(req_fallback)

    # Auto-fallback 2 (synthetic safety net): if still empty, pick argmax similarity
    if not res.keep:
        # simple similarity -> choose one best k
        sim = np.asarray(prototypes, dtype=np.float32) @ np.asarray(emb, dtype=np.float32)
        k = int(np.argmax(sim))
        res.keep = [k]
        # if order empty, use keep; give whole-span window
        res.order = [k]
        res.windows = {k: (0, int(T) - 1)}

    return {
        "keep": res.keep, "order": res.order, "windows": res.windows,
        "zfloor": res.zfloor, "env": res.debug.get("env"), "debug": res.debug
    }
