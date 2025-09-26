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

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .pca_prior import load_pca_prior, apply_pca_prior
from .wdd import wdd_audit  # reuse existing decision rule

# Simple LRU-style cache so we don’t repeatedly load the same prior
_PRIOR_CACHE: Dict[str, Dict[str, np.ndarray]] = {}

def _get_prior_from_policy(policy: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    audit_cfg = (policy or {}).get("audit", {})
    prior_path = audit_cfg.get("prior_path")
    if not prior_path:
        return None
    p = str(Path(prior_path))
    if p not in _PRIOR_CACHE:
        _PRIOR_CACHE[p] = load_pca_prior(p)
    return _PRIOR_CACHE[p]

def pca_audit(*, domain: str, emb: Optional[np.ndarray], policy: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    PCA-backed audit that projects an embedding with a prior, then defers to WDD.
    - If no emb is provided, we still run WDD unchanged (no-op projection).
    - If a prior is configured, we project emb -> z and pass z into WDD as 'emb'.
    """
    prior = _get_prior_from_policy(policy)
    proj = None
    if prior is not None and emb is not None:
        proj = apply_pca_prior(emb, prior)

    # Defer to WDD, but prefer the projected vector if available
    out = wdd_audit(domain=domain, emb=(proj if proj is not None else emb), policy=policy, **kwargs)
    # Tag result so downstream artifacts/debugging show the path taken
    out.setdefault("artifacts", {})
    out["artifacts"]["audit_backend"] = "pca"
    if prior is not None:
        out["artifacts"]["pca_prior_shape"] = {
            "mean": list(prior["mean"].shape),
            "components": list(prior["components"].shape),
        }
        if proj is not None:
            out["artifacts"]["pca_emb_dim"] = int(proj.shape[0])
    out.setdefault("reason", "audit:pca→wdd")
    return out
