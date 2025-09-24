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

from .types import AuditRequest, AuditResult, FamilySpec, Mode, Peak
from .pca_prior import load_pca_prior, apply_pca_prior
from .wdd import wdd_audit
from .orchestrator import run_wdd
from importlib import import_module
from typing import Callable, Dict, Any
from .pca_backend import pca_audit
from . import wdd as _wdd
from dataclasses import dataclass
from typing import List, Dict, Any

__all__ = [
     "AuditRequest", "AuditResult", "FamilySpec", "Mode", "Peak", "run_wdd",
    "wdd_audit", "pca_audit", "run_families",
    "get_audit_backend",
    "load_pca_prior",
    "apply_pca_prior",
    "get_audit_backend",
]

# Minimal stub so the notebook smoke test runs now (mapper-free).
def run_families(prompt: str, fams: List[FamilySpec],
                 policy: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    p = prompt.lower()
    order = []
    if any(t in p for t in ("swap","convert","trade","exchange")):
        order.append("swap_asset")
    if any(t in p for t in ("deposit","supply","provide","add liquidity","top up","stake","move")):
        order.append("deposit_asset")
    # de-dupe
    seen = set(); order = [x for x in order if not (x in seen or seen.add(x))]
    return {
        "order": order,
        "keep": [f.name for f in fams],
        "route": ["defi", "wdd", "family"],
        "reason": "audit.facade_stub"
    }

def get_audit_backend(name: str):
    """
    Resolve an audit backend module by name.
    Returns the imported module so callers can use module.audit(...) etc.
    """
    key = (name or "").lower()

    # Common aliases
    if key == "pca":
        return pca_audit
    if key in ("wdd", "wdd_audit"):
        return import_module("micro_lm.core.audit.wdd")
    if key in ("threshold", "tier1", "t1"):
        return import_module("micro_lm.core.audit.threshold")

    # Fallback: try to import a submodule directly by the given key
    try:
        return import_module(f"micro_lm.core.audit.{key}")
    except ModuleNotFoundError as e:
        raise ImportError(f"Unknown audit backend '{name}'") from e

# Optional convenience export: only present if the module imports cleanly.
try:
    from .wdd import wdd_audit  # noqa: F401
except Exception:
    # Keep the symbol present but unusable if wdd isn't available.
    def wdd_audit(*_args, **_kwargs):
        raise RuntimeError("WDD backend not available")
