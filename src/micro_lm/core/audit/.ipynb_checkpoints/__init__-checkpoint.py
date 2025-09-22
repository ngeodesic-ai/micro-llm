from .types import AuditRequest, AuditResult, FamilySpec, Mode, Peak
from .pca_prior import load_pca_prior, apply_pca_prior
from .wdd import wdd_audit
from .orchestrator import run_wdd
from importlib import import_module
from typing import Callable, Dict, Any
from .pca_backend import pca_audit
from . import wdd as _wdd

__all__ = [
     "AuditRequest", "AuditResult", "FamilySpec", "Mode", "Peak", "run_wdd",
    "wdd_audit", "pca_audit",
    "get_audit_backend",
    "load_pca_prior",
    "apply_pca_prior",
    "get_audit_backend",
]

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

