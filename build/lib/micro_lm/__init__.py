# src/micro_lm/__init__.py
"""micro_lm — Tier-2 WDD engine with DeFi + ARC domains."""

# Entrypoint(s)
from .core.runner import run_micro  # stable call surface

# WDD family registry (DeFi)
from .domains.defi.families_wdd import defi_family_registry

__all__ = [
    "run_micro",
    "defi_family_registry",
]

__version__ = "2.0.0-tier2"
