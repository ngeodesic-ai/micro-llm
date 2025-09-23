# src/micro_lm/core/audit/checks.py
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Any

def assert_deterministic(run_once: Callable[[int], Dict[str, Any]], seed: int = 0):
    a = run_once(seed)
    b = run_once(seed)
    keys = ("keep","order","windows","zfloor")
    for k in keys:
        if k == "windows":
            assert a[k].keys() == b[k].keys()
        else:
            assert a[k] == b[k], f"non-deterministic on key {k}"
