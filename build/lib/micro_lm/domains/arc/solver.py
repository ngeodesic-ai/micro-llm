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
from typing import Dict, Any, List, Tuple
import numpy as np
from .args import extract_args
from .exec import replay

# Simple D4 canonicalization helpers
_D4 = [
    lambda x: x,
    lambda x: np.rot90(x, 1),
    lambda x: np.rot90(x, 2),
    lambda x: np.rot90(x, 3),
    lambda x: np.flipud(x),
    lambda x: np.fliplr(x),
    lambda x: np.rot90(np.flipud(x), 1),
    lambda x: np.rot90(np.fliplr(x), 1),
]

def _canon_best(img: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return the D4 transform of `img` with lexicographically-smallest bytes, and its index."""
    cands = [f(img) for f in _D4]
    # Compare by raw bytes of the flattened array for a strict total order.
    idx = min(range(len(cands)), key=lambda i: cands[i].tobytes())
    return cands[idx], idx

def _uncanon(img: np.ndarray, idx: int) -> np.ndarray:
    # apply the inverse transform (same set closed under inverse)
    # brute force search for inverse by reapplying until match
    for j, f in enumerate(_D4):
        if np.array_equal(_D4[j]( _D4[idx](img) ), img):
            return _D4[j](img)
    return img

def solve_arc_io(io_pair: Dict[str,Any], audit_out: Dict[str,Any], policy: Dict[str,Any]) -> Dict[str,Any]:
    """Solve ARC-like I/O with ops extracted from audit.
    io_pair: {"input": HxW ndarray, "output": HxW ndarray}
    audit_out: result of wdd_arc_audit with keys keep/order/windows/env/labels (labels optional)
    returns: {"ops": [...], "verified": bool, "reason": str}
    """
    img_in = io_pair["input"]; img_out = io_pair["output"]
    # Canonicalize INPUT, and carry the SAME transform to OUTPUT so both are in one frame.
    img_in_c, idx_in = _canon_best(img_in)
    img_out_c = _D4[idx_in](img_out)

    keep = audit_out.get("keep", [])
    order = audit_out.get("order", [])
    windows = audit_out.get("windows", {})
    env = audit_out.get("env")  # (T,K)
    labels = audit_out.get("labels", [str(k) for k in range(env.shape[1])]) if env is not None else []

    if env is None or env.ndim != 2:
        return {"ops": [], "verified": False, "reason": "no_env"}

    ops = extract_args(keep, order, windows, env, labels)

    # beam: top-1 (deterministic) → extendable later
    img_hat_c = replay(img_in_c, ops, L_MAX=int(policy.get("arc", {}).get("L_MAX", 3)))

    verified = np.array_equal(img_hat_c, img_out_c)
    reason = "ok" if verified else "mismatch"
    return {"ops": ops, "verified": bool(verified), "reason": reason}