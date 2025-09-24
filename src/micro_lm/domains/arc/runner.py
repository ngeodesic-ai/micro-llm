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

"""
ARC domain scaffold for micro_lm

Drop this file at: src/micro_lm/domains/arc/runner.py

Provides:
- Primitive ops: flip_h, flip_v, rotate_cw, rotate_ccw
- Tiny intent mapper (regex/keyword) for ARC prompts
- Planner: turns an intent into a single-step plan [primitive]
- Executor: applies primitive to context["grid"] and returns updated grid
- register_arc(): optional helper to hook into a minimal registry so
  other parts of the system can find this domain runner.

This is intentionally self-contained so you can iterate without touching
DeFi code paths.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

Grid = List[List[Any]]

# -------------------------
# Primitives (pure functions)
# -------------------------

def flip_h(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in grid]

def flip_v(grid: Grid) -> Grid:
    return list(reversed([list(row) for row in grid]))

def rotate_cw(grid: Grid) -> Grid:
    # 90° clockwise: transpose + reverse rows
    return [list(row)[::-1] for row in zip(*grid)]

def rotate_ccw(grid: Grid) -> Grid:
    # 90° counter-clockwise: reverse rows + transpose
    return list(map(list, zip(*grid[::-1])))

PRIMS = {
    "flip_h": flip_h,
    "flip_v": flip_v,
    "rotate_cw": rotate_cw,
    "rotate_ccw": rotate_ccw,
}

# -------------------------
# Mapper (prompt -> intent)
# -------------------------

@dataclass
class MapOut:
    intent: str
    score: float
    reason: str

_INTENT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("flip_h", re.compile(r"\b(flip|mirror).*(horiz|left\s*to\s*right|right\s*to\s*left)", re.I)),
    ("flip_v", re.compile(r"\b(flip|mirror).*(vert|top\s*to\s*bottom|bottom\s*to\s*top)", re.I)),
    ("rotate_cw", re.compile(r"\b(rotat(e|ion)).*(clock\s*wise|cw|90)\b", re.I)),
    ("rotate_ccw", re.compile(r"\b(rotat(e|ion)).*(counter|anti).*wise|ccw|(-|\s)90\b", re.I)),
]

_KEYWORDS = {
    "flip_h": ["flip horizontally", "horizontal flip", "mirror horizontally"],
    "flip_v": ["flip vertically", "vertical flip", "mirror vertically"],
    "rotate_cw": ["rotate clockwise", "rotate 90", "rotate right"],
    "rotate_ccw": ["rotate counter", "rotate anticlockwise", "rotate left"],
}


def map_prompt_arc(prompt: str, *, fallback_threshold: float = 0.51) -> Optional[MapOut]:
    p = prompt.strip()
    if not p:
        return None

    # 1) regex patterns (higher confidence)
    for intent, pat in _INTENT_PATTERNS:
        if pat.search(p):
            return MapOut(intent=intent, score=0.95, reason="regex_match")

    # 2) keyword contains (medium confidence)
    low: List[Tuple[str, float]] = []
    lp = p.lower()
    for intent, kws in _KEYWORDS.items():
        for kw in kws:
            if kw in lp:
                low.append((intent, 0.75))
                break

    if low:
        low.sort(key=lambda t: t[1], reverse=True)
        best = low[0]
        return MapOut(intent=best[0], score=best[1], reason="keyword_match")

    # 3) soft heuristics (very low confidence)
    if "flip" in lp and "horiz" in lp:
        return MapOut(intent="flip_h", score=fallback_threshold, reason="heuristic")
    if "flip" in lp and "vert" in lp:
        return MapOut(intent="flip_v", score=fallback_threshold, reason="heuristic")
    if "rotate" in lp and ("cw" in lp or "clock" in lp or "90" in lp):
        return MapOut(intent="rotate_cw", score=fallback_threshold, reason="heuristic")
    if "rotate" in lp and ("ccw" in lp or "counter" in lp or "anti" in lp):
        return MapOut(intent="rotate_ccw", score=fallback_threshold, reason="heuristic")

    return None

# -------------------------
# Planner (intent -> plan)
# -------------------------

@dataclass
class Plan:
    sequence: List[str]
    aux: Dict[str, Any]


def plan_arc(intent: Optional[str]) -> Plan:
    if not intent or intent not in PRIMS:
        return Plan(sequence=[], aux={"abstain": True, "reason": "no_intent"})
    return Plan(sequence=[intent], aux={})

# -------------------------
# Verifier / Executor
# -------------------------

@dataclass
class Verify:
    ok: bool
    reason: str = ""


def _get_grid(context: Dict[str, Any]) -> Optional[Grid]:
    g = context.get("grid")
    if g is None:
        return None
    # light validation
    if not isinstance(g, list) or not g or not isinstance(g[0], list):
        return None
    return g


def execute_arc(plan: Plan, context: Dict[str, Any]) -> Dict[str, Any]:
    grid = _get_grid(context)
    if grid is None:
        return {"verify": Verify(ok=False, reason="no_grid").__dict__, "result": None}

    if not plan.sequence:
        return {"verify": Verify(ok=False, reason="abstain_non_exec").__dict__, "result": grid}

    prim = plan.sequence[0]
    fn = PRIMS.get(prim)
    if not fn:
        return {"verify": Verify(ok=False, reason="unknown_primitive").__dict__, "result": grid}

    try:
        out = fn(grid)
        return {"verify": Verify(ok=True, reason="executed").__dict__, "result": out}
    except Exception as e:
        return {"verify": Verify(ok=False, reason=f"exec_error:{e}").__dict__, "result": None}

# -------------------------
# Public runner
# -------------------------


def run_arc(prompt: str, *, context: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    """End-to-end ARC runner (mapper -> plan -> execute)."""
    m = map_prompt_arc(prompt) or MapOut(intent="abstain", score=0.0, reason="no_match")
    pl = plan_arc(m.intent if m.intent != "abstain" else None)
    exe = execute_arc(pl, context)

    return {
        "ok": bool(exe["verify"]["ok"]),
        "label": m.intent if m.intent != "abstain" else "abstain",
        "score": float(m.score),
        "reason": exe["verify"]["reason"] if pl.sequence else m.reason,
        "plan": {"sequence": pl.sequence},
        "verify": exe["verify"],
        "result": exe["result"],
        "aux": {"mapper_reason": m.reason},
    }

# -------------------------
# Registry hook (optional)
# -------------------------

# If you maintain a simple domain registry elsewhere in the package, you can
# import and call register_arc() to make this discoverable.
_DOMAIN_REGISTRY: Dict[str, Any] = {}


def register_arc() -> None:
    _DOMAIN_REGISTRY["arc"] = run_arc


def get_domain_runner(name: str):
    return _DOMAIN_REGISTRY.get(name)

# -------------------------
# Minimal self-test
# -------------------------

if __name__ == "__main__":
    g = [[0,1,2],[3,4,5]]
    r = run_arc("flip the grid horizontally", context={"grid": g}, policy={})
    assert r["plan"]["sequence"] == ["flip_h"], r
    assert r["ok"] and r["result"] == [[2,1,0],[5,4,3]], r
    print("ARC scaffold OK:\n", r)
