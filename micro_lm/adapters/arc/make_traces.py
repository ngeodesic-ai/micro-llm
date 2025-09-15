# micro_lm/adapters/arc/make_traces.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any
from ..base import DomainAdapter, AdapterInput
from .mapper import map_prompt  # NEW

# map primitive -> feature key
PRIM_MAP = {
    "flip_h": "sym_h",
    "flip_v": "sym_v",
    "rotate": "rot90",   # use rot90 channel; angle goes into slots
}

class ARCAdapter(DomainAdapter):
    def __init__(self):
        self._boost_key = None
        self._prior_strength = 0.0
        self._last_prior = {}

    def build_state(self, inp: AdapterInput) -> Dict[str, Any]:
        s: Dict[str, Any] = {"slots": {}, "policy": inp.policy or {}}
        mapper_cfg = (inp.policy or {}).get("mapper", {})
        slots, conf = map_prompt(inp.prompt, {}, mapper_cfg)
        s["slots"] = slots
        prim = slots.get("primitive", "unknown")
        if prim == "non_exec":
            s["prior_strength"] = 0.0
            s["non_exec"] = True
        else:
            s["prior_strength"] = float(conf)
        return s

    def build_features(self, state: Dict[str, Any], T: int) -> Dict[str, np.ndarray]:
        t = np.linspace(0, 1, T)
        # simple cues
        sym_h  = 0.6 * (1.0 - np.exp(-4*t))
        sym_v  = 0.6 * (1.0 - np.exp(-4*t))
        rot90  = np.clip(np.sin(2*np.pi*t), 0, None)

        feats = {"sym_h": sym_h, "sym_v": sym_v, "rot90": rot90}

        # intent-aware shaping
        prim = state.get("slots", {}).get("primitive", "")
        boost_key = PRIM_MAP.get(prim)
        prior_strength = float(state.get("prior_strength", 0.0))
        self._boost_key = boost_key
        self._prior_strength = prior_strength

        if boost_key in feats:
            b = 1.0 + 0.6 * prior_strength
            for k in list(feats.keys()):
                if k == boost_key:
                    feats[k] = feats[k] * b + 0.10 * prior_strength
                else:
                    feats[k] = np.maximum(0.0, feats[k] - 0.25 * prior_strength)

        return feats

    def residuals_from_features(self, f: Dict[str, np.ndarray], T: int) -> Dict[str, np.ndarray]:
        relu = lambda x: np.maximum(0.0, x)
        traces = {
            "flip_h": relu(f["sym_h"] - 0.35*f["sym_v"]),
            "flip_v": relu(f["sym_v"] - 0.35*f["sym_h"]),
            "rotate": relu(f["rot90"]),
        }
        bk = self._boost_key
        self._last_prior = {
            "flip_h": 1.0 if bk == "sym_h" else 0.0,
            "flip_v": 1.0 if bk == "sym_v" else 0.0,
            "rotate": 1.0 if bk == "rot90" else 0.0,
        }
        return traces

    def early_safety_flags(self, state: Dict[str, Any], feats: Dict[str, np.ndarray]) -> Dict[str, bool]:
        return {"abstain_non_exec": bool(state.get("non_exec", False))}
