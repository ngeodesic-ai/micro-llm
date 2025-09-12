import numpy as np
from typing import Dict, Any
from ..base import AdapterInput, DomainAdapter
from .state import normalize_arc_context

class ARCAdapter(DomainAdapter):
    PRIMS = ("flip_h", "flip_v", "rotate")

    def build_state(self, inp: AdapterInput) -> Dict[str, Any]:
        return normalize_arc_context(inp.context)

    def build_features(self, state: Dict[str, Any], T: int) -> Dict[str, np.ndarray]:
        t = np.linspace(0, 1, T)
        return {
            "sym_h": np.abs(np.sin(2*np.pi*t)),
            "sym_v": np.abs(np.cos(2*np.pi*t)),
            "rot90": np.exp(-((t-0.6)**2)/(2*0.06**2)),
        }

    def residuals_from_features(self, feats: Dict[str, np.ndarray], T: int) -> Dict[str, np.ndarray]:
        f = feats
        return {
            "flip_h": np.maximum(0.0, f["sym_h"] - 0.5*f["sym_v"]),
            "flip_v": np.maximum(0.0, f["sym_v"] - 0.5*f["sym_h"]),
            "rotate": np.maximum(0.0, f["rot90"]),
        }

    def early_safety_flags(self, state: Dict[str, Any], feats: Dict[str, np.ndarray]) -> Dict[str, bool]:
        return {}
