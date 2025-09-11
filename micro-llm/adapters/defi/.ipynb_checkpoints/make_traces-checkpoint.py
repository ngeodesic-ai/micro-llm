import numpy as np
from typing import Dict, Any
from ..base import AdapterInput, DomainAdapter
from .state import normalize_defi_context

class DeFiAdapter(DomainAdapter):
    PRIMS = ("borrow_asset", "repay_loan", "swap_asset", "add_collateral", "remove_collateral")

    def build_state(self, inp: AdapterInput) -> Dict[str, Any]:
        s = normalize_defi_context(inp.context)
        s["policy"] = inp.policy
        return s

    def build_features(self, state: Dict[str, Any], T: int) -> Dict[str, np.ndarray]:
        t = np.linspace(0, 1, T)
        hf = float(state.get("risk", {}).get("hf", 1.2))
        borrow_cue = np.clip(hf - 1.05, 0, 1) * (0.2 + 0.8*t)
        repay_cue  = np.clip(1.05 - hf, 0, 1) * (1.0 - t)
        swap_cue   = np.abs(np.sin(2*np.pi*t))
        addcol_cue = np.clip(1.0 - hf, 0, 1) * np.ones_like(t)
        remcol_cue = np.clip(hf - 1.3, 0, 1) * (0.5 + 0.5*np.cos(2*np.pi*t))
        return {"borrow": borrow_cue, "repay": repay_cue, "swap": swap_cue, "addcol": addcol_cue, "remcol": remcol_cue}

    def residuals_from_features(self, f: Dict[str, np.ndarray], T: int) -> Dict[str, np.ndarray]:
        relu = lambda x: np.maximum(0.0, x)
        return {
            "borrow_asset":     relu(f["borrow"] - 0.4*f["repay"]),
            "repay_loan":       relu(f["repay"]  - 0.4*f["borrow"]),
            "swap_asset":       relu(f["swap"]),
            "add_collateral":   relu(f["addcol"]),
            "remove_collateral":relu(f["remcol"]),
        }

    def early_safety_flags(self, state: Dict[str, Any], feats: Dict[str, np.ndarray]) -> Dict[str, bool]:
        o = state.get("oracle", {})
        return {"oracle_stale": bool(o.get("age_sec", 0) > o.get("max_age_sec", 30))}
