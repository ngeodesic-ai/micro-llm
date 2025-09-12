# micro_llm/adapters/defi/make_traces.py
import numpy as np
from typing import Dict, Any
from ..base import AdapterInput, DomainAdapter
from .state import normalize_defi_context
from .prompt_map import parse_prompt

PRIM_MAP = {
    "deposit_asset": "deposit",
    "withdraw_asset": "withdraw",
    "borrow_asset": "borrow",
    "repay_loan": "repay",
    "swap_asset": "swap",
    "swap_asset_pair": "swap",
}


class DeFiAdapter(DomainAdapter):
    PRIMS = (
        "deposit_asset", "withdraw_asset",
        "borrow_asset", "repay_loan",
        "swap_asset", "add_collateral", "remove_collateral"
    )

    def __init__(self):
        # carry prompt prior across methods
        self._boost_key = None
        self._prior_strength = 0.0
        self._last_prior = {}

    def build_state(self, inp: AdapterInput) -> Dict[str, Any]:
        s = normalize_defi_context(inp.context)
        s["policy"] = inp.policy
        slots = parse_prompt(inp.prompt)
        s["slots"] = slots
        if slots.get("primitive") == "non_exec":
            s["prior_strength"] = 0.0
            s["non_exec"] = True
        else:
            s["prior_strength"] = 0.85 if slots.get("primitive") in PRIM_MAP else 0.0
        return s

    def early_safety_flags(self, state, feats):
        flags = {}
        if state.get("non_exec"):
            flags["abstain_non_exec"] = True
        o = state.get("oracle", {})
        flags["oracle_stale"] = bool(o.get("age_sec", 0) > o.get("max_age_sec", 30))
        return flags

    def build_features(self, state: Dict[str, Any], T: int) -> Dict[str, np.ndarray]:
        t = np.linspace(0, 1, T)
        hf = float(state.get("risk", {}).get("hf", 1.2))

        borrow_cue = np.clip(hf - 1.05, 0, 1) * (0.2 + 0.8*t)
        repay_cue  = np.clip(1.05 - hf, 0, 1) * (1.0 - t)
        swap_cue   = np.abs(np.sin(2*np.pi*t))
        deposit_cue  = 0.6*(1.0 - np.exp(-4*t))
        withdraw_cue = 0.6*np.exp(-4*t) * np.clip(hf - 1.25, 0, 1)
        addcol_cue = np.clip(1.0 - hf, 0, 1) * np.ones_like(t)
        remcol_cue = np.clip(hf - 1.3, 0, 1) * (0.5 + 0.5*np.cos(2*np.pi*t))

        feats = {
            "deposit":  deposit_cue,
            "withdraw": withdraw_cue,
            "borrow":   borrow_cue,
            "repay":    repay_cue,
            "swap":     swap_cue,
            "addcol":   addcol_cue,
            "remcol":   remcol_cue,
        }

    
        # ---- intent-aware shaping ----
        prim = state.get("slots", {}).get("primitive", "")
        boost_key = PRIM_MAP.get(prim)
        prior_strength = float(state.get("prior_strength", 0.0))
        
        self._boost_key = boost_key
        self._prior_strength = prior_strength
        
        if boost_key in feats:
            b = 1.0 + 0.6 * prior_strength
            for k in feats:
                if k == boost_key:
                    feats[k] = feats[k] * b + 0.10 * prior_strength
                else:
                    # stronger inhibition across the board
                    feats[k] = np.maximum(0.0, feats[k] - 0.25 * prior_strength)
        
            # targeted hush for confusing pairs
            if boost_key == "deposit":
                feats["borrow"] *= (1.0 - 0.90 * prior_strength)   # was 0.50
                feats["repay"]  *= (1.0 - 0.40 * prior_strength)   # was 0.20
            elif boost_key == "borrow":
                feats["deposit"] *= (1.0 - 0.70 * prior_strength)

        return feats

    def residuals_from_features(self, f: Dict[str, np.ndarray], T: int) -> Dict[str, np.ndarray]:
        relu = lambda x: np.maximum(0.0, x)
        traces = {
            "deposit_asset":     relu(f["deposit"]  - 0.35*f["withdraw"]),
            "withdraw_asset":    relu(f["withdraw"] - 0.35*f["deposit"]),
            "borrow_asset":      relu(f["borrow"]   - 0.35*f["repay"]),
            "repay_loan":        relu(f["repay"]    - 0.35*f["borrow"]),
            "swap_asset":        relu(f["swap"]),
            "add_collateral":    relu(f["addcol"]),
            "remove_collateral": relu(f["remcol"]),
        }

        # build per-primitive prior for Stage-10 fallback weighting
        bk = self._boost_key
        self._last_prior = {
            "deposit_asset":     1.0 if bk == "deposit"  else 0.0,
            "withdraw_asset":    1.0 if bk == "withdraw" else 0.0,
            "borrow_asset":      1.0 if bk == "borrow"   else 0.0,
            "repay_loan":        1.0 if bk == "repay"    else 0.0,
            "swap_asset":        1.0 if bk == "swap"     else 0.0,
            "add_collateral":    0.0,
            "remove_collateral": 0.0,
        }
        return traces
