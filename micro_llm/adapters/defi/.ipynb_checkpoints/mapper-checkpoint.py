# Minimal hybrid mapper: (A) try trained model if present, else (B) rules.
from __future__ import annotations
import json, re, pathlib
from typing import Dict, Any, Tuple, Optional

try:
    import joblib  # for loading sklearn pipeline
except Exception:
    joblib = None  # still fine for rule fallback

# --- Rule fallback (reuse your existing regexes) ---
AMOUNT = r"(?P<amount>\d+(?:\.\d+)?)"
ASSET  = r"(?P<asset>[A-Z]{3,6})"
PAIR   = r"(?P<base>[A-Z]{3,6})\/(?P<quote>[A-Z]{3,6})"

_ASSET = r"[A-Za-z0-9_\-.:/]+"
_AMT   = r"\d+(?:\.\d+)?"
_VENUE = r"[A-Za-z0-9_\-]+"

NON_EXEC_PATTERNS = [
    r"(check|show|what('| i)?s)\s+(my\s+)?(balance|health\s*factor|hf)\b",
    r"(how\s+much)\s+(can\s+i\s+borrow|ltv)\b",
    r"(price|oracle)\s+(age|status)\b",
]

PATTERNS = [
    (rf"(deposit|supply)\s+{AMOUNT}\s+{ASSET}", "deposit_asset"),
    (rf"(withdraw|redeem)\s+{AMOUNT}\s+{ASSET}", "withdraw_asset"),
    (rf"(borrow)\s+{AMOUNT}\s+{ASSET}", "borrow_asset"),
    (rf"(repay|pay back)\s+{AMOUNT}\s+{ASSET}", "repay_loan"),
    (rf"(swap|exchange)\s+{AMOUNT}\s+{ASSET}\s+for\s+{ASSET}", "swap_asset"),
    (rf"(swap)\s+{PAIR}", "swap_asset_pair"),
]

RULES = [
    # --- deposit/supply/add collateral ---
    ("deposit_asset", [
        rf"\b(?:deposit|supply|add)\s+(?P<amount>{_AMT})\s*(?P<asset>{_ASSET})(?:\s+(?:into|to)\s+(?P<venue>{_VENUE}))?\b",
        rf"\b(?:deposit|supply|add)\s+(?P<asset>{_ASSET})\s+(?P<amount>{_AMT})(?:\s+(?:into|to)\s+(?P<venue>{_VENUE}))?\b",
    ]),
    # --- withdraw/remove collateral ---
    ("withdraw_asset", [
        rf"\b(?:withdraw|remove)\s+(?P<amount>{_AMT})\s*(?P<asset>{_ASSET})\b",
        rf"\b(?:withdraw|remove)\s+(?P<asset>{_ASSET})\s+(?P<amount>{_AMT})\b",
    ]),
    # --- borrow/take loan ---
    ("borrow_asset", [
        rf"\b(?:borrow|take(?:\s+out)?)\s+(?P<amount>{_AMT})\s*(?P<asset>{_ASSET})\b",
    ]),
    # --- repay/pay back ---
    ("repay_loan", [
        rf"\b(?:repay|pay\s*back)\s+(?P<amount>{_AMT})\s*(?P<asset>{_ASSET})?\b",
    ]),
    # --- swap/trade ---
    ("swap_asset", [
        rf"\b(?:swap|trade)\s+(?P<amount>{_AMT})\s*(?P<asset>{_ASSET})\s+(?:for|to)\s+(?P<dst_asset>{_ASSET})\b",
        rf"\b(?:swap|trade)\s+(?P<asset>{_ASSET})\s+(?P<amount>{_AMT})\s+(?:for|to)\s+(?P<dst_asset>{_ASSET})\b",
    ]),
    # --- add/remove collateral (explicit verbs) ---
    ("add_collateral", [
        rf"\b(?:add\s+collateral)\s+(?P<amount>{_AMT})\s*(?P<asset>{_ASSET})\b",
    ]),
    ("remove_collateral", [
        rf"\b(?:remove\s+collateral)\s+(?P<amount>{_AMT})\s*(?P<asset>{_ASSET})\b",
    ]),
]

INTENTS = [
    "deposit_asset","withdraw_asset","borrow_asset","repay_loan","swap_asset","swap_asset_pair","non_exec","unknown"
]

# # Old code - pytest -v
# def _rule_map(prompt: str) -> Tuple[Dict[str, Any], float]:
#     p = prompt.strip()
#     for pat in NON_EXEC_PATTERNS:
#         if re.search(pat, p, flags=re.I):
#             return ({"primitive": "non_exec", "query": p}, 1.0)
#     for pat, prim in PATTERNS:
#         m = re.search(pat, p, flags=re.I)
#         if m:
#             d = {k: (v.upper() if isinstance(v, str) else v)
#                  for k, v in m.groupdict(default="").items()}
#             return ({"primitive": prim, **d}, 0.85)
#     return ({"primitive": "unknown"}, 0.0)


def _rule_map(prompt: str) -> Tuple[Dict[str, Any], float]:
    p = prompt.strip()
    # 1) Non-exec first: map to explicit "non_exec" so rails/runner abstain cleanly
    for pat in NON_EXEC_PATTERNS:
        if re.search(pat, p, flags=re.I):
            return {"primitive": "non_exec", "query": p}, 1.0

    # 2) Executable primitives via simple patterns (no duplicate group names)
    for prim, pats in RULES:
        for pat in pats:
            m = re.search(pat, p, flags=re.I)
            if m:
                slots = {"primitive": prim}
                g = m.groupdict()
                if g.get("amount"):     slots["amount"]     = g["amount"]
                if g.get("asset"):      slots["asset"]      = g["asset"].upper()
                if g.get("dst_asset"):  slots["dst_asset"]  = g["dst_asset"].upper()
                if g.get("venue"):      slots["venue"]      = g["venue"].lower()
                return slots, 0.92  # rule-based confidence

    # 3) Unknown → let mapper blend or rails abstain later
    return {"primitive": "unknown"}, 0.0



# --- Hybrid mapper ---
class HybridMapper:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        if model_path and joblib:
            mp = pathlib.Path(model_path)
            if mp.exists():
                self.model = joblib.load(mp)  # sklearn Pipeline with predict_proba

    def map_prompt(self, prompt: str, ctx: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Returns (slots, confidence in [0,1])
        """
        # (A) trained model available?
        if self.model:
            try:
                proba = self.model.predict_proba([prompt])[0]
                labels = self.model.classes_
                best_idx = int(proba.argmax())
                intent = labels[best_idx] if best_idx < len(labels) else "unknown"
                conf = float(proba[best_idx])
                # slots via light regex pass (still helpful)
                slots, _ = _rule_map(prompt)
                slots["primitive"] = intent
                return slots, conf
            except Exception:
                pass  # fall back to rules

        # (B) rules only
        return _rule_map(prompt)

# default singleton (adapter will import and use)
_mapper_singleton: Optional[HybridMapper] = None

def get_mapper(cfg: Dict[str, Any]) -> HybridMapper:
    global _mapper_singleton
    if _mapper_singleton is None:
        model_path = cfg.get("model_path")
        _mapper_singleton = HybridMapper(model_path=model_path)
    return _mapper_singleton

def map_prompt(prompt: str, ctx: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    return get_mapper(cfg).map_prompt(prompt, ctx, cfg)

def rule_only(prompt: str) -> Tuple[Dict[str, Any], float]:
    # public wrapper so other modules don't import a private name
    return _rule_map(prompt)
