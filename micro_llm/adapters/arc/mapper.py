# Minimal hybrid-ish mapper for ARC. Tries a trained model if you add one later;
# otherwise uses clean regex rules that return (slots, confidence).
from __future__ import annotations
import re
from typing import Dict, Any, Tuple, Optional

try:
    import joblib
except Exception:
    joblib = None

INTENTS = ["flip_h", "flip_v", "rotate", "non_exec", "unknown"]

def _rule_map(prompt: str) -> Tuple[Dict[str, Any], float]:
    p = prompt.strip().lower()

    # non-exec queries (describe-only, ask-about)
    if re.search(r"\b(show|describe|what|visualize)\b", p):
        return ({"primitive": "non_exec"}, 1.0)

    # horizontal flip
    if re.search(r"\bflip\b.*\b(horiz|left|right)\b", p) or "flip the grid horizontally" in p:
        return ({"primitive": "flip_h"}, 0.9)

    # vertical flip
    if re.search(r"\bflip\b.*\b(vert|up|down)\b", p) or "flip the grid vertically" in p:
        return ({"primitive": "flip_v"}, 0.9)

    # rotations (capture angle if present, normalize to 90/180/270)
    m = re.search(r"\brotate\b(?:\s+by)?\s+(\d{2,3})", p)
    if m:
        ang = int(m.group(1)) % 360
        if ang in (90, 180, 270):
            return ({"primitive": "rotate", "angle": ang}, 0.9)
    if re.search(r"\brotate\b", p):
        # default rotate if angle omitted
        return ({"primitive": "rotate", "angle": 90}, 0.8)

    return ({"primitive": "unknown"}, 0.0)

class HybridMapperARC:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path and joblib:
            try:
                self.model = joblib.load(model_path)
            except Exception:
                self.model = None

    def map_prompt(self, prompt: str, ctx: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        # If you later add a model: predict intent; keep rules for slots/angle.
        if self.model:
            try:
                proba = self.model.predict_proba([prompt])[0]
                labels = self.model.classes_
                idx = int(proba.argmax())
                intent = labels[idx] if idx < len(labels) else "unknown"
                conf = float(proba[idx])
                slots, _ = _rule_map(prompt)
                slots["primitive"] = intent
                return slots, conf
            except Exception:
                pass
        return _rule_map(prompt)

# public helpers
_mapper_singleton: Optional[HybridMapperARC] = None

def get_mapper(cfg: Dict[str, Any]) -> HybridMapperARC:
    global _mapper_singleton
    if _mapper_singleton is None:
        _mapper_singleton = HybridMapperARC(cfg.get("model_path"))
    return _mapper_singleton

def map_prompt(prompt: str, ctx: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    return get_mapper(cfg).map_prompt(prompt, ctx, cfg)

def rule_only(prompt: str) -> Tuple[Dict[str, Any], float]:
    return _rule_map(prompt)
