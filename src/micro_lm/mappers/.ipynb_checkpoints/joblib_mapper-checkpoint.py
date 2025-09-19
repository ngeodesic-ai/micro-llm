from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from joblib import load
import numpy as np
from micro_lm.core.interfaces.intent_mapper import IntentMapper, IntentResult

@dataclass
class JoblibMapperConfig:
    model_path: str
    confidence_threshold: float = 0.7

class JoblibMapper(IntentMapper):
    def __init__(self, cfg: JoblibMapperConfig):
        self.cfg = cfg
        self.model = load(cfg.model_path)
        self.classes_: List[str] = list(getattr(self.model, "classes_", []))

    def _probs(self, text: str) -> List[Tuple[str, float]]:
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba([text])[0]
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function([text])[0]
            ex = np.exp(scores - scores.max())
            probs = ex / ex.sum()
        else:
            raise RuntimeError("Model lacks predict_proba/decision_function")
        pairs = list(zip(self.classes_, probs))
        pairs.sort(key=lambda p: float(p[1]), reverse=True)
        return [(c, float(p)) for c, p in pairs]

    def infer(self, text: str, *, context: Dict[str, Any] | None = None) -> IntentResult:
        topk = self._probs(text)
        intent, score = (topk[0] if topk else (None, 0.0))
        reason = "joblib:predict_proba"
        if score < self.cfg.confidence_threshold:
            return IntentResult(intent=None, score=score, topk=topk, reason="low_confidence")
        return IntentResult(intent=intent, score=score, topk=topk, reason=reason)
