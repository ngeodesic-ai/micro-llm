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
        try:
            self.model = load(cfg.model_path)
        except ModuleNotFoundError as e:
            if e.name == "encoders":
                import importlib, sys
                alias = importlib.import_module("micro_lm.encoders")
                sys.modules["encoders"] = alias
                self.model = load(cfg.model_path)
            else:
                raise
        
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
