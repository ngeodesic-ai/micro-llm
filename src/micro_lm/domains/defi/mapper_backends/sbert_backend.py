# Tier-1 SBERT-backed mapper wrapper (Stage 7)
from __future__ import annotations
from typing import List, Tuple, Optional
import os, joblib

class SbertBackend:
    name = "sbert"
    def __init__(self, model_path: str = ".artifacts/defi_mapper.joblib", confidence_threshold: float = 0.7) -> None:
        self.th = float(confidence_threshold)
        self.model_path = model_path
        self.model = None
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
            except Exception:
                self.model = None  # degrade gracefully

    def predict(self, prompts: List[str]) -> List[Tuple[Optional[str], float]]:
        # Expect a joblib with .predict_proba or a (label,conf) list callable.
        out: List[Tuple[Optional[str], float]] = []
        if self.model is None:
            # Graceful fallback: abstain with low confidence
            return [(None, 0.0) for _ in prompts]
        try:
            if hasattr(self.model, "predict_proba") and hasattr(self.model, "classes_"):
                import numpy as np
                P = self.model.predict_proba(prompts)  # type: ignore
                cls = self.model.classes_
                for i in range(len(prompts)):
                    j = int(np.argmax(P[i]))
                    conf = float(P[i][j])
                    label = str(cls[j]) if conf >= self.th else None
                    out.append((label, conf))
                return out
            # else: assume callable returning (label, conf)
            for p in prompts:
                label, conf = self.model(p)  # type: ignore
                out.append((label if conf>=self.th else None, float(conf)))
            return out
        except Exception:
            return [(None, 0.0) for _ in prompts]
