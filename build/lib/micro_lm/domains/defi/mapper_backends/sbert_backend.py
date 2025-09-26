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
