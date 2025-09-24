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

from typing import Protocol, List, Tuple, Optional, Dict, Any

class IntentModel(Protocol):
    # prefer predict_proba, fall back to decision_function
    def predict_proba(self, X: List[str]) -> List[List[float]]: ...
    @property
    def classes_(self) -> List[str]: ...

class IntentPrediction(TypedDict):
    intent: str
    score: float
    topk: List[Tuple[str, float]]

class IntentShim:
    def __init__(self, model: Optional[IntentModel], *, topk: int = 5, debug: bool = False):
        self.model = model
        self.topk = topk
        self.debug = debug

    def infer(self, text: str) -> Optional[IntentPrediction]:
        if not self.model:
            return None
        # predict
        probs = self.model.predict_proba([text])[0]
        pairs = list(zip(self.model.classes_, map(float, probs)))
        pairs.sort(key=lambda x: x[1], reverse=True)
        if self.debug:
            print("[mapper.shim] top:", pairs[:self.topk])
        top = pairs[0]
        return {"intent": top[0], "score": top[1], "topk": pairs[:self.topk]}
