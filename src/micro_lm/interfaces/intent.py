# interfaces/intent.py
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
