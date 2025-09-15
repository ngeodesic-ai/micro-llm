from typing import Dict, Any, List

def summarize_metrics(events: List[Dict[str, Any]]) -> Dict[str, float]:
    # Expect events with fields: {"label":..., "pred":..., "abstain":bool, "violation":bool}
    n = len(events) or 1
    acc = sum(e.get("label")==e.get("pred") and not e.get("abstain", False) for e in events)/n
    abstain = sum(e.get("abstain", False) for e in events)/n
    hall = sum((e.get("pred") is not None) and (e.get("label")!=e.get("pred")) for e in events)/n
    omit = sum((e.get("label") is not None) and (e.get("pred") is None) for e in events)/n
    return {"accuracy": acc, "abstain_rate": abstain, "hallucination_rate": hall, "omission_rate": omit}
