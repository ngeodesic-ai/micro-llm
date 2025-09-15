from typing import Dict, Any
import numpy as np

def _score_curve(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(x.max() * (x.sum() / (len(x) + 1e-9)))

def run_stage10(traces: Dict[str, np.ndarray], config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = config or {}
    pri = cfg.get("prior", {})  # dict[str,float] in [0,1]
    try:
        from ngeodesic.core.parser import geodesic_parse_report
        rep = geodesic_parse_report(traces)
        ordered = rep.get("ordered") or rep.get("sequence") or []
        if ordered:
            return {"report": rep, "ordered": ordered}
        raise RuntimeError("empty order")
    except Exception:
        alpha = 3.5    # ↑ prior influence (multiplier)
        gamma = 0.50   # + additive bias for very confident priors
        scored = []
        # get a scale for additive term roughly on par with base magnitudes
        base_vals = [ _score_curve(v) for v in traces.values() ]
        base_scale = (np.median(base_vals) or 1.0)
        for k, v in traces.items():
            base = _score_curve(v)
            p = float(pri.get(k, 0.0))
            score = base * (1.0 + alpha * p) + gamma * p * base_scale
            scored.append((k, score))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return {"report": {"note": "fallback ranking"}, "ordered": [k for k, _ in scored]}
