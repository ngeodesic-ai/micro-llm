# micro_lm/rails/stage11.py
from typing import Dict, Any
import numpy as np

def _simple_ema_med(x: np.ndarray, ema_beta=0.85, k=3):
    # lightweight fallback if TemporalDenoiser isn't compatible
    y = np.empty_like(x, dtype=float)
    acc = 0.0
    for i, v in enumerate(x.astype(float)):
        acc = ema_beta*acc + (1-ema_beta)*v if i else v
        y[i] = acc
    # median smooth
    if k > 1:
        pad = k//2
        xp = np.pad(y, (pad, pad), mode="edge")
        y = np.array([np.median(xp[i:i+k]) for i in range(len(y))])
    return y

def run_stage11(traces: Dict[str, np.ndarray], config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    # Try to use ngeodesic's denoiser, but gracefully degrade if the signature differs.
    denoised = {}
    try:
        from ngeodesic.core.denoise import TemporalDenoiser  # type: ignore
        import inspect
        sig = inspect.signature(TemporalDenoiser.__init__)
        kwargs = {}
        # Map our preferred knobs to whatever the class actually supports
        if "ema_decay" in sig.parameters:      kwargs["ema_decay"] = 0.85
        if "ema_beta" in sig.parameters:       kwargs["ema_beta"]  = 0.85  # alt name
        if "median_k" in sig.parameters:       kwargs["median_k"]  = 3
        if "conf_gate" in sig.parameters:      kwargs["conf_gate"] = 0.65
        if "gate" in sig.parameters:           kwargs["gate"]      = 0.65  # alt name
        if "noise_floor" in sig.parameters:    kwargs["noise_floor"] = 0.03
        den = TemporalDenoiser(**kwargs)
        for k, v in traces.items():
            denoised[k] = den.apply(v)
    except Exception:
        # Fallback: simple EMA + median; good enough for plumbing tests
        for k, v in traces.items():
            denoised[k] = _simple_ema_med(np.asarray(v, float), ema_beta=0.85, k=3)

    return {"report": {"denoised": denoised}}
