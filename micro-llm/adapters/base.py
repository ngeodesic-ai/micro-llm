from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Protocol
import numpy as np

@dataclass
class AdapterInput:
    prompt: str
    context: Dict[str, Any]
    policy: Dict[str, Any]
    T: int = 180

@dataclass
class ResidualBundle:
    traces: Dict[str, np.ndarray]
    aux: Dict[str, Any]
    flags: Dict[str, bool]

class DomainAdapter(Protocol):
    def build_state(self, inp: AdapterInput) -> Dict[str, Any]: ...
    def build_features(self, state: Dict[str, Any], T: int) -> Dict[str, np.ndarray]: ...
    def residuals_from_features(self, feats: Dict[str, np.ndarray], T: int) -> Dict[str, np.ndarray]: ...
    def early_safety_flags(self, state: Dict[str, Any], feats: Dict[str, np.ndarray]) -> Dict[str, bool]: ...

def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float); return (x - x.mean()) / (x.std() + 1e-8)

def _smooth_ma(x: np.ndarray, k: int = 7) -> np.ndarray:
    if k <= 1: return x.copy()
    pad = k // 2; xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")

def _exclusive(curves: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    ks = list(curves.keys())
    X = np.stack([_z(curves[k]) for k in ks], axis=0)
    cm = X.mean(axis=0, keepdims=True)
    R = np.clip(X - cm, 0, None)
    return {k: _smooth_ma(R[i], 7) for i, k in enumerate(ks)}

def make_residuals(adapter: DomainAdapter, inp: AdapterInput) -> ResidualBundle:
    state = adapter.build_state(inp)
    feats = adapter.build_features(state, inp.T)
    base  = adapter.residuals_from_features(feats, inp.T)
    base  = {k: np.maximum(0.0, np.asarray(v, float)[:inp.T]) for k, v in base.items()}
    traces = _exclusive(base)
    flags = adapter.early_safety_flags(state, feats) or {}
    return ResidualBundle(traces=traces, aux={"features": list(feats.keys())}, flags=flags)
