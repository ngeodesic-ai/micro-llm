from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

class Mode(str, Enum):
    PURE = "pure"
    FAMILY = "family"
    GUIDED = "guided"

@dataclass
class FamilySpec:
    name: str
    idxs: List[int]
    z_abs: float = 0.6
    keep_frac: float = 0.75
    template_width: int = 64

@dataclass
class AuditRequest:
    emb: np.ndarray              # (D,)
    prototypes: np.ndarray       # (K,D) unit directions
    anchors: np.ndarray          # (K,D) anchor centers
    T: int = 600
    seed: int = 0
    prob_trace: Optional[np.ndarray] = None  # (T,K) optional
    families: Optional[List[FamilySpec]] = None
    mode: Mode = Mode.PURE

@dataclass
class Peak:
    k: int
    t_star: int
    corr_max: float
    area: float
    z_abs: float

@dataclass
class AuditResult:
    keep: List[int] = field(default_factory=list)
    order: List[int] = field(default_factory=list)
    peaks: List[Peak] = field(default_factory=list)
    windows: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    zfloor: float = 0.0
    seed: int = 0
    debug: Dict[str, Any] = field(default_factory=dict)