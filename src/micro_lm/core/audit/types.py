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