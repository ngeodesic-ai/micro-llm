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
import numpy as np
from .types import AuditRequest, AuditResult, Mode
from .traces import synth_traces
from .detector import Detector

def run_wdd(req: AuditRequest) -> AuditResult:
    assert req.emb.ndim == 1
    P, A = req.prototypes, req.anchors
    assert P.shape == A.shape and P.ndim == 2
    guided = req.prob_trace if (req.mode == Mode.GUIDED and req.prob_trace is not None) else None
    _, env = synth_traces(req.emb, P, A, T=req.T, guided_env=guided)
    det = Detector(template_width=64, null_shifts=64)
    res = det.parse(env, families=req.families, mode=req.mode, seed=req.seed)
    res.debug.update({"env": env, "prototypes": P, "anchors": A})
    return res