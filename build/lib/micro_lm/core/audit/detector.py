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
from typing import List, Optional, Tuple, Dict
import numpy as np
from .types import AuditResult, Peak, FamilySpec, Mode
from .utils import nxcorr, halfmax_window, circshift, make_rng

class Detector:
    def __init__(self, template_width: int = 64, null_shifts: int = 64):
        t = np.linspace(0, np.pi, int(template_width), dtype=np.float32)
        self.template = np.sin(t)
        self.null_shifts = int(null_shifts)
    def _null_floor(self, env_1d: np.ndarray, rng: np.random.Generator) -> float:
        T = len(env_1d)
        if T <= len(self.template):
            return 0.0
        mx = []
        for _ in range(self.null_shifts):
            s = int(rng.integers(1, T - 1))
            e = circshift(env_1d, s)
            mx.append(float(nxcorr(e, self.template).max()))
        return float(np.quantile(np.array(mx, dtype=np.float32), 0.99))
    def parse(self, env: np.ndarray, families: Optional[List[FamilySpec]] = None,
              mode: Mode = Mode.PURE, seed: int = 0) -> AuditResult:
        rng = make_rng(seed)
        T, K = env.shape
        corr_max = np.zeros(K, dtype=np.float32)
        tstars = np.zeros(K, dtype=np.int32)
        areas = np.zeros(K, dtype=np.float32)
        zfloor = 0.0
        windows: Dict[int, Tuple[int, int]] = {}
        peaks: List[Peak] = []
        for k in range(K):
            ez = env[:, k].astype(np.float32)
            zfloor = max(zfloor, self._null_floor(ez, rng))
            corr = nxcorr(ez, self.template)
            t0 = int(np.argmax(corr)) + len(self.template) // 2
            t0 = min(max(0, t0), T - 1)
            a, b = halfmax_window(ez, t0)
            areas[k] = float(ez[max(0, a):min(T, b+1)].sum())
            corr_max[k] = float(corr.max())
            tstars[k] = t0
        best = float(corr_max.max() + 1e-8)
        keep_mask = (corr_max >= 0.5 * best) & (corr_max >= max(zfloor, 0.2))
        if families and mode != Mode.PURE:
            rel = corr_max / (best + 1e-8)
            fam_keep = np.zeros_like(keep_mask)
            for fam in families:
                kidx = np.array(fam.idxs, dtype=int)
                if kidx.size == 0:                 # ⟵ guard: nothing in this family
                    continue
                fam_corr = corr_max[kidx]
                if fam_corr.size == 0:             # ⟵ (double guard; safe on weird inputs)
                    continue
                fam_best = float(fam_corr.max() + 1e-8)
                fam_mask = (fam_corr >= max(zfloor, fam.z_abs)) & (fam_corr >= fam.keep_frac * fam_best)
                fam_keep[kidx] = fam_mask
            keep_mask &= fam_keep
        keep = np.where(keep_mask)[0].tolist()
        order = sorted(keep, key=lambda k: int(tstars[k]))
        for k in range(K):
            peaks.append(Peak(k=int(k), t_star=int(tstars[k]), corr_max=float(corr_max[k]), area=float(areas[k]), z_abs=float(zfloor)))
            windows[k] = halfmax_window(env[:, k], int(tstars[k]))
        return AuditResult(keep=keep, order=order, peaks=peaks, windows=windows, zfloor=float(zfloor), seed=int(seed),
                           debug={"corr_max": corr_max, "tstars": tstars, "areas": areas})