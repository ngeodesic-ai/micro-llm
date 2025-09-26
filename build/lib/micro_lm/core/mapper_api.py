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
from typing import Any, Dict, Tuple
from .backends import sbert, wordmap


class MapperAPI:
    def __init__(self, *, backend: str, domain: str, policy: Dict[str, Any]):
        self.backend = backend
        self.domain = domain
        self.policy = policy
        self.debug = {}

        thr = policy.get("mapper", {}).get("confidence_threshold", 0.5)
        self.threshold = float(thr)

        if backend == "sbert":
            try:
                self.impl = sbert.SBertMapper(domain=domain, policy=policy)
            except Exception as e:
                # degrade to wordmap
                self.impl = wordmap.WordMapMapper(domain=domain, policy=policy)
                self.backend = "wordmap"
                self.debug.update({"degraded": True, "reason": str(e)})
        elif backend == "wordmap":
            self.impl = wordmap.WordMapMapper(domain=domain, policy=policy)
        else:
            raise ValueError(f"Unknown backend: {backend}")

