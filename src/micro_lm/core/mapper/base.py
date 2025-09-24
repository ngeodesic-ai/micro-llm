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

# Mapper backend interface (Stage 7) — patched
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any, Protocol

class MapperBackend(Protocol):
    name: str
    def predict(self, prompts: List[str]) -> List[Tuple[Optional[str], float]]: ...

def load_backend(kind: str, **kwargs) -> MapperBackend:
    kind = (kind or "wordmap").lower()
    print('[mapper] backend=', kind, 'kwargs=', kwargs)
    if kind == "wordmap":
        from micro_lm.domains.defi.mapper_backends.wordmap_backend import WordmapBackend
        return WordmapBackend(**kwargs)
    if kind == "sbert":
        from micro_lm.domains.defi.mapper_backends.sbert_backend import SbertBackend
        return SbertBackend(**kwargs)
    raise ValueError(f"Unknown mapper backend: {kind}")
