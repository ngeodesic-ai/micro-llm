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
from typing import Iterable, Optional
import numpy as np

class SbertEncoder:
    def __init__(self, model_name: Optional[str] = None, encode_fn=None):
        self.model_name = model_name
        self._model = None
        self._encode_fn = encode_fn
    def _lazy(self):
        if self._encode_fn is not None:
            return
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise RuntimeError("Install sentence-transformers or pass encode_fn") from e
            self._model = SentenceTransformer(self.model_name or "sentence-transformers/all-MiniLM-L6-v2")
            self._encode_fn = self._model.encode
    def encode(self, texts: Iterable[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        self._lazy()
        vecs = self._encode_fn(list(texts), batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=False)
        vecs = np.asarray(vecs, dtype=np.float32)
        if normalize:
            n = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / n
        return vecs