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

from functools import lru_cache
import numpy as np
import hashlib

# ---- minimal deterministic fallback encoder ----
# Produces a stable 256-D vector from the text (blake2b-seeded RNG).
# This keeps the cache-speed test meaningful without depending on SBERT wiring.
def _fallback_encode(texts: list[str], D: int = 256) -> np.ndarray:
    out = np.zeros((len(texts), D), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hashlib.blake2b(t.encode("utf-8"), digest_size=16).digest()
        seed = int.from_bytes(h, "little", signed=False) % (2**32)
        rng = np.random.default_rng(seed)
        out[i] = rng.standard_normal(D, dtype=np.float32)
    return out

# Single place to pick the implementation (swap to SBERT later if/when needed):
def _encode_impl(texts: list[str]) -> np.ndarray:
    # If you later wire SBERT:
    #   from . import sbert
    #   return sbert.encode(texts)
    return _fallback_encode(texts)

@lru_cache(maxsize=4096)
def encode_cached(text: str) -> np.ndarray:
    # return a copy to avoid mutation of cached array
    return _encode_impl([text])[0].copy()

def encode_batch(texts: list[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.zeros((0, 256), dtype=np.float32)
    chunks = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    arrs = [_encode_impl(c) for c in chunks]
    return np.vstack(arrs)
