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

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class SBERTEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 batch_size=64, normalize=True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    def fit(self, X, y=None):
        # lazy import to avoid heavy import at module load
        from sentence_transformers import SentenceTransformer
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        embs = self._model.encode(
            list(X),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(embs)
