# milestones/encoders.py
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
