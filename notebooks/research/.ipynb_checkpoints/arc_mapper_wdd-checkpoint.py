# ---------------------- CELL 1 ---------------------- #

# --- Robust notebook shim for legacy joblib artifacts expecting `encoders.*` ---
import sys, types, numpy as np

# Create/replace a lightweight 'encoders' module in sys.modules
enc_mod = types.ModuleType("encoders")

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None
    print("NOTE: sentence-transformers not available:", e)

class _SBERTBase:
    """
    Compat shim implementing the sklearn Transformer API expected by saved Pipelines.
    Handles pickles that don't call __init__ and are missing attributes.
    Provides both class names: SBERTEncoder and SBERTFeaturizer.
    """
    # NOTE: __init__ might not be called during unpickle; use _ensure_attrs() everywhere.
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        self.model_name = model
        self._enc = None
        self._kwargs = kwargs

    def _ensure_attrs(self):
        # Add any attributes that might be missing from legacy pickles
        if not hasattr(self, "model_name") or self.model_name is None:
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        if not hasattr(self, "_enc"):
            self._enc = None
        if not hasattr(self, "_kwargs"):
            self._kwargs = {}

    def _ensure_encoder(self):
        self._ensure_attrs()
        if self._enc is None:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers not installed in this kernel; "
                    "pip install sentence-transformers && restart kernel"
                )
            self._enc = SentenceTransformer(self.model_name)

    # sklearn API
    def fit(self, X, y=None):
        self._ensure_attrs()
        return self

    def transform(self, X):
        self._ensure_encoder()
        return np.asarray(self._enc.encode(list(X), show_progress_bar=False))

    # some older code may call .encode directly; alias it
    def encode(self, X):
        return self.transform(X)

# Expose both legacy names on the encoders module
class SBERTEncoder(_SBERTBase): ...
class SBERTFeaturizer(_SBERTBase): ...

enc_mod.SBERTEncoder = SBERTEncoder
enc_mod.SBERTFeaturizer = SBERTFeaturizer
sys.modules["encoders"] = enc_mod

# Make sure your package code is importable too (if needed)
import pathlib
if str(pathlib.Path("src").resolve()) not in sys.path:
    sys.path.append(str(pathlib.Path("src").resolve()))
print("encoders shim ready (SBERTEncoder + SBERTFeaturizer) and sys.path configured")

# ---------------------- CELL 2 ---------------------- #
import joblib
from pathlib import Path

def load_mapper():
    for name in [".artifacts/arc_mapper.joblib"]:
        p = Path(name).resolve()
        if p.exists():
            print("Loading:", p.as_posix())
            return joblib.load(p.as_posix())
    raise FileNotFoundError("No mapper artifact found in .artifacts/")

pipe = load_mapper()
print(pipe)