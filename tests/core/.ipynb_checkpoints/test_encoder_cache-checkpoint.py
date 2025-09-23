import time
import numpy as np
from micro_lm.core.encoder import encode_cached

def test_encode_cached_speedup():
    t0 = time.perf_counter()
    a = encode_cached("hello world")  # cold
    t1 = time.perf_counter()
    b = encode_cached("hello world")  # hot
    t2 = time.perf_counter()
    assert isinstance(a, np.ndarray) and a.shape == b.shape
    cold_ms = (t1 - t0) * 1000
    hot_ms = (t2 - t1) * 1000
    # require at least 5x faster hot path (very lax)
    assert hot_ms * 5 < max(cold_ms, 1e-3)
