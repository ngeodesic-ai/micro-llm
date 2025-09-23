# add near top-level (or extend existing utils)
import time
from contextlib import contextmanager

@contextmanager
def timed(name: str, sink: list):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        sink.append({"phase": name, "ms": dt_ms})
