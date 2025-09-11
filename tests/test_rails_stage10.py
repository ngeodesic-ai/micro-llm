import numpy as np
from micro_llm.rails.stage10 import run_stage10
def test_stage10_shapes():
    traces = {"a": np.ones(160), "b": np.linspace(0,1,160)}
    out = run_stage10(traces)
    assert "report" in out
