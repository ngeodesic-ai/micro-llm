from micro_lm.adapters.base import AdapterInput, make_residuals
from micro_lm.adapters.arc.make_traces import ARCAdapter
from micro_lm.rails.stage10 import run_stage10

def test_flip_h_prior():
    inp = AdapterInput(prompt="flip the grid horizontally", context={}, policy={"mapper":{}}, T=128)
    b = make_residuals(ARCAdapter(), inp)
    ordered = run_stage10(b.traces, config={"prior": b.aux.get("prior", {})})["ordered"]
    assert ordered[0] == "flip_h"
