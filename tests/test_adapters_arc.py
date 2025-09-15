from micro_lm.adapters.base import AdapterInput, make_residuals
from micro_lm.adapters.arc import ARCAdapter
def test_arc_adapter_shapes():
    inp = AdapterInput(prompt="flip", context={"train_pairs": [], "test_inputs": []}, policy={"max_ops":3}, T=120)
    b = make_residuals(ARCAdapter(), inp)
    assert set(b.traces.keys()) == {"flip_h","flip_v","rotate"}
    assert all(len(v)==120 for v in b.traces.values())
