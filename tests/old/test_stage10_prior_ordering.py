from micro_lm.adapters.base import AdapterInput, make_residuals
from micro_lm.adapters.defi import DeFiAdapter
from micro_lm.rails.stage10 import run_stage10

def test_prior_pushes_deposit_first():
    inp = AdapterInput(
        prompt="deposit 10 ETH into aave",
        context={"risk":{"hf":1.22},"oracle":{"age_sec":5,"max_age_sec":30}},
        policy={"ltv_max":0.75}, T=180
    )
    bundle = make_residuals(DeFiAdapter(), inp)
    ordered = run_stage10(bundle.traces, config={"prior": bundle.aux["prior"], "prior_alpha":3.5, "prior_gamma":0.5})["ordered"]
    assert ordered[0] == "deposit_asset"