from micro_lm.core.backends.sbert import SBertMapper

def test_sbert_mapper_fallback_works_without_model():
    m = SBertMapper(domain="defi", policy={"mapper":{"model_path":".does_not_exist.joblib"}})
    label, score, aux = m.map_prompt("swap 2 eth to usdc")
    assert label in {"swap_assets","abstain"}  # depends on heuristic path
    assert "reason" in aux
