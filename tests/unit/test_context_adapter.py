import pytest
from micro_lm.adapters.simple_context import SimpleContextAdapter

def test_normalize_minimum():
    ctx = {"oracle": {"age_sec": 5, "max_age_sec": 30},
           "account": {"balances": {"ETH": 2}},
           "market": {"venues": ["aave", "uniswap"]}}
    adapter = SimpleContextAdapter()
    out = adapter.normalize(ctx)
    assert out.oracle_age_sec == 5
    assert out.oracle_max_age_sec == 30
    assert out.account_balances["ETH"] == 2
    assert "aave" in out.venues
