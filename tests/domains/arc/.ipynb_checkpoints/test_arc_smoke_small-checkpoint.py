# tests/domains/arc/test_arc_smoke_small.py
import json
from pathlib import Path
import pytest
from micro_lm.core.runner import run_micro

def _load_jsonl(p: Path):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "fixtures" / "arc" / "cases_small.jsonl"

@pytest.mark.parametrize("case", _load_jsonl(CASES))
def test_arc_smoke_case(case):
    out = run_micro(
        "arc",
        case.get("prompt", ""),
        context=case.get("context", {}),
        policy={
            "audit": {"backend": "wdd"},   # ← ensure we use WDD, not Tier-1
            "arc": {"solve": False},
        },
        rails="stage11",
        T=128,
        backend="wordmap",
    )
    assert isinstance(out, dict)
    assert out is not None
