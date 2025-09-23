import json
from pathlib import Path
import pytest

from micro_lm.core.runner import run_micro

def _load_jsonl(p: Path):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "fixtures" / "defi" / "cases_small.jsonl"

@pytest.mark.parametrize("case", _load_jsonl(CASES))
def test_defi_smoke_case(case):
    # Minimal policy; adjust if you want to force specific audit/backend
    out = run_micro(
        "defi",
        case.get("prompt", ""),
        context=case.get("context", {}),
        policy={},            # keep empty to honor repo defaults
        rails="stage11",
        T=128,
        backend="wordmap",    # mapper backend (wordmap|sbert)
    )
    # Schema-agnostic smoke: it ran and returned a dict-like result
    assert isinstance(out, dict)
    # Optional: ensure it isn't some totally empty sentinel
    assert out is not None
