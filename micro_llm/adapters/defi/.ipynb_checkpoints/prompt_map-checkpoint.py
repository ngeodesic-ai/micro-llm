import re
from typing import Dict, Any

AMOUNT = r"(?P<amount>\d+(?:\.\d+)?)"
ASSET  = r"(?P<asset>[A-Z]{3,6})"
PAIR   = r"(?P<base>[A-Z]{3,6})\/(?P<quote>[A-Z]{3,6})"

PATTERNS = [
    (rf"(deposit|supply)\s+{AMOUNT}\s+{ASSET}", "deposit_asset"),
    (rf"(withdraw|redeem)\s+{AMOUNT}\s+{ASSET}", "withdraw_asset"),
    (rf"(borrow)\s+{AMOUNT}\s+{ASSET}", "borrow_asset"),
    (rf"(repay|pay back)\s+{AMOUNT}\s+{ASSET}", "repay_loan"),
    (rf"(swap|exchange)\s+{AMOUNT}\s+{ASSET}\s+for\s+{ASSET}", "swap_asset"),
    (rf"(swap)\s+{PAIR}", "swap_asset_pair"),
]


# micro_llm/adapters/defi/prompt_map.py
NON_EXEC_PATTERNS = [
    r"(check|show|what('| i)?s)\s+(my\s+)?(balance|health\s*factor|hf)\b",
    r"(how\s+much)\s+(can\s+i\s+borrow|ltv)\b",
    r"(price|oracle)\s+(age|status)\b",
]

def parse_prompt(prompt: str) -> Dict[str, Any]:
    p = prompt.strip()
    # non-exec first
    for pat in NON_EXEC_PATTERNS:
        if re.search(pat, p, flags=re.I):
            return {"primitive": "non_exec", "query": p}

    for pat, prim in PATTERNS:
        m = re.search(pat, p, flags=re.I)
        if m:
            d = {k: (v.upper() if isinstance(v, str) else v)
                 for k, v in m.groupdict(default="").items()}
            return {"primitive": prim, **d}
    return {"primitive": "unknown"}

