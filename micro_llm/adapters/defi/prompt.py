import re
from typing import Dict, Any, Optional

AMOUNT = r"(?P<amount>\d+(?:\.\d+)?)"
ASSET  = r"(?P<asset>[A-Z]{3,6})"
PAIR   = r"(?P<base>[A-Z]{3,6})\/(?P<quote>[A-Z]{3,6})"

PATTERNS = [
    (rfr"(deposit|supply)\s+{AMOUNT}\s+{ASSET}", "deposit_asset"),
    (rfr"(withdraw|redeem)\s+{AMOUNT}\s+{ASSET}", "withdraw_asset"),
    (rfr"(borrow)\s+{AMOUNT}\s+{ASSET}", "borrow_asset"),
    (rfr"(repay|pay back)\s+{AMOUNT}\s+{ASSET}", "repay_loan"),
    (rfr"(swap|exchange)\s+{AMOUNT}\s+{ASSET}\s+for\s+{ASSET}", "swap_asset"),
    (rfr"(swap)\s+{PAIR}", "swap_asset_pair"),
]

def parse_prompt(prompt: str) -> Dict[str, Any]:
    p = prompt.lower().strip()
    for pat, prim in PATTERNS:
        m = re.search(pat, p, flags=re.I)
        if m:
            d = {k: v.upper() if isinstance(v, str) else v
                 for k, v in m.groupdict(default="").items()}
            return {"primitive": prim, **d}
    return {"primitive": "unknown"}
