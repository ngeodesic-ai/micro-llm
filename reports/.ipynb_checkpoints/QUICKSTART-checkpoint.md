# micro-lm Quickstart

This guide shows how to run **mapper + audit** using the refactored
`micro-lm` codebase.

------------------------------------------------------------------------

## 🚀 Quickstart (Jupyter Notebook)

``` python
from micro_lm.core.runner import run_micro

# Example prompt
prompt = "deposit 10 ETH into aave"

# Minimal policy & context
policy = {
    "mapper": {
        "model_path": ".artifacts/defi_mapper.joblib",
        "confidence_threshold": 0.5,
    }
}
context = {}

# Run through micro-lm pipeline
out = run_micro(
    domain="defi",
    prompt=prompt,
    context=context,
    policy=policy,
    rails="stage11",
    T=180,
    backend="wordmap",   # or "sbert"
)

print(out)
```

------------------------------------------------------------------------

## ✅ Example Output

``` python
{
  'ok': True,
  'label': 'deposit_asset',
  'score': 0.71,
  'reason': 'shim:accept:stage-4',
  'artifacts': {
    'mapper': {
      'score': 0.71,
      'reason': 'heuristic:deposit',
      'aux': {'reason': 'heuristic:deposit'}
    },
    'verify': {
      'ok': True,
      'reason': 'shim:accept:stage-4'
    },
    'schema': {
      'v': 1,
      'keys': ['mapper', 'verify']
    }
  }
}
```

------------------------------------------------------------------------

## 🔎 Output Breakdown

### Top-level fields

-   **`ok: True`** → Overall run succeeded, action allowed.\
-   **`label: 'deposit_asset'`** → Canonical intent chosen.\
-   **`score: 0.71`** → Mapper's confidence.\
-   **`reason: 'shim:accept:stage-4'`** → Accepted by Stage-4 rails
    shim.

### Artifacts

-   **`mapper`**
    -   Raw mapper result.\
    -   Score + heuristic reason.
-   **`verify`**
    -   Rails/audit check result.\
    -   `ok=True` → passed safety/policy.
-   **`schema`**
    -   Metadata about which artifact keys exist.

------------------------------------------------------------------------

## 🧩 Interpretation

This tells us:\
1. Prompt looked like a **deposit**.\
2. Mapper classified with \~71% confidence.\
3. Audit/rails verifier confirmed no violations.\
4. Final decision → **allow**, with `deposit_asset` as the action.
