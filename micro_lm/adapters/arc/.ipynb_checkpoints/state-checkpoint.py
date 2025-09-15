from typing import Dict, Any

def normalize_arc_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    # expand later with validators
    return {
        "train_pairs": ctx.get("train_pairs", []),
        "test_inputs": ctx.get("test_inputs", []),
        "palette": ctx.get("palette", {"colors": list(range(10))}),
    }
