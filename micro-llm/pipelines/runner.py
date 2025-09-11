from typing import Dict, Any
from micro_llm.adapters.base import AdapterInput, make_residuals
from micro_llm.adapters.arc import ARCAdapter
from micro_llm.adapters.defi import DeFiAdapter
from micro_llm.rails import run_stage10, run_stage11
from micro_llm.verify import arc_verify, defi_verify

def _get_adapter(domain: str):
    return {"arc": ARCAdapter, "defi": DeFiAdapter}[domain.lower()]()

def _get_verifier(domain: str):
    return {"arc": arc_verify, "defi": defi_verify}[domain.lower()]

def run_micro(domain: str, prompt: str, context: Dict[str, Any], policy: Dict[str, Any], rails: str = "stage11", T: int = 180) -> Dict[str, Any]:
    adapter = _get_adapter(domain)
    inp = AdapterInput(prompt=prompt, context=context, policy=policy, T=T)
    bundle = make_residuals(adapter, inp)

    if rails.lower() == "stage10":
        out = run_stage10(bundle.traces)
    else:
        out = run_stage11(bundle.traces)

    # Minimal placeholder “plan” (replace with parsed order when Stage-10 report is wired)
    plan = {"sequence": list(bundle.traces.keys())[:1]}  # stub

    verifier = _get_verifier(domain)
    vres = verifier(plan, context)

    return {
        "domain": domain,
        "flags": bundle.flags,
        "rails": rails,
        "plan": plan,
        "verify": vres,
        "aux": bundle.aux,
        "report": out.get("report", {}),
    }
