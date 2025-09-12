from typing import Dict, Any
from micro_llm.adapters.base import AdapterInput, make_residuals
from micro_llm.adapters.arc import ARCAdapter
from micro_llm.adapters.defi import DeFiAdapter
from micro_llm.rails import run_stage10, run_stage11
from micro_llm.verify import arc_verify
from micro_llm.verify.defi_verify import defi_verify

def _get_adapter(domain: str):
    return {"arc": ARCAdapter, "defi": DeFiAdapter}[domain.lower()]()

def _get_verifier(domain: str):
    return {"arc": arc_verify, "defi": defi_verify}[domain.lower()]

def run_micro(domain: str, prompt: str, context: Dict[str, Any], policy: Dict[str, Any], rails: str = "stage11", T: int = 180) -> Dict[str, Any]:
    adapter = _get_adapter(domain)
    inp = AdapterInput(prompt=prompt, context=context, policy=policy, T=T)
    bundle = make_residuals(adapter, inp)
    
    # Early abstain on non-exec
    if bundle.flags.get("abstain_non_exec"):
        return {
            "domain": domain,
            "flags": bundle.flags,
            "rails": rails,
            "plan": {"sequence": []},
            "verify": {"ok": False, "reason": "abstain_non_exec"},
            "aux": {"features": bundle.aux.get("features", []), "note": "non-executable intent"},
            "report": {}
        }
    
    prior = bundle.aux.get("prior", {})                     # NEW
    s10 = run_stage10(bundle.traces, config={"prior": prior})  # NEW
    s11 = run_stage11(bundle.traces)
    
    sequence = s10.get("ordered", [])
    # optional: clamp to top-1 when prior was decisive
    if max(prior.values(), default=0.0) >= 0.8:
        sequence = sequence[:1]
    
    plan = {"sequence": sequence}

    if domain.lower() == "defi":
        vres = defi_verify(plan, {"policy": policy, **context})
    else:
        # existing ARC or other-domain verifier
        from micro_llm.verify.arc_verify import arc_verify  # if you have one, else simple OK
        vres = arc_verify(plan, {"policy": policy, **context}) if 'arc' in domain.lower() else {"ok": True, "reason": ""}

    return {
        "domain": domain,
        "flags": bundle.flags,
        "rails": rails,
        "plan": plan,
        "verify": vres,
        "aux": {"features": bundle.aux.get("features", []), "stage10": s10.get("report")},
        "report": s11.get("report", {}),
    }
