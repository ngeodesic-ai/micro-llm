# micro_llm/pipelines/runner.py

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


def run_micro(
    domain: str,
    prompt: str,
    context: Dict[str, Any],
    policy: Dict[str, Any],
    rails: str = "stage11",
    T: int = 180,
) -> Dict[str, Any]:
    """
    End-to-end micro runner:
      - build residual traces via the domain adapter
      - early-abstain if intent is non-executable
      - Stage-10 ranks primitives (optionally biased by adapter prior)
      - Stage-11 (optional) generates detection/denoise report
      - domain verifier enforces hard invariants (HF/LTV/oracle freshness, etc.)
    """
    adapter = _get_adapter(domain)
    inp = AdapterInput(prompt=prompt, context=context, policy=policy, T=T)
    bundle = make_residuals(adapter, inp)

    # Early abstain on non-exec — still surface prior/mapper_confidence for observability
    if bundle.flags.get("abstain_non_exec"):
        return {
            "domain": domain,
            "flags": bundle.flags,
            "rails": rails,
            "plan": {"sequence": []},
            "verify": {"ok": False, "reason": "abstain_non_exec"},
            "aux": {
                "features": bundle.aux.get("features", []),
                "prior": bundle.aux.get("prior"),
                "mapper_confidence": bundle.aux.get("mapper_confidence"),
                "note": "non-executable intent",
            },
            "report": {},
        }

    # Stage-10 ranking (optionally bias with adapter prior)
    prior = bundle.aux.get("prior", {})
    s10 = run_stage10(bundle.traces, config={"prior": prior})
    sequence = s10.get("ordered", [])

    # --- Shim: rails policy switches for Tier-1 sandbox ---
    rails_cfg = policy.get("rails", {}) if isinstance(policy, dict) else {}
    use_wdd = bool(rails_cfg.get("use_wdd", True))       # default True = Tier-2 preserves old behavior
    use_denoise = bool(rails_cfg.get("denoise", True))   # may be used by downstream rails
    
    # Stage-11 reporting only when requested AND WDD enabled
    if rails == "stage11" and use_wdd:
        s11 = run_stage11(bundle.traces)
    else:
        s11 = {"report": {}} 
    # --- Shim: rails policy switches for Tier-1 sandbox ---
    

    # If the adapter prior is decisive, clamp to top-1 for determinism
    if prior and max(prior.values()) >= 0.8:
        sequence = sequence[:1]

    plan = {"sequence": sequence}

    # Domain-specific verify
    if domain.lower() == "defi":
        vres = defi_verify(plan, {"policy": policy, **context})
    elif domain.lower() == "arc":
        vres = arc_verify(plan, {"policy": policy, **context})
    else:
        vres = {"ok": True, "reason": ""}

    return {
        "domain": domain,
        "flags": bundle.flags,
        "rails": rails,
        "plan": plan,
        "verify": vres,
        "aux": {
             "features": bundle.aux.get("features", []),
             "prior": bundle.aux.get("prior"),
             "mapper_confidence": bundle.aux.get("mapper_confidence"),
             "stage10": s10.get("report"),
         },
         "report": s11.get("report", {}),
     }
    
