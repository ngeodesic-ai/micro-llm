# interfaces/runner.py
from typing import Protocol, Dict, Any, Optional, TypedDict

class RunnerOut(TypedDict, total=False):
    plan: Dict[str, Any]
    verify: Dict[str, Any]
    flags: Dict[str, Any]
    aux: Dict[str, Any]
    label: Optional[str]   # if your pipeline sets it
    score: Optional[float]
    reason: Optional[str]

class RunMicro(Protocol):
    def __call__(
        self,
        domain: str,
        prompt: str,
        *,
        context: Dict[str, Any],
        policy: Dict[str, Any],
        rails: str,
        T: int
    ) -> RunnerOut: ...

# tiny adapter, so your code can be typed but still import dynamically
def bind_run_micro(run_micro_obj) -> RunMicro:
    def _runner(domain, prompt, *, context, policy, rails, T):
        return run_micro_obj(domain, prompt, context=context, policy=policy, rails=rails, T=T)
    return _runner
