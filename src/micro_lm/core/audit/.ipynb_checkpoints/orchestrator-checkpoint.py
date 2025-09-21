from __future__ import annotations
import numpy as np
from .types import AuditRequest, AuditResult, Mode
from .traces import synth_traces
from .detector import Detector

def run_wdd(req: AuditRequest) -> AuditResult:
    assert req.emb.ndim == 1
    P, A = req.prototypes, req.anchors
    assert P.shape == A.shape and P.ndim == 2
    guided = req.prob_trace if (req.mode == Mode.GUIDED and req.prob_trace is not None) else None
    _, env = synth_traces(req.emb, P, A, T=req.T, guided_env=guided)
    det = Detector(template_width=64, null_shifts=64)
    res = det.parse(env, families=req.families, mode=req.mode, seed=req.seed)
    res.debug.update({"env": env, "prototypes": P, "anchors": A})
    return res