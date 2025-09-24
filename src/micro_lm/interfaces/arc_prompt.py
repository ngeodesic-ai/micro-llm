# src/micro_lm/interfaces/arc_prompt.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Sequence
import json, hashlib
import joblib, os
import numpy as np

# ARC Tier-2 harness (family/detector)
from micro_lm.domains.arc.wdd_arc_harness import run_arc_wdd

# Defaults (kept to mirror DeFi prompt interface)
DEFAULT_RAILS = "stage11"
DEFAULT_T = 180
DEFAULT_POLICY: Dict[str, Any] = {}    # ARC doesn’t need mapper defaults here
DEFAULT_CONTEXT: Dict[str, Any] = {}   # Not used by ARC today

ARC_MAPPER_PATH = os.getenv("ARC_MAPPER_PATH", ".artifacts/arc_mapper.joblib")


# -------------------------- primitive canon + utils ---------------------------

def _canon_primitive(name: str) -> str:
    """Canonicalize ARC primitive names (match DeFi’s _canon_primitive style)."""
    t = (name or "").strip().lower()
    table = {
        "flip_h": "flip_h",
        "flip-horizontal": "flip_h",
        "flip_horizontal": "flip_h",
        "mirror_lr": "flip_h",
        "flip_v": "flip_v",
        "flip-vertical": "flip_v",
        "flip_vertical": "flip_v",
        "mirror_tb": "flip_v",
        "rot90": "rotate",
        "rotate90": "rotate",
        "rotate": "rotate",
    }
    return table.get(t, t)



# --- put near the top of src/micro_lm/interfaces/arc_prompt.py ---
import sys, os, joblib  # ensure these are imported

def _ensure_sbert_in_main():
    """Expose SBERTEncoder on __main__ for notebook-pickled artifacts."""
    m = sys.modules.get("__main__")
    if m is None or hasattr(m, "SBERTEncoder"):
        return

    class SBERTEncoder:  # noqa: N801
        def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                     normalize=True, **kwargs):
            # When unpickled, __init__ is NOT called; keep for fresh cases only.
            self._model_name = model_name
            self.normalize = bool(normalize)

        def _ensure_model(self):
            # On unpickle, there may be no .model; hydrate if needed.
            model = getattr(self, "model", None) or getattr(self, "_model", None)
            if model is None:
                from sentence_transformers import SentenceTransformer
                name = getattr(self, "_model_name", "sentence-transformers/all-MiniLM-L6-v2")
                model = SentenceTransformer(name)
                try: self.model = model
                except Exception: pass
                try: self._model = model
                except Exception: pass
            return model

        def transform(self, texts):
            import numpy as np
            model = self._ensure_model()
            normalize = getattr(self, "normalize", True)
            emb = model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            return np.asarray(emb)

    setattr(m, "SBERTEncoder", SBERTEncoder)
# -------------------------------------------------------------------

ARC_MAPPER_PATH = os.getenv("ARC_MAPPER_PATH", ".artifacts/arc_mapper.joblib")
_arc_mapper = None

def _get_arc_mapper():
    global _arc_mapper
    if _arc_mapper is None:
        _ensure_sbert_in_main()                 # <-- IMPORTANT
        _arc_mapper = joblib.load(ARC_MAPPER_PATH)
    return _arc_mapper


def _seq_hash(seq: Sequence[str] | None) -> str:
    s = "|".join(seq) if seq else "∅"
    return hashlib.sha256(s.encode()).hexdigest()[:12]

def _is_abstain(minimal: dict, res: dict) -> bool:
    v = (minimal.get("verify") or {})
    reason = (v.get("reason") or "").lower()
    seq = (minimal.get("plan") or {}).get("sequence") or []
    flags = res.get("flags") or {}
    abstain_token = ("abstain" in reason) or ("low_conf" in reason)
    # ARC also surfaces decisions under aux.stage11.wdd.arc
    arc_dec = (((res.get("aux") or {}).get("stage11") or {})
               .get("wdd") or {}).get("arc", {}).get("mode")
    # If no sequence and no explicit OK reason, treat as abstain
    return (not seq) or abstain_token or (str(arc_dec).upper() == "ABSTAIN")

def _summarize_wdd(aux: dict) -> dict:
    """Build a DeFi-like WDD table summary from ARC aux."""
    # ARC nests under aux.stage11.wdd.arc.{mode,results}
    st11 = (aux.get("stage11") or {})
    w = (st11.get("wdd") or {})
    arc = (w.get("arc") or {})
    results = arc.get("results") or {}
    # Decision is derived from keep/order in the harness’ wdd_summary; fall back to “PASS” if any ok
    keep = [k for k, v in results.items() if (v or {}).get("ok")]
    decision = "PASS" if keep else "ABSTAIN"

    # Try to collect per-primitive sigma/proto/which_prior if present
    sigma = {}
    proto_w = {}
    which_prior = {}
    for k, v in results.items():
        info = (v or {}).get("info") or {}
        if "sigma" in info: sigma[k] = info["sigma"]
        if "proto_w" in info: proto_w[k] = info["proto_w"]
        wp = info.get("which_prior") or v.get("which") or k
        which_prior[k] = wp

    # Prefer the harness’ own summary when available
    summary = (st11.get("wdd_summary") or {})  # optional: if you stash it there
    note = summary.get("note") if isinstance(summary, dict) else None

    return {
        "decision": decision,
        "keep": keep,
        "sigma": (sigma or None),
        "proto_w": (proto_w or None),
        "which_prior": (which_prior or None),
        "note": note,
    }

def _lexical_primitives(prompt: str) -> List[str] | None:
    """Tiny lexical shim (off unless you want it). Kept for parity with DeFi."""
    p = (prompt or "").lower()
    muts: List[str] = []
    if "rotate" in p:
        muts.append("rotate")
    if "flip" in p and ("horizontal" in p or "left-right" in p or "horizontally" in p):
        muts.append("flip_h")
    if "flip" in p and ("vertical" in p or "top-bottom" in p or "vertically" in p):
        muts.append("flip_v")
    return muts or None


# ----------------------------- public entrypoint ------------------------------

def run_arc(prompt: str,
            grid: list[list[int]] | np.ndarray | str,
            policy: Dict[str, Any] | None = None,
            context: Dict[str, Any] | None = None,
            rails: str = DEFAULT_RAILS,
            T: int = DEFAULT_T,
            *,
            use_wdd: bool = True,
            profile: bool = False,
            verbose: bool = False,
            sequence: Optional[Sequence[str]] = None,
            debug: bool = False, 
            ) -> Dict[str, Any]:
    """
    Singular prompt interface for ARC (nearly indistinguishable from DeFi’s).
    Returns a dict with plan.sequence, verify, flags, aux, det_hash, wdd_summary, abstained.
    """
    # Merge defaults (deep-copy via JSON like DeFi)
    pol = json.loads(json.dumps({**DEFAULT_POLICY, **(policy or {})}))
    ctx = json.loads(json.dumps({**DEFAULT_CONTEXT, **(context or {})}))

    # Drive WDD via policy.audit backend (parity with DeFi)
    if use_wdd:
        pol.setdefault("audit", {})
        pol["audit"]["backend"] = "wdd"
        pol["audit"]["profile"] = bool(profile)

    # NEW: honor CLI --debug
    pol.setdefault("audit", {})["debug"] = bool(debug)

    # Allow notebook/mapper-provided sequence to be honored by harness if policy supports it
    # (No separate ARC runner; call the harness directly)
    # Parse grid
    if isinstance(grid, str):
        grid_np = np.asarray(json.loads(grid), dtype=int)
    else:
        grid_np = np.asarray(grid, dtype=int)

    # 2) get mapper sequence, but let CLI override if provided
    mapper = _get_arc_mapper()
    cand_seq = list(sequence) if sequence else mapper.predict(prompt)
    cand_seq = [_canon_primitive(s) for s in cand_seq]
    
    # 3) pass the (possibly overridden) sequence into the harness
    res = run_arc_wdd(
        prompt=prompt,
        grid=grid_np,
        policy=pol,
        sequence=cand_seq,
        debug=bool(pol.get("audit", {}).get("debug", False))
    )

    # Normalize plan.sequence (canonicalize primitive names)
    seq: List[str] = [(res.get("plan") or {}).get("sequence") or []][0]
    seq = [_canon_primitive(s) for s in seq if s]

    # --- Verification shim (ARC) ---
    # IF family mode yields an order (seq non-empty) → ok
    # ELSE treat as Stage-4 accept (like DeFi's rails pass) unless abstain
    rails_v = res.get("verify") if isinstance(res.get("verify"), dict) else {"ok": True, "reason": "shim:accept:stage-4"}
    ok = bool(rails_v.get("ok", False))
    reason = rails_v.get("reason", "")
    if not seq and ok:
        # Treat as non-exec accept; keep reason (shim) for parity
        pass

    # Tags (same spirit as DeFi)
    tags = list(rails_v.get("tags") or [])
    rtag = f"rails:{rails}"
    if rtag not in tags:
        tags.append(rtag)
    audit = pol.get("audit") or {}
    if str(audit.get("backend", "")).lower() == "wdd":
        if "wdd:on" not in tags: tags.append("wdd:on")
        if "audit:wdd" not in tags: tags.append("audit:wdd")

    v_block = {"ok": ok, "reason": reason, "tags": tags}

    # Build minimal payload
    minimal = {
        "plan":   {"sequence": list(seq)},
        "verify": {"ok": bool(v_block.get("ok")), "reason": str(v_block.get("reason") or "")},
    }

    if not verbose:
        return minimal

    # Verbose: include flags/aux and an ARC-specific WDD summary table
    aux   = res.get("aux") or {}
    flags = res.get("flags") or {}

    # Compose deterministic hash of the sequence for quick comparisons
    det_hash = _seq_hash(seq)
    wdd_summary = res.get("wdd_summary") or _summarize_wdd(aux)
    abstained = _is_abstain(minimal, res)

    return {
        "prompt": prompt,
        "domain": "arc",
        "rails": rails,
        "T": T,
        "top1": None,              # ARC has no single top-primitive like DeFi mapper
        "sequence": list(seq),
        "plan": minimal["plan"],
        "verify": {**minimal["verify"], "tags": tags},
        "flags": flags,
        "aux": aux,
        "det_hash": det_hash,
        "wdd_summary": wdd_summary,
        "abstained": bool(abstained),
    }
