# src/micro_lm/core/audit/wdd.py

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Iterable

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float((a @ b) / (na * nb))

def _centroid(M: np.ndarray) -> np.ndarray:
    if M.size == 0:
        return np.zeros((M.shape[-1],), dtype=float)
    return M.mean(axis=0)

def wdd_audit(
    emb: np.ndarray | None,
    *,
    bank: Dict[str, Iterable[np.ndarray]] | None,
    anchors: Dict[str, Iterable[np.ndarray]] | None,
    tau_abs: float = 0.60,
    tau_rel: float = 0.50,
    tau_span: float = 0.50,
    mode: str = "pure",
) -> Dict[str, Any]:
    """
    Lightweight, deterministic WDD-like gate.
    - If embeddings/banks are missing, fall back to a neutral 'ok' with n_keep=0.
    - Otherwise compute simple cosine-based span/relative checks.
    Returns: {"ok": bool, "score": float, "reason": str, "n_keep": int}
    """
    # graceful fallback for Tier-0 wordmap or missing latents
    if emb is None or bank is None or anchors is None:
        return {"ok": True, "score": 1.0, "reason": "wdd:skip", "n_keep": 0}

    # flatten banks/anchors into arrays
    bank_vecs = []
    for _, vecs in bank.items():
        for v in vecs:
            bank_vecs.append(np.asarray(v, dtype=float))
    bank_arr = np.vstack(bank_vecs) if bank_vecs else np.zeros((0, emb.shape[-1]))

    anchor_vecs = []
    for _, vecs in anchors.items():
        for v in vecs:
            anchor_vecs.append(np.asarray(v, dtype=float))
    anchor_arr = np.vstack(anchor_vecs) if anchor_vecs else np.zeros((0, emb.shape[-1]))

    if bank_arr.size == 0 or anchor_arr.size == 0:
        return {"ok": True, "score": 1.0, "reason": "wdd:degenerate", "n_keep": 0}

    c_bank = _centroid(bank_arr)
    c_anchor = _centroid(anchor_arr)

    s_abs = _cosine(emb, c_bank)                  # absolute alignment with bank centroid
    s_rel = max(0.0, s_abs - _cosine(emb, c_anchor))  # relative vs. anchors
    s_span = float(np.clip((s_abs + s_rel) / 2.0, 0.0, 1.0))

    ok = (s_abs >= tau_abs) and (s_rel >= tau_rel) and (s_span >= tau_span)
    score = float(0.5 * s_abs + 0.3 * s_rel + 0.2 * s_span)

    reason = (
        f"wdd:{'pass' if ok else 'abstain'}:"
        f"abs={s_abs:.2f},rel={s_rel:.2f},span={s_span:.2f}"
    )
    # n_keep is a proxy for “how much we’d keep post-filter”; simple heuristic:
    n_keep = int(ok)

    return {"ok": ok, "score": score, "reason": reason, "n_keep": n_keep}
