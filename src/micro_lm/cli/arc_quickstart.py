"""
# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Copyright 2025 Ian C. Moore (Provisional Patents #63/864,726, #63/865,437, #63/871,647 and #63/872,334)
# Email: ngeodesic@gmail.com
# Part of Noetic Geodesic Framework (NGF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


Tiny CLI to execute one ARC task at a time (mirrors DeFi quickstart).

Examples:
    # JSON grid
    python -m micro_lm.cli.arc_quickstart \
        --prompt "flip the grid horizontally then rotate it" \
        --grid '[[3,0,1,2],[3,2,1,0],[3,2,0,5],[6,1,4,2]]' \
        --mode family

    # Explicit sequence override (ignores detection)
    python -m micro_lm.cli.arc_quickstart \
        --prompt "just do flips" \
        --grid '[[1,2],[3,4]]' \
        --sequence flip_h,flip_v

    # Detector audit mode (keep list but do not plan)
    python -m micro_lm.cli.arc_quickstart \
        --prompt "rotate the grid 90 degrees" \
        --grid '[[1,2],[3,4]]' \
        --mode detector
"""

from __future__ import annotations
import argparse, json, sys
from typing import Any, Dict, List, Optional
import numpy as np
from micro_lm.interfaces.arc_prompt import run_arc

class SBERTEncoder:  # noqa: N801
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 normalize=True, **kwargs):
        # NOTE: when unpickled, __init__ is NOT called. Keep this for the fresh case.
        self._model_name = model_name
        self.normalize = bool(normalize)

    def _ensure_model(self):
        # When unpickled, __init__ usually didn't run; try existing attrs first.
        m = getattr(self, "model", None) or getattr(self, "_model", None)
        if m is None:
            from sentence_transformers import SentenceTransformer
            name = getattr(self, "_model_name", "sentence-transformers/all-MiniLM-L6-v2")
            m = SentenceTransformer(name)
            # set both for compatibility with different pickled layouts
            try:
                setattr(self, "model", m)
            except Exception:
                pass
            try:
                setattr(self, "_model", m)
            except Exception:
                pass
        return m

    def transform(self, texts):
        import numpy as np
        m = self._ensure_model()
        normalize = getattr(self, "normalize", True)
        emb = m.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.asarray(emb)

# Make sure unpickler can resolve __main__.SBERTEncoder
setattr(sys.modules[__name__], "SBERTEncoder", SBERTEncoder)
# -------------------------------------------------------------------------



def _parse_grid(arg: str) -> np.ndarray:
    try:
        data = json.loads(arg)
        return np.asarray(data, dtype=int)
    except Exception as e:
        print(f"[arc] Failed to parse --grid JSON: {e}", file=sys.stderr)
        raise


def _maybe_print_wdd_verbose(args, out):
    if not args.verbose:
        return
    aux = (out.get("aux") or {})
    arc = (((aux.get("stage11") or {}).get("wdd") or {}).get("arc") or {})
    results = arc.get("results") or {}
    seq = (out.get("plan") or {}).get("sequence") or []
    print(f"[WDD] prompt={args.prompt!r} | seq={seq}")

    # pretty header
    print("[WDD] {:8s} {:5s} {:7s} {:6s} {:9s} {:9s} {:9s} {:9s}".format(
        "act","keep","t_peak","p_map","corr_max","z_abs","area","rel_ok"
    ))
    for name, rec in results.items():
        ok = bool(rec.get("ok"))
        info = rec.get("info") or {}
        gate = info.get("gate") or {}
        fam  = gate.get("family","")
        rel_ok = gate.get("rel_ok")
        corr = info.get("corr_max")
        z    = info.get("z_abs")
        area = info.get("area")
        tpk  = (info.get("t_peak") or {}).get(name)
        pmap = info.get("mapper_p")
        print("[WDD] {:8s} {:5s} {:7} {:6} {:9.3f} {:9.3f} {:9.3f} {:9}".format(
            name, "PASS" if ok else "—", tpk, f"{pmap:.2f}" if pmap is not None else "—",
            corr if corr is not None else float("nan"),
            z if z is not None else float("nan"),
            area if area is not None else float("nan"),
            str(rel_ok) if rel_ok is not None else "—",
        ))

    # thresholds once
    thr = (out.get("wdd_summary") or {}).get("thresholds")
    if thr:
        print(f"[WDD] thresholds: {thr}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ARC quickstart — singular prompt interface")
    ap.add_argument("--prompt", type=str, required=True, help="Instruction for the ARC grid")
    ap.add_argument("--grid", type=str, required=True, help="Grid as JSON list-of-lists, e.g. '[[1,2],[3,4]]'")
    ap.add_argument("--sequence", type=str, default="", help="Optional comma-separated primitive list (e.g., flip_h,rotate)")
    ap.add_argument("--mode", type=str, default="", choices=["", "family", "detector", "passthrough"],
                    help="Audit/detection mode forwarded into policy.audit.mode")
    ap.add_argument("--rails", type=str, default="stage11", help="Rails label for display (default: stage11)")
    ap.add_argument("--policy_json", type=str, default="", help="Optional extra policy JSON to merge")
    ap.add_argument("--verbose", action="store_true", help="Verbose WDD trace (like DeFi)")
    ap.add_argument("--debug", action="store_true", help="Verbose harness logs")
    ap.add_argument("--out_json", type=str, default="", help="Write full result JSON to file")
    args = ap.parse_args(argv)

    # Build policy (mode + optional merge)
    policy: Dict[str, Any] = {}
    if args.mode:
        policy.setdefault("audit", {})["mode"] = args.mode
    
    if args.policy_json:
        try:
            extra = json.loads(args.policy_json)
            # deep-merge into policy, preserving audit.mode while adding other audit keys
            if "audit" in extra:
                policy.setdefault("audit", {})
                for k, v in extra["audit"].items():
                    if k == "mode" and "mode" in policy["audit"]:
                        continue  # keep CLI --mode
                    policy["audit"][k] = v
                extra = {k: v for k, v in extra.items() if k != "audit"}
            # shallow merge the rest
            policy = {**extra, **policy} if policy else extra
        except Exception as e:
            print(f"[arc] bad --policy_json: {e}", file=sys.stderr)
            return 2

    grid = _parse_grid(args.grid)
    seq = [s.strip() for s in args.sequence.split(",") if s.strip()] or None

    out = run_arc(
        prompt=args.prompt,
        grid=grid,
        policy=policy or None,
        sequence=seq,
        debug=bool(args.debug),
    )

    # Enforce rails label for display parity
    out["rails"] = args.rails or out.get("rails", "stage11")

    # Verbose WDD trace (parity with DeFi)
    _maybe_print_wdd_verbose(args, out)

    # Console summary (compact)
    plan = (out.get("plan") or {})
    seq_print = plan.get("sequence") or []
    verify = (out.get("verify") or {})
    keep = ((out.get("wdd_summary") or {}).get("keep") or [])
    order = ((out.get("wdd_summary") or {}).get("order") or [])
    decision = ((out.get("wdd_summary") or {}).get("decision") or "")

    print(json.dumps(out, indent=2))  # print the full JSON like your DeFi example

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[arc] wrote {args.out_json}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
