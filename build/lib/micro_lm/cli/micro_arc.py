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
"""

from __future__ import annotations
import argparse, json, os, sys
from typing import Any, Dict, List, Optional
import numpy as np

from micro_lm.interfaces.arc_prompt import run_arc

def _parse_json(label: str, s: str | None) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        print(f"[micro-arc] invalid {label} JSON: {e}", file=sys.stderr)
        sys.exit(2)

def _parse_grid(s: str) -> np.ndarray:
    try:
        data = json.loads(s)
        return np.asarray(data, dtype=int)
    except Exception as e:
        print(f"[micro-arc] invalid --grid JSON: {e}", file=sys.stderr)
        sys.exit(2)

def main() -> None:
    p = argparse.ArgumentParser(prog="micro-arc", description="ARC quickstart CLI")
    p.add_argument("-p", "--prompt", type=str, help="user prompt (e.g. 'rotate then flip horizontally')")
    p.add_argument("--grid", type=str, required=True,
                   help="Grid as JSON list-of-lists, e.g. '[[1,2],[3,4]]'")
    p.add_argument("--rails", type=str, default="stage11", help="rails stage (default: stage11)")
    p.add_argument("--policy", type=str, default=None, help="JSON policy string")
    p.add_argument("--context", type=str, default=None, help="JSON context string")
    p.add_argument("--use_wdd", action="store_true",
                   help="run WDD detector/audit (mapper-driven); usually on via policy")
    p.add_argument("--mode", type=str, choices=["family", "detector", "passthrough"],
                   help="audit mode -> policy.audit.mode")
    p.add_argument("--T", type=int, default=180, help="time budget (s)")
    p.add_argument("--verbose", action="store_true", help="verbose output (include aux, summary)")
    # Debug/dev only: manual sequence override (off by default)
    p.add_argument("--allow-sequence-override", action="store_true",
                   help="Allow manual --sequence (debug/dev only). Disabled by default.")
    p.add_argument("--sequence", type=str, default="",
                   help="Comma-separated primitives (e.g., rotate,flip_h). Ignored unless --allow-sequence-override is set.")
    # Positional prompt fallback (parity with micro-defi)
    p.add_argument("positional_prompt", nargs="?", help="prompt (positional alternative)")

    args = p.parse_args()
    prompt = args.prompt or args.positional_prompt
    if not prompt:
        p.error("missing prompt (use -p/--prompt or positional)")

    pol = _parse_json("policy", args.policy)
    ctx = _parse_json("context", args.context)

    # Honor WDD debug like in DeFi
    if args.verbose:
        os.environ.setdefault("MICRO_LM_WDD_DEBUG", "1")

    # Merge CLI audit.mode into policy (deep-merge just for audit)
    if args.mode:
        pol.setdefault("audit", {})["mode"] = args.mode

    # Grid
    grid = _parse_grid(args.grid)

    # Manual sequence override (opt-in only)
    seq: Optional[List[str]] = None
    if args.allow_sequence_override:
        seq = [s.strip() for s in args.sequence.split(",") if s.strip()]
    elif args.sequence and args.verbose:
        print("[micro-arc] Ignoring --sequence (blind mode). Use --allow-sequence-override to enable.",
              file=sys.stderr)

    # Run
    out = run_arc(
        prompt=prompt,
        grid=grid,
        policy=pol,
        context=ctx,
        rails=args.rails,
        T=args.T,
        use_wdd=args.use_wdd,
        verbose=True,          # print a rich, DeFi-like payload
        sequence=seq,
        debug=args.verbose,    # propagate debug through policy.audit.debug
    )

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
