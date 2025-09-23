# src/micro_lm/quickstarts/defi.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import json, argparse, os
from micro_lm.core.runner import run_micro  # public in module map
from micro_lm.interfaces.prompt import quickstart  # public in module map
from micro_lm.domains.defi.verify_local import verify_action_local  # package map shows it's exported
import hashlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--policy", default="")
    ap.add_argument("--context", default="")
    ap.add_argument("--rails", default="stage11")
    ap.add_argument("--T", type=int, default=180)
    ap.add_argument("--use_wdd", action="store_true")
    ap.add_argument("--pca_prior", default=None)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    policy = json.loads(args.policy) if args.policy else None
    context = json.loads(args.context) if args.context else None
    out = quickstart(
        args.prompt,
        policy=policy,
        context=context,
        rails=args.rails,
        T=args.T,
        use_wdd=args.use_wdd,
        pca_prior=args.pca_prior,
        profile=args.profile,
        verbose=args.verbose,
    )
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

