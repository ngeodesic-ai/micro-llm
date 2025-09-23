# src/micro_lm/cli/micro_defi.py
from __future__ import annotations
import argparse, json, os, sys
from typing import Any, Dict
from micro_lm.cli.defi_quickstart import quickstart

def _parse_json(label: str, s: str | None) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        print(f"[micro-defi] invalid {label} JSON: {e}", file=sys.stderr)
        sys.exit(2)

def main() -> None:
    p = argparse.ArgumentParser(prog="micro-defi", description="DeFi quickstart CLI")
    p.add_argument("-p", "--prompt", type=str, help="user prompt (e.g. 'deposit 10 ETH into aave')")
    p.add_argument("--rails", type=str, default="stage11", help="rails stage (default: stage11)")
    p.add_argument("--policy", type=str, default=None, help="JSON policy string")
    p.add_argument("--context", type=str, default=None, help="JSON context string")
    p.add_argument("--use_wdd", action="store_true", help="run WDD detector (mapper-driven)")
    p.add_argument("--pca_prior", type=str, default=None, help="optional PCA prior .npz for WDD")
    p.add_argument("--T", type=int, default=180, help="time budget (s)")
    p.add_argument("--verbose", action="store_true", help="verbose/WDD debug passthrough")
    p.add_argument("positional_prompt", nargs="?", help="prompt (positional alternative)")

    args = p.parse_args()
    prompt = args.prompt or args.positional_prompt
    if not prompt:
        p.error("missing prompt (use -p/--prompt or positional)")

    pol = _parse_json("policy", args.policy)
    ctx = _parse_json("context", args.context)

    # honor WDD debug like the CLI examples
    if args.verbose:
        os.environ.setdefault("MICRO_LM_WDD_DEBUG", "1")

    out = quickstart(
        prompt=prompt,
        policy=pol,
        context=ctx,
        rails=args.rails,
        T=args.T,
        use_wdd=args.use_wdd,        # set when you want detector mode without policy
        pca_prior=args.pca_prior,
        verbose=args.verbose,
    )
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
