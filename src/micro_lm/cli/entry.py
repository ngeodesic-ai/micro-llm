import argparse, json
from micro_lm.core.runner import run_micro

def main(argv=None):
    ap = argparse.ArgumentParser("micro-lm")
    ap.add_argument("domain", choices=["defi", "arc"])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--rails", default="stage11")
    ap.add_argument("--T", type=int, default=180)
    ap.add_argument("--context", default="{}")
    ap.add_argument("--policy", default="{}")
    ap.add_argument("--backend", default="sbert")
    args = ap.parse_args(argv)

    out = run_micro(
        args.domain,
        args.prompt,
        context=json.loads(args.context),
        policy=json.loads(args.policy),
        rails=args.rails,
        T=args.T,
        backend=args.backend,
    )
    print(json.dumps(out, indent=2))
