import json, argparse
from micro_lm.pipelines.runner import run_micro

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="borrow 500 USDC vs ETH")
    ap.add_argument("--rails", default="stage11")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    context = {"risk": {"hf": 1.22}, "oracle": {"age_sec": 5, "max_age_sec": 30}}
    policy = {"ltv_max": 0.75}

    res = run_micro("defi", args.prompt, context, policy, rails=args.rails, T=180)
    s = json.dumps(res, indent=2, default=str)
    print(s)
    if args.out:
        with open(args.out, "w") as f: f.write(s)

if __name__ == "__main__": main()
