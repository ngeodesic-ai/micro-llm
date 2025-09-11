import json, argparse
from micro_llm.pipelines.runner import run_micro

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="flip then rotate")
    ap.add_argument("--rails", default="stage11")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    context = {"train_pairs": [], "test_inputs": []}
    policy = {"max_ops": 3}

    res = run_micro("arc", args.prompt, context, policy, rails=args.rails, T=160)
    s = json.dumps(res, indent=2, default=str)
    print(s)
    if args.out:
        with open(args.out, "w") as f: f.write(s)

if __name__ == "__main__": main()
