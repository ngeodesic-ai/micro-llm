import json, argparse
from micro_llm.rails.metrics import summarize_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=False, help="path to events.jsonl")
    args = ap.parse_args()
    events = []
    if args.events:
        with open(args.events) as f:
            for line in f: events.append(json.loads(line))
    print(json.dumps(summarize_metrics(events), indent=2))

if __name__ == "__main__": main()
