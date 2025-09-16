import csv, json

def load_prompts_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def load_prompts_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))
