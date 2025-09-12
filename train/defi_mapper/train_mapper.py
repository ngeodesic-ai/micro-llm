# Tiny sklearn trainer: CountVectorizer + LogisticRegression
from __future__ import annotations
import json, argparse, pathlib
from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def load_jsonl(path: str) -> Tuple[List[str], List[str]]:
    X, y = [], []
    with open(path, "r") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            X.append(obj["text"])
            y.append(obj["intent"])
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset.jsonl")
    ap.add_argument("--out",  required=True, help="path to write mapper.joblib")
    ap.add_argument("--test", help="optional heldout.jsonl")
    args = ap.parse_args()

    X, y = load_jsonl(args.data)
    pipe = Pipeline([
        ("vec", CountVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, multi_class="auto"))
    ])
    pipe.fit(X, y)

    if args.test:
        Xte, yte = load_jsonl(args.test)
        yhat = pipe.predict(Xte)
        print(classification_report(yte, yhat))

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out)
    print(f"saved model to {args.out}")

if __name__ == "__main__":
    main()
