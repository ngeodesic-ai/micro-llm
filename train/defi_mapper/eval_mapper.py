import argparse, joblib, json
from sklearn.metrics import classification_report

def load_jsonl(p):
    X,y=[],[]
    for ln in open(p):
        if ln.strip():
            o=json.loads(ln); X.append(o["text"]); y.append(o["intent"])
    return X,y

ap=argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
args=ap.parse_args()

pipe = joblib.load(args.model)
X,y = load_jsonl(args.data)
pred = pipe.predict(X)
print(classification_report(y, pred))
