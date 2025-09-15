"""
python3 scripts/m7_choose_per_class_thresholds.py .artifacts/m7_rows.csv \
  > .artifacts/m7_per_class_thresholds.json
"""


import json, csv, numpy as np, sys
rows = list(csv.DictReader(open(sys.argv[1])))
classes = sorted(set(r["gold_label"] for r in rows))
# sweep
cand = [0.2,0.25,0.3,0.35,0.4,0.45,0.5]
best = {}
for c in classes:
    best[c] = 0.2
    best_util = -1
    for t in cand:
        fired = [r for r in rows if r["predicted"]==c and float(r["confidence"])>=t]
        abstain = [r for r in rows if r["predicted"]==c and float(r["confidence"])<t]
        acc = (sum(r["gold_label"]==r["predicted"] for r in fired) / max(1,len(fired)))
        util = acc * (len(fired)/(len(fired)+len(abstain)+1e-9))
        if util > best_util:
            best[c], best_util = t, util
print(json.dumps(best, indent=2))