# Prompt -> Primitive Mapper: Scale Bench to 2–5k (Milestone 8)

### 1) Curate data
* Labeled training set (for the mapper)
    * File: defi_mapper_labeled_large.csv (your 1k, 8-class, balanced set — 125/class)
    * Schema: prompt,label
* Evaluation corpus at scale (5k)
    * Labels CSV: defi_mapper_labeled_5k.csv (balanced 625/class)
    * Prompts JSONL: defi_mapper_5k_prompts.jsonl
* (Optional) Per-class thresholds seed
    {
      "deposit_asset": 0.75,
      "withdraw_asset": 0.78,
      "swap_asset": 0.70,
      "borrow_asset": 0.68,
      "repay_asset": 0.72,
      "stake_asset": 0.75,
      "unstake_asset": 0.70,
      "claim_rewards": 0.72
    }

### 2) Train a calibrated mapper

```bash
PYTHONPATH=. python3 milestones/train_mapper_embed.py \
  --labels_csv tests/fixtures/defi_mapper_labeled_large.csv \
  --out_path .artifacts/defi_mapper_embed.joblib \
  --sbert sentence-transformers/all-mpnet-base-v2 \
  --C 8 --max_iter 2000 --calibrate
```

### 3) Threshold sweep
```bash
python3 milestones/defi_milestone8.py \
  --mapper_path .artifacts/defi_mapper_embed.joblib \
  --prompts_jsonl tests/fixtures/defi_mapper_5k_prompts.jsonl \
  --labels_csv   tests/fixtures/defi_mapper_labeled_5k.csv
  --thresholds "0.2,0.25,0.3,0.35,0.4" \
  --max_abstain_rate 0.20 \
  --min_overall_acc 0.85 \
  --choose_by utility \
  --per_class_thresholds tests/fixtures/m8_per_class_thresholds.json \
  --rows_csv .artifacts/m8_rows.csv \
  --out_summary .artifacts/m8_sum.json \
  --out_csv .artifacts/m8_metrics.csv
```

### 4) Evaluate

#### Global metrics
```bash
column -s, -t < .artifacts/m8_metrics.csv
```

OUTPUT:
```bash
threshold  total  abstain  abstain_rate  coverage  fired  correct_on_fired  accuracy_on_fired   overall_correct  overall_accuracy
0.2        5000   False    0.0           0.9994    4997   4942              0.9889933960376226  4942             0.9884
0.25       5000   False    0.0           0.9994    4997   4942              0.9889933960376226  4942             0.9884
0.3        5000   False    0.0           0.9994    4997   4942              0.9889933960376226  4942             0.9884
0.35       5000   False    0.0           0.9994    4997   4942              0.9889933960376226  4942             0.9884
0.4        5000   False    0.0           0.9976    4988   4939              0.9901764234161988  4939             0.9878
```

#### See which threshold M8 chose
```bash
jq '.chosen' .artifacts/m8_sum.json
```
{
  "threshold": 0.4,
  "abstain_rate": 0.0,
  "coverage": 0.9976,
  "accuracy_on_fired": 0.9901764234161988,
  "overall_accuracy": 0.9878
}

#### inspect mistakes
```bash
awk -F, 'NR==1 || ($2!=$3 && $3!="")' .artifacts/m8_rows.csv | head -25
```
"restake 33.8529 MATIC with balancer on solana — this minute, use normal gas",stake_asset,stake_asset,1.0000,False,0.4
"convert 7064 USDT into LINK via sushiswap on arbitrum — minimize gas, safe mode",swap_asset,swap_asset,0.8862,False,0.4
"collect incentives at compound (optimism) — use fast gas, normal mode",claim_rewards,claim_rewards,1.0000,False,0.4
"withdraw staked 0.667255 WBTC from pendle optimism — right away, use normal gas",unstake_asset,unstake_asset,1.0000,False,0.4

#### Inspect abstains (should be ~3 rows total)
```bash
awk -F, 'NR==1 || $5=="True"' .artifacts/m8_rows.csv
```
prompt,gold_label,predicted,confidence,abstain,threshold
draw 18096 DAI from compound on polygon,borrow_asset,,0.3750,True,0.4
draw 1.375626 BTC from maker on polygon,borrow_asset,,0.3859,True,0.4
convert 3258.0979 OP into AAVE via uniswap on optimism — minimize gas,swap_asset,,0.3750,True,0.4
unstow 1553 DAI from balancer on optimism,withdraw_asset,,0.1250,True,0.4
unstow 1.189804 WBTC from yearn on base — high yield mode,withdraw_asset,,0.1250,True,0.4
borrow 799.5374 LINK from aave on avalanche,borrow_asset,,0.3750,True,0.4
unlock 1.802823 BTC at yearn (base) — high yield mode,unstake_asset,,0.3750,True,0.4
unstow 1.080917 WBTC from uniswap on solana — minimize gas,withdraw_asset,,0.1250,True,0.4
ian_moore@Macmini micro-lm % 


#### Per-class accuracy (simple awk rollup)
```bash
awk -F, 'NR>1{g[$2]++; if($2==$3) c[$2]++} END{for(k in g) printf "%-18s %5d  acc=%.4f\n", k, g[k], (c[k]+0.0)/g[k]}' .artifacts/m8_rows.csv | sort
```

#### swap misclassifications, top examples
```bash
awk -F, 'NR>1 && $2=="swap_asset" && $3!=$2 {print $0}' .artifacts/m8_rows.csv | head -10
```
convert 26.7284 ETH into ARB via curve on base,swap_asset,deposit_asset,0.8333,False,0.4
convert 3258.0979 OP into AAVE via uniswap on optimism — minimize gas,swap_asset,,0.3750,True,0.4
convert 3.4123 AVAX into LINK via balancer on solana — asap,swap_asset,deposit_asset,0.5898,False,0.4
convert 36.7964 AVAX into ETH via balancer on polygon,swap_asset,deposit_asset,0.8685,False,0.4
convert 27.6588 AVAX into LINK via sushiswap on optimism,swap_asset,deposit_asset,0.6071,False,0.4
convert 625.8438 AAVE into MATIC via balancer on base,swap_asset,deposit_asset,0.9328,False,0.4
convert 4681.5472 ARB into ETH via uniswap on base,swap_asset,deposit_asset,0.4638,False,0.4
convert 32.8488 SOL into ETH via sushiswap on solana,swap_asset,deposit_asset,0.5300,False,0.4
convert 45.6938 MATIC into SOL via balancer on solana,swap_asset,deposit_asset,0.6667,False,0.4
convert 20.5658 MATIC into ARB via curve on avalanche — safe mode,swap_asset,deposit_asset,0.5909,False,0.4

#### borrow misclassifications
```bash
awk -F, 'NR>1 && $2=="borrow_asset" && $3!=$2 {print $0}' .artifacts/m8_rows.csv | head -10
```
draw 2003.1529 LINK from aave on optimism — ok with higher gas,borrow_asset,deposit_asset,0.4167,False,0.4
draw 18096 DAI from compound on polygon,borrow_asset,,0.3750,True,0.4
draw 1.375626 BTC from maker on polygon,borrow_asset,,0.3859,True,0.4
draw 1393.4151 LINK from maker on avalanche — now,borrow_asset,deposit_asset,0.6395,False,0.4
borrow 799.5374 LINK from aave on avalanche,borrow_asset,,0.3750,True,0.4
draw 2130.6562 LINK from aave on optimism,borrow_asset,deposit_asset,0.5236,False,0.4
draw 12004 DAI from maker on solana — ok with higher gas,borrow_asset,deposit_asset,0.4167,False,0.4
draw 659.3806 LINK from aave on base,borrow_asset,deposit_asset,0.9792,False,0.4
draw 3568.4042 LINK from maker on optimism — right away,borrow_asset,deposit_asset,0.4475,False,0.4
draw 3042.6438 LINK from aave on solana,borrow_asset,deposit_asset,0.5660,False,0.4

#### unstake misclassifications
```bash
awk -F, 'NR>1 && $2=="unstake_asset" && $3!=$2 {print $0}' .artifacts/m8_rows.csv | head -10
```
unstake 40.3119 ETH from curve on polygon,unstake_asset,withdraw_asset,0.5617,False,0.4
unstake 1508.0058 ARB from curve on optimism,unstake_asset,withdraw_asset,0.5243,False,0.4
unstake 6446 USDC from pendle on ethereum,unstake_asset,withdraw_asset,0.5000,False,0.4
unstake 5648 USDC from pendle on ethereum,unstake_asset,withdraw_asset,0.5000,False,0.4
unstake 9480 USDC from yearn on polygon,unstake_asset,withdraw_asset,0.5794,False,0.4
unstake 19660 USDC from pendle on polygon,unstake_asset,withdraw_asset,0.5528,False,0.4
unstake 2031.7065 ARB from lido on polygon — minimize gas,unstake_asset,withdraw_asset,0.5309,False,0.4
unlock 1.802823 BTC at yearn (base) — high yield mode,unstake_asset,,0.3750,True,0.4
unstake 23422 DAI from rocket pool on optimism — use normal gas,unstake_asset,withdraw_asset,0.5361,False,0.4
unstake 2384 USDC from rocket pool on base,unstake_asset,withdraw_asset,0.4984,False,0.4


#### Quick, robust per-class accuracy (no installs)
```bash
python3 - <<'PY'
import csv,collections
g=collections.Counter(); c=collections.Counter()
with open(".artifacts/m8_rows.csv", newline="") as f:
    r=csv.DictReader(f)
    for row in r:
        gold=row["gold_label"]; pred=row["predicted"]
        g[gold]+=1
        if pred==gold: c[gold]+=1
for k in sorted(g):
    print(f"{k:15s} {g[k]:5d}  acc={c[k]/g[k]:.4f}")
PY
```
borrow_asset      625  acc=0.9776
claim_rewards     625  acc=1.0000
deposit_asset     625  acc=0.9984
repay_asset       625  acc=0.9984
stake_asset       625  acc=0.9984
swap_asset        625  acc=0.9696
unstake_asset     625  acc=0.9808
withdraw_asset    625  acc=0.9792

---
# Prompt -> Primitive Mapper: small scale (< 200) (Milestone 7)

### 1) Paths
```bash
export LAB_MINI="tests/fixtures/defi_mapper_labeled_mini.csv"  
export MAPPER_OUT=".artifacts/defi_mapper_embed.joblib"
mkdir -p .artifacts
```

### 2) Curate a tiny, clean training set

* CSV: prompt,label
    * 20–50 examples/class, balanced
    * Include synonyms you know are production-relevant (“top up”, “add” → deposit_asset)
    * Keep non-execs out (we handle those with front-gate/rails), unless you explicitly add a nonexec class later

### 3) Train a simple calibrated mapper
```bash
PYTHONPATH=. python3 milestones/train_mapper_embed.py \
  --labels_csv "$LAB_MINI" \
  --out_path  "$MAPPER_OUT" \
  --sbert sentence-transformers/all-mpnet-base-v2 \
  --C 8 --max_iter 2000 --calibrate
```
* It’s the same invocation shown in the training script’s header docstring, including the calibration switch and SBERT backbone .
* The script enforces prompt,label columns and does tiny-data-friendly calibration automatically (isotonic/sigmoid or skip when too few samples)

### 4) Threshold sweep (eval-only)
```bash
export PROMPTS_JSONL="tests/fixtures/defi_mapper_stress_prompts.jsonl"
export LABELS_CSV="tests/fixtures/defi_mapper_stress_labeled.csv" # optional but recommended

python3 milestones/defi_milestone7.py \
  --mapper_path "$MAPPER_OUT" \
  --prompts_jsonl "$PROMPTS_JSONL" \
  --labels_csv    "$LABELS_CSV" \
  --thresholds "0.20,0.25,0.30,0.35,0.40" \
  --max_abstain_rate 0.20 \
  --min_overall_acc 0.85 \
  --choose_by utility \
  --rows_csv .artifacts/m7_rows.csv \
  --out_summary .artifacts/m7_summary.json \
  --out_csv     .artifacts/m7_metrics.csv
```
* Selection rule: minimize abstain under the cap, then maximize overall accuracy (and/or utility).
*Operating point (OP) for your run: thr=0.30 (0.925 overall, 0% abstain, full coverage).
* Artifacts to freeze: `mapper.joblib`, `m7_summary.json`, `m7_metrics.csv`, and `m7_rows.csv`.
* Handy one-liners to glance at outputs (these exact examples are in the M7 header docstring):
```bash
column -s, -t < .artifacts/m7_metrics.csv | sed -n '1,12p'
awk -F, 'NR==1 || ($5=="False" && $2!=$3)' .artifacts/m7_rows.csv | sed -n '1,20p'
jq '.chosen' .artifacts/m7_summary.json
```

### 5) Per-class tweaks (optional, only where needed)
If confusions cluster (e.g., borrow vs deposit), create:
```bash
{
  "borrow_asset": 0.62,
  "deposit_asset": 0.58
}
```
…and pass --per_class_thresholds into defi_milestone7.py. This keeps global coverage while shaving specific mistakes.

### 6) Template Makefile targets (so it’s one-command next time)

```bash
ARTIFACTS := .artifacts
MAPPER := $(ARTIFACTS)/defi_mapper_embed.joblib
LAB := tests/fixtures/defi_mapper_labeled_mini.csv
PROMPTS := tests/fixtures/defi_mapper_stress_prompts.jsonl
LABELS := tests/fixtures/defi_mapper_stress_labeled.csv

.PHONY: mapper-train
mapper-train:
\tPYTHONPATH=. python3 milestones/train_mapper_embed.py \
\t  --labels_csv $(LAB) --out_path $(MAPPER) \
\t  --sbert sentence-transformers/all-mpnet-base-v2 \
\t  --C 8 --max_iter 2000 --calibrate

.PHONY: m7-sweep
m7-sweep:
\tpython3 milestones/defi_milestone7.py \
\t  --mapper_path $(MAPPER) \
\t  --prompts_jsonl $(PROMPTS) \
\t  --labels_csv $(LABELS) \
\t  --thresholds "0.20,0.25,0.30,0.35,0.40" \
\t  --max_abstain_rate 0.20 \
\t  --min_overall_acc 0.85 \
\t  --choose_by utility \
\t  --rows_csv $(ARTIFACTS)/m7_rows.csv \
\t  --out_summary $(ARTIFACTS)/m7_summary.json \
\t  --out_csv $(ARTIFACTS)/m7_metrics.csv
```
