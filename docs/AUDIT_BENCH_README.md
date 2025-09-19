# Audit Bench — Evidence-Only (Tautology-Free)

This bench reproduces the fixed `audit_bench_metrics.py` behavior inside the package.

## Run

```bash
python -m micro_lm.cli.defi_audit_bench   --prompts_jsonl tests/fixtures/defi/defi_mapper_5k_prompts.jsonl   --labels_csv    tests/fixtures/defi/defi_mapper_labeled_5k.csv   --sbert sentence-transformers/all-MiniLM-L6-v2   --n_max 4 --tau_span 0.50 --tau_rel 0.60 --tau_abs 0.93   --L 160 --beta 8.6 --sigma 0.0   --out_dir .artifacts/defi/audit_bench   --competitive_eval
```

## Outputs

- `.artifacts/defi/audit_bench/rows_audit.csv`
- `.artifacts/defi/audit_bench/metrics_audit.json`

## Quick sanity checks

Count pass vs. abstain from CSV:
```bash
awk -F, 'NR>1 && $5=="True"{p++} NR>1 && ($3=="" || tolower($6) ~ /abstain/){a++} END{print "pass=",p,"abstain=",a}' .artifacts/defi/audit_bench/rows_audit.csv
```

Show key metrics:
```bash
jq '{coverage,abstain_rate,span_yield_rate,hallucination_rate,multi_accept_rate,params}' .artifacts/defi/audit_bench/metrics_audit.json
```

## Notes

- The audit is **evidence-only**: no mapper signals, no confidence thresholds—fixes the tautology.
- Tune ambiguity via `--tau_rel` and `--tau_abs`. Increase either to reduce multi-accepts.
