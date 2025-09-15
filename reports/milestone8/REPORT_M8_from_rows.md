# Milestone 8 — Grid Summary (from row-level CSVs)

- Files scanned: **9**
- Selection rule: **Max exec_accuracy_exact with abstain ≤ 0.10**
- Output CSV: `/mnt/data/m8_grid_from_rows.csv`
- Plot: `/mnt/data/m8_execacc_vs_abstain.png`

## Chosen Operating Point
```
{
  "thr": 0.6,
  "near_margin": 0.95,
  "exec_accuracy_exact": 1.0,
  "abstain_rate": 0.0,
  "nonexec_abstain_rate": 0.0,
  "nonexec_hallucination_rate": 1.0,
  "exec_verify_pass_rate": 1.0,
  "exec_stability_rate": 1.0,
  "blocked_verify_block_rate": 0.2
}
```