# Milestone 8 (scaled 5k prompts; identical structure to m7)
# Milestone 7 — Mapper Threshold Sweep (Eval)
- Prompts: **5000**
- Labels provided: **True**

## Per-threshold metrics
| thr | abstain_rate | coverage | acc_on_fired | overall_acc |
|---:|---:|---:|---:|---:|
| 0.20 | 0.000 | 0.999 | 0.989 | 0.988 |
| 0.25 | 0.000 | 0.999 | 0.989 | 0.988 |
| 0.30 | 0.000 | 0.999 | 0.989 | 0.988 |
| 0.35 | 0.000 | 0.999 | 0.989 | 0.988 |
| 0.40 | 0.000 | 0.998 | 0.990 | 0.988 |

## Chosen operating point
- **threshold:** `0.40`
- **abstain_rate:** `0.000`
- **coverage:** `0.998`
- **accuracy_on_fired:** `0.990`
- **overall_accuracy:** `0.988`