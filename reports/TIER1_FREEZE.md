# Milestone 5 — Smoke Benchmark Freeze ✅

**Goal:**  
Establish a minimal working benchmark for DeFi domain with local verification + shim fallback. This milestone locks in a reproducible baseline before expanding to integration benchmarks.

---

## Acceptance Criteria
- Local domain verification (`domains/defi/`) is preferred; shim fallback is available if ngeodesic not present.  
- `rails_shim` returns stable reasons (`local:verified`, `shim:accept:stage-N`, or `ltv`).  
- Smoke benchmark **passes with ok accuracy ≥ 0.66**.

---

## Commands

### Run Smoke Benchmark
```bash
MICROLM_DISABLE_RAILS=1 micro-lm-bench defi \
  --file benches/defi_smoke.jsonl \
  --out .artifacts/defi_smoke_results.jsonl \
  --summary-out .artifacts/defi_smoke_summary.json \
  --gate-metric ok_acc --gate-min 0.66
```

### Verify Summary
```bash
python3 - <<'PY'
import json, sys
s = json.load(open(".artifacts/defi_smoke_summary.json"))
acc = s.get("ok_acc", 0.0)
print(f"[bench] ok={s['ok']} total={s['total']} acc={acc:.2f}")
sys.exit(0 if acc >= 0.66 else 1)
PY
```

---

## Results (Frozen)
```
total:   3
ok:      2
ok_acc:  0.667
label_acc: 1.0
expect_ok_acc: 1.0
exact_acc: 1.0
```

**Gate:** `ok_acc ≥ 0.66`  
**Status:** PASS ✅  

---

## Notes
- Stage 5 is the **freeze point**. No further changes should regress below this benchmark.  
- This milestone sets the baseline for Stage 6 integration benchmarks.  
- Artifacts (`.artifacts/defi_smoke_results.jsonl`, `.artifacts/defi_smoke_summary.json`) are reference outputs.  
