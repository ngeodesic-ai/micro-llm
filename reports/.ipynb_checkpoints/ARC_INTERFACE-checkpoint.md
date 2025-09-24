# ARC_INTERFACE.md

## Quickstart Flags & Modes — ARC (Tier 1 & Tier 2)

This document explains how to run **micro-lm** ARC prompts using the `micro-arc` entry point (CLI + Python).  
It mirrors the DeFi interface but highlights ARC’s specific differences: the **grid input**, the **primitive space** (`rotate`, `flip_h`, `flip_v`), and the role of WDD (detector/family).

---

## 0) TL;DR Presets

### Tier-1 (mapper audit only, no WDD)
```bash
micro-arc -p "rotate the grid 90 degrees"   --grid '[[1,2],[3,4]]'   --rails stage11   --policy '{"mapper":{"model_path":".artifacts/arc_mapper.joblib","confidence_threshold":0.70}}'   --verbose
```

### Tier-2 (WDD detector audit; mapper decides action)
```bash
micro-arc -p "flip the grid horizontally"   --grid '[[1,2],[3,4]]'   --rails stage11   --policy '{"audit":{"backend":"wdd"},"mapper":{"confidence_threshold":-1.0}}'   --verbose
```

### Tier-2 (WDD family; WDD supplies plan/ordering)
```bash
micro-arc -p "rotate the grid then flip vertically"   --grid '[[1,2],[3,4]]'   --rails stage11   --policy '{"audit":{"backend":"wdd","mode":"family","K":12,"template_width":64,"z_abs":0.55,"keep_frac":0.70}}'   --verbose
```

---

## 1) Entry Points

- **CLI:**  
  ```bash
  micro-arc -p "<prompt>" --grid '<json-grid>' [--rails stage11] [--policy JSON] [--verbose]
  ```

- **Python:**  
  ```python
  from micro_lm.interfaces.arc_prompt import run_arc

  out = run_arc("flip the grid vertically", [[1,2],[3,4]],
                policy={"audit":{"backend":"wdd","mode":"family"}})
  ```

Output contract matches DeFi (`plan`, `verify`, `flags`, `aux`, `wdd_summary`).

---

## 2) Policy Schema

### 2.1 `policy.mapper`
- Same as DeFi.
- Points to ARC mapper artifact (`.artifacts/arc_mapper.joblib` by default).
- Keys: `model_path`, `confidence_threshold`, `skip_gate`.

### 2.2 `policy.audit`
- Keys match DeFi:
  - `backend: "wdd"`
  - `mode: "detector"` → mapper sequence is authoritative, WDD is advisory.  
  - `mode: "family"`   → WDD is authoritative, plan sequence comes from WDD.  
  - `K`, `template_width`, `z_abs`, `keep_frac` control family sampler.  
  - `debug: true` prints full metrics.

### 2.3 ARC-specific
Currently only the WDD audit path is supported. (Future: RAG/coverage checks.)

---

## 3) Environment Variables

- `MICRO_LM_WDD_DEBUG=1` → emit `[WDD] act=...` lines with primitive metrics (`t_peak`, `corr_max`, etc.)
- `ARC_MAPPER_PATH` → override default `.artifacts/arc_mapper.joblib`

---

## 4) Execution Semantics

1. **Mapper** proposes candidate primitives for the prompt.  
2. **WDD** audits those candidates:
   - **Detector mode:** mapper sequence remains; WDD only attaches keep/metrics.  
   - **Family mode:** WDD supplies full ordered plan.  
3. **Verify**:  
   - Detector → `shim:accept:stage-4` unless gated.  
   - Family   → `wdd:family:arc` if non-empty; `wdd:family:abstain` if empty.

---

## 5) CLI Flags

| Flag | Meaning |
|------|---------|
| `-p/--prompt` | Natural language instruction (“rotate then flip vertically”). |
| `--grid` | JSON grid (list of lists). Required. |
| `--rails` | Rails stage (default: `stage11`). |
| `--policy` | Policy JSON string (mapper + audit). |
| `--context` | Context JSON (not yet used in ARC). |
| `--mode` | Shorthand for `policy.audit.mode`. |
| `--use_wdd` | Shorthand for `policy.audit.backend="wdd"`. |
| `--verbose` | Print full JSON output including aux + summary. |
| `--allow-sequence-override` | Dev-only: feed a manual `--sequence`. |
| `--sequence` | Comma-separated primitive list. Ignored unless override enabled. |

---

## 6) FAQ

**Q: Why is my plan empty with WDD on?**  
A: In family mode, if WDD rejects all primitives, `verify.ok=false` and `plan.sequence=[]`.

**Q: How do I read aux metrics?**  
- `t_peak`: index of response peak.  
- `corr_max`: peak correlation.  
- `area`: integrated matched filter response.  
- `window`: window used for detection.  
- `z_abs`: z-score vs null distribution.  

**Q: Which primitives were considered?**  
See `aux.stage11.wdd.arc.results`.

---

## 7) Primitive Space

Canonical ARC primitives:
```
rotate   → rotate 90°
flip_h   → flip horizontally
flip_v   → flip vertically
```
