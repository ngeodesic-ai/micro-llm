# Tier-2 WDD Refactor & Delivery Plan (ARC + DeFi)

This staged plan turns the current working ARC/DeFi WDD notebooks into a clean, reusable **Tier-2** package with a shared core and thin domain layers. Each stage lists: **Goal → Tasks → Acceptance → Deliverables**.

## Stage 0 — Repo hygiene (½ day)
**Goal:** Lock a clean baseline.

**Tasks**
- Create `tier2/` skeleton (core + domains) and move shared bits into place.
- Add tooling: `pytest`, `ruff`, `black`, `mypy` (optional), `pre-commit`.
- Pin dependencies; add `requirements.txt` or `pyproject.toml`.

**Acceptance**
- CI runs `pytest -q` and lints on push; green on current scripts.

**Deliverables**
- `tier2/` tree + CI config.

## Stage 1 — Extract the WDD core (1 day)
**Goal:** One engine for all domains.

**Tasks**
- `core/encoder.py`: SBERT load/cache + mapper unwrap.
- `core/prototypes.py`: prototype directions from `coef_`/centroids; optional prototype bank.
- `core/traces.py`: vectorized paths `(T,K,D)` + energies (`E_perp`) and optional probability-trace.
- `core/parser.py`: ngeodesic wrapper (matched filter, null‐Z, relative gates, peak ordering).
- `core/wdd.py`: orchestrator (`AuditRequest → AuditResult`) with modes: `pure | family | guided`.
- `core/utils.py`: normalize, batching, seeding, timers, small math helpers.

**Acceptance**
- Unit tests on synthetic channels: correct keep/order; shapes/dtypes match; deterministic with seed.

**Deliverables**
- `tier2/core/*.py`, with docstrings + `tests/test_core_*.py`.


## Stage 2 — Domain registries (ARC & DeFi) (1 day)
**Goal:** Thin domain layers that “register” families.

**Tasks**
- `domains/arc/families.py`: flip/rotate (live) + stubs for recolor/translate/crop; per-family thresholds; one-sibling-only where needed.
- `domains/defi/families.py`: map DeFi signal families to envelopes/thresholds.
- `config/arc.yaml`, `config/defi.yaml`: `T`, `alpha_max`, per-family `keep_frac`, `z_abs`, `mode`.
- `domains/*/audit.py`: small wrappers calling the core (`wdd_arc_audit`, `wdd_defi_audit`).

**Acceptance**
- Old notebooks call new wrappers and reproduce prior outputs (within tolerance).

**Deliverables**
- Two YAMLs, two `families.py`, two `audit.py`.


## Stage 3 — ARC arguments + executors (1–1.5 days)
**Goal:** WDD picks families; grid code resolves **arguments**.

**Tasks**
- `domains/arc/args.py`:
  - `infer_rotate_k`, `infer_recolor_map`, `infer_translate_dxdy`, `infer_crop_bbox` (and simple selectors for components).
- `domains/arc/exec.py`: NumPy executors for rotate/flip/recolor/translate/crop (and simple morphology).
- `domains/arc/solver.py`: synthesize hints → WDD → infer args → **replay-sanity**; canonicalize D4; cap sequence length `L_MAX = 3`.

**Acceptance**
- Unit tests: toy grids verify each extractor + executor; replay improves train loss per step.

**Deliverables**
- `args.py`, `exec.py`, `solver.py`; tests `test_arc_args.py`, `test_arc_solver.py`.


## Stage 4 — DeFi polish (½–1 day)
**Goal:** Match Tier-2 shape for DeFi.

**Tasks**
- Consolidate DeFi signal construction, envelopes, and thresholds in `families.py`.
- Ensure audit outputs parity with previous notebook (including debug traces).

**Acceptance**
- Golden tests comparing Tier-1 vs Tier-2 outputs on sample data.

**Deliverables**
- `test_defi_audit.py`, refreshed `wdd_defi_tier2_base.ipynb` using wrappers.


## Stage 5 — Config + ablations (½ day)
**Goal:** Switch-controlled behavior & reproducibility.

**Tasks**
- Implement `MODE` switches in core (pure/family/guided) and read from YAML.
- Seed control; deterministic runs; save debug dumps (envelopes, peaks, Z, gates).
- Ablations: pure ↔ family ↔ guided; rotate prob-trace on/off; flip hint prior on/off.

**Acceptance**
- Small ablation report showing expected deltas on a held-out set.

**Deliverables**
- `reports/arc_tier2_ablation.md` (brief table + notes).


## Stage 6 — Performance pass (½ day)
**Goal:** “No-time” runtime.

**Tasks**
- `float32` everywhere; vectorized `einsum` in traces; batch `predict_proba` calls.
- Skip families by text/prob gating; cache embeddings & prototypes.

**Acceptance**
- Runtime: ≤10ms per prompt for ~20 classes on laptop CPU (sanity metric).

**Deliverables**
- Profiling note in README; optimized `core/traces.py`.


## Stage 7 — Packaging & notebooks (½ day)
**Goal:** Tier-2 developer UX.

**Tasks**
- Minimal API examples in `examples/arc_tier2_demo.ipynb` and `examples/defi_tier2_demo.ipynb`.
- (Optional) CLI: `python -m tier2.domains.arc.solver --task_dir ...`

**Acceptance**
- One-click notebook that runs end-to-end with cached artifacts.

**Deliverables**
- Two example notebooks + (optional) CLI.


## Stage 8 — Documentation & “Definition of Done” (½ day)
**Goal:** Clear docs, ready to extend.

**Tasks**
- `README.md` with architecture diagram, modes, YAML fields, how-to add a family.
- Per-family cookbook snippet (ARC): envelope choice, thresholds, tests to copy.

**Acceptance**
- New dev can add a primitive in ≤30 minutes using the cookbook.

**Deliverables**
- `docs/`, updated repo README.

---

## Checklists

### Core DoD
- [ ] `AuditRequest/AuditResult` APIs stable.
- [ ] Unit tests: traces, parser, ordering, null gates.
- [ ] MODE flags honored end-to-end.
- [ ] Vectorized `(T,K,D)` implementation.

### ARC DoD (Tier-2)
- [ ] Families: `rotate`, `flip_h/v` (live) + stubs for `recolor`, `translate`, `crop`.
- [ ] `solver.py` flow: synthesize → WDD → args → replay verify → sequence (L≤3).
- [ ] Tests: rotate/recolor/translate/crop; sequence canonicalization; abstain path.
- [ ] Config in YAML; reproducible runs.

### DeFi DoD (Tier-2)
- [ ] Families registered; envelopes + thresholds in YAML.
- [ ] Audit parity with Tier-1 on sample data.
- [ ] Tests for at least two signal families.


---

## Cutover plan
1) Land Stages **1–2**; switch notebooks to `domains/*/audit.py` wrappers.
2) Land Stage **3** + tests; ARC Tier-2 solver replaces ad-hoc code.
3) Lock YAMLs, tag **v2.0.0-tier2**.
4) Only then add new ARC families (recolor/translate/crop first).