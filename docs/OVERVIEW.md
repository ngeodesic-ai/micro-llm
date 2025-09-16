# Proposed `micro-lm` Directory Structure

This document outlines the **proposed directory structure** for the `micro-lm` project.  
It consolidates the DeFi Tier-1 milestones (M1–M11) and extends support to ARC primitives,  
demonstrating that `micro-lm` is a general scientific framework.

---

## Master File Structure (Full Scope)

```
micro-lm/
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── LICENSE
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   ├── global.yml
│   ├── defi/
│   │   ├── tier1_mvp.json
│   │   └── context.json
│   └── arc/
│       └── demo.json
├── src/
│   └── micro_lm/
│       ├── __init__.py
│       ├── cli.py
│       ├── core/
│       │   ├── runner.py
│       │   ├── mapper_api.py
│       │   ├── rails_shim.py
│       │   ├── bench_io.py
│       │   └── metrics.py
│       ├── domains/
│       │   ├── defi/
│       │   │   ├── labels.py
│       │   │   ├── verify.py
│       │   │   ├── mapper.py
│       │   │   └── benches/
│       │   │       ├── mapper_bench.py
│       │   │       ├── rails_bench.py
│       │   │       └── e2e_bench.py
│       │   └── arc/
│       │       ├── labels.py
│       │       ├── verify.py
│       │       ├── mapper.py
│       │       └── benches/
│       │           ├── mapper_bench.py
│       │           ├── rails_bench.py
│       │           └── e2e_bench.py
│       └── adapters/
│           └── README.md
├── tests/
│   ├── unit/
│   │   ├── test_core_mapper_api.py
│   │   ├── test_core_rails_shim.py
│   │   ├── test_defi_verify.py
│   │   └── test_arc_verify.py
│   ├── smoke/
│   │   ├── test_defi_mapper_bench_fast.py
│   │   ├── test_defi_rails_bench_fast.py
│   │   ├── test_defi_e2e_bench_fast.py
│   │   ├── test_arc_mapper_bench_fast.py
│   │   ├── test_arc_rails_bench_fast.py
│   │   └── test_arc_e2e_bench_fast.py
│   └── fixtures/
│       ├── defi/
│       │   ├── defi_mapper_5k_prompts.jsonl
│       │   ├── defi_mapper_labeled_5k.csv
│       │   └── m8_per_class_thresholds.json
│       └── arc/
│           ├── arc_prompts.jsonl
│           ├── arc_labeled.csv
│           └── arc_mapper.joblib
├── examples/
│   ├── quickstart_defi.py
│   ├── quickstart_arc.py
│   └── colab_micro_lm_tier1.ipynb
├── docs/
│   ├── ARCHITECTURE.md
│   ├── BENCHMARKS.md
│   └── API.md
├── scripts/
│   └── milestones/
│       ├── defi_milestone1.py … defi_milestone11.py
│       └── README_MILESTONES.md
└── .artifacts/
    ├── defi/mapper_bench/
    ├── defi/rails_bench/
    ├── defi/e2e_bench/
    ├── arc/mapper_bench/
    ├── arc/rails_bench/
    └── arc/e2e_bench/
```

---

## Folder & File Descriptions

### Root
- **`pyproject.toml`** — Build configuration and dependencies.
- **`README.md`** — Project overview and quickstart guide.
- **`CHANGELOG.md`** — Release history and version notes.
- **`LICENSE`** — Licensing information.

### `.github/workflows/`
- **`ci.yml`** — Continuous integration workflow (lint, tests, smoke benches).

### `configs/`
- **`global.yml`** — Shared configuration (logging, seeds, fast mode).
- **`defi/`** — Policy and context for DeFi Tier-1.
- **`arc/`** — Configs for ARC domain demos.

### `src/micro_lm/`
Core package.

- **`__init__.py`** — Exports public API.
- **`cli.py`** — Entry point: `micro-lm <domain> <bench>`.
- **`core/`** — Domain-agnostic utilities (runner, mapper API, rails shim, metrics, I/O).
- **`domains/defi/`** — DeFi-specific primitives, mapper, verifiers, and benchmarks.
- **`domains/arc/`** — ARC-specific primitives, mapper, verifiers, and benchmarks.
- **`adapters/`** — Placeholder for future domains.

### `tests/`
- **`unit/`** — Fine-grained logic tests (mapper API, rails shim, verifiers).
- **`smoke/`** — Fast benchmarks to validate integration.
- **`fixtures/`** — Data fixtures for DeFi and ARC benchmarks.

### `examples/`
- **`quickstart_defi.py`** — Demonstrates DeFi happy path + edge case.
- **`quickstart_arc.py`** — Demonstrates ARC primitives pipeline.
- **`colab_micro_lm_tier1.ipynb`** — Notebook demo.

### `docs/`
- **`ARCHITECTURE.md`** — High-level system design (core + domains).
- **`BENCHMARKS.md`** — Explains three breakpoints: mapper-only, rails-only, end-to-end.
- **`API.md`** — Public API reference.

### `scripts/milestones/`
- Historical milestone scripts (M1–M11).  
- **`README_MILESTONES.md`** — Marks these as reference-only.

### `.artifacts/`
- Stores run outputs (summaries, reports, trained mappers).  
- Structured by domain (`defi/`, `arc/`) and bench type (mapper, rails, e2e).

---

## Key Principles

- **Domain-agnostic core** (`core/`) + **pluggable adapters** (`domains/`).  
- **Three demoable breakpoints** in both domains: mapper-only, rails-only, end-to-end.  
- **Historical milestones preserved** under `scripts/` for provenance.  
- **Stakeholder clarity**: shows micro-lm is a framework, not just ARC or DeFi.

