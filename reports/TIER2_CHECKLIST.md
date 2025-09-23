# Tier-2 Refactor --- Checkpoint Summary (Stages 0--7)

This document records the closure status for **Stages 0--7** of the
Tier-2 refactor.\
The goal of Tier-2 is to harden **micro-lm** into a deterministic,
auditable, and extensible package with both DeFi + ARC domains.

------------------------------------------------------------------------

## Stage 0 --- Repo Hygiene & Freeze

**Goal:** Lock Tier-1 milestones into read-only state; prepare clean
slate for Tier-2.\
**Work:** - Archived M1--M11 into `scripts/milestones/`. - Added
`README_MILESTONES.md` (reference only).\
- Created branch `refactor/tier2`.\
- Skeleton CI pipeline in `.github/workflows/ci.yml`.\
- Global config defaults in `configs/global.yml`.

**Outputs:**\
- Stable, reproducible repo baseline for refactor.

------------------------------------------------------------------------

## Stage 1 --- Core Runner Shim

**Goal:** Establish a canonical `run_micro` entrypoint.\
**Work:** - Implemented `runner.py` with safe MapperShim fallback.\
- Clear fail-loud behavior on abstain / low-confidence.\
- Integrated baseline rails (stage-11).

**Outputs:**\
- `run_micro(domain, prompt, …)` as the universal invocation surface.\
- Deterministic harness behavior independent of legacy milestone
scripts.

------------------------------------------------------------------------

## Stage 2 --- DeFi Adapter Lift

**Goal:** Port DeFi logic into Tier-2 package structure.\
**Work:** - Moved DeFi domain to `src/micro_lm/domains/defi/`.\
- Integrated mapper, priors, and rails under `run_micro`.\
- Verified against DeFi smoke benchmarks.

**Outputs:**\
- Functional DeFi adapter, matching Tier-1 accuracy.

------------------------------------------------------------------------

## Stage 3 --- ARC Adapter Lift

**Goal:** Port ARC (grid tasks) into Tier-2 package structure.\
**Work:** - Added ARC domain at `src/micro_lm/domains/arc/`.\
- Implemented `run_micro` pathway for grid prompts.\
- Validated via ARC smoke tests.

**Outputs:**\
- Functional ARC adapter, ready for Tier-2 audit integration.

------------------------------------------------------------------------

## Stage 4 --- Rails Integration

**Goal:** Generalize rails logic into Tier-2 core.\
**Work:** - Imported stage-11 rails into `src/micro_lm/core/rails/`.\
- Added support for deterministic execution + gating.\
- Confirmed parity with Tier-1 rails benchmarks.

**Outputs:**\
- Unified rails tooling across DeFi + ARC.

------------------------------------------------------------------------

## Stage 5 --- Golden Harness Integration

**Goal:** Introduce golden harness regression testing inside Tier-2.\
**Work:** - Ported harness from Tier-1 closure steps.\
- CLI runner with `--debug` flag.\
- Safe mapper shim fallback for OK paths.\
- Smoke suite for deposit/swap and edge reject cases.

**Outputs:**\
- Harness ensures Tier-2 changes don't regress core behavior.

------------------------------------------------------------------------

## Stage 6 --- Profiling Layer

**Goal:** Add optional profiling for traceability & audit.\
**Work:** - Implemented `profile=True` policy flag.\
- Profiles written to `.artifacts/{domain}/profile/{ts}/`.\
- `profile.json` (step-by-step) + `topline.json` (summary).\
- Tests validate profile creation, contents, and topline metrics.

**Outputs:**\
- Full audit trail of parse → map → rail phases.\
- Negligible runtime overhead when disabled.

------------------------------------------------------------------------

## Stage 7 --- PCA Priors & Audit Hardening

**Goal:** Strengthen audit backends with PCA priors.\
**Work:** - Moved PCA tooling into `src/micro_lm/core/audit/`.\
- Implemented `load_pca_prior`, `apply_pca_prior` with validation.\
- Tests cover round-trip, centering, dimension reduction, dtype
consistency, and invalid-shape rejection.\
- Unified with WDD audit backend.\
- Ensured full DeFi + ARC integration under `run_micro`.

**Outputs:**\
- Deterministic, test-backed PCA prior tooling.\
- Audit layer hardened with production-ready components.\
- All stages (0--7) green under pytest.

------------------------------------------------------------------------

## Status

-   ✅ Stages 0--7 complete.\
-   🔜 Stage 8: Consolidation (final Tier-2 close-out).
    -   Merge DeFi + ARC adapters into one package.\
    -   Ensure unified audit/rails/golden harness stack.\
    -   Deliver Tier-2 reference benchmarks.
