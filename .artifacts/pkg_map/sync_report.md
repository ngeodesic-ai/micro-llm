# micro_lm ⇄ notebooks sync report

## Summary

- Notebook-only symbols (candidates to promote): **101**
- Package-only symbols: **97**
- Facade export count (__all__): **2**
- Import errors: **0**

## 1) Promote from notebooks → package

These appear only in notebooks (likely core WDD/ARC/DeFi logic). Move them into `domains/<domain>/wdd_runtime.py` and wire via `core.audit.run_families`.

- [ ] `analytic_core_template`
- [ ] `build_H_E_from_traces`
- [ ] `build_pca_channels`
- [ ] `build_priors_feature_MFpeak`
- [ ] `build_prototypes`
- [ ] `mapper_predict_scores`
- [ ] `matched_filter_deposit_score`
- [ ] `matched_filter_scores`
- [ ] `predict_scores`
- [ ] `wdd_null_scores`
- [ ] `wdd_score_sequence`

## 2) Facade gaps (public but not re-exported)

For notebook stability, re-export key symbols in `micro_lm/__init__.py` so notebooks import the facade only.
- [ ] `ArtifactWriter`
- [ ] `AuditRequest`
- [ ] `AuditResult`
- [ ] `Context`
- [ ] `ContextAdapter`
- [ ] `Counter`
- [ ] `Detector`
- [ ] `Emb`
- [ ] `Enum`
- [ ] `ExecPlanner`
- [ ] `FamilySpec`
- [ ] `IntentMapper`
- [ ] `IntentResult`
- [ ] `JoblibMapper`
- [ ] `JoblibMapperConfig`
- [ ] `MapOut`
- [ ] `MapperAPI`
- [ ] `Mode`
- [ ] `Path`
- [ ] `Peak`
- [ ] `Plan`
- [ ] `PlanStep`
- [ ] `Protocol`
- [ ] `Rails`
- [ ] `RulePlanner`
- [ ] `RunInputs`
- [ ] `SBertMapper`
- [ ] `SbertEncoder`
- [ ] `SentenceTransformer`
- [ ] `SimpleContextAdapter`
- [ ] `Verify`
- [ ] `WordMapMapper`
- [ ] `apply_crop_bbox`
- [ ] `apply_pca_prior`
- [ ] `apply_recolor_map`
- [ ] `apply_rotate_k`
- [ ] `apply_translate`
- [ ] `arc_family_registry`
- [ ] `assert_deterministic`
- [ ] `build_term_vectors_and_protos`
- [ ] `check_hf`
- [ ] `check_ltv`
- [ ] `check_oracle`
- [ ] `circshift`
- [ ] `contextmanager`
- [ ] `dataclass`
- [ ] `decide_from_scores`
- [ ] `default_anchors`
- [ ] `defi_audit`
- [ ] `discover_context`
- [ ] `discover_policy`
- [ ] `encode_batch`
- [ ] `execute_arc`
- [ ] `extract_args`
- [ ] `extract_crop_bbox`
- [ ] `extract_recolor_map`
- [ ] `extract_rotate_k`
- [ ] `extract_translate`
- [ ] `f1_micro`
- [ ] `field`
- [ ] `flip_h`
- [ ] `flip_v`
- [ ] `from_centroids`
- [ ] `from_linear_coef`
- [ ] `get_audit_backend`
- [ ] `get_domain_runner`
- [ ] `halfmax_window`
- [ ] `import_module`
- [ ] `install`
- [ ] `kaiser_window`
- [ ] `l2_normalize`
- [ ] `load_domain_config`
- [ ] `load_pca_prior`
- [ ] `lru_cache`
- [ ] `main`
- [ ] `make_rng`
- [ ] `map_prompt_arc`
- [ ] `matched_filter_scores_all`
- [ ] `nxcorr`
- [ ] `pca_audit`
- [ ] `plan_arc`
- [ ] `register_arc`
- [ ] `replay`
- [ ] `rotate_ccw`
- [ ] `rotate_cw`
- [ ] `run_arc`
- [ ] `run_audit`
- [ ] `run_families`
- [ ] `run_wdd`
- [ ] `solve_arc_io`
- [ ] `span_similarity_max`
- [ ] `spans_all`
- [ ] `spans_gold_only`
- [ ] `synth_traces`
- [ ] `timed`
- [ ] `verify_action_local`
- [ ] `wdd_arc_audit`
- [ ] `wdd_audit`
- [ ] `wdd_defi_audit`
- [ ] `zscore`

## 3) Potential dead code (notebooks don’t call these & not in facade)

- [ ] `ArtifactWriter`
- [ ] `AuditRequest`
- [ ] `AuditResult`
- [ ] `Context`
- [ ] `ContextAdapter`
- [ ] `Counter`
- [ ] `Detector`
- [ ] `Enum`
- [ ] `ExecPlanner`
- [ ] `FamilySpec`
- [ ] `IntentMapper`
- [ ] `IntentResult`
- [ ] `JoblibMapper`
- [ ] `JoblibMapperConfig`
- [ ] `MapOut`
- [ ] `MapperAPI`
- [ ] `Mode`
- [ ] `Path`
- [ ] `Peak`
- [ ] `Plan`
- [ ] `PlanStep`
- [ ] `Protocol`
- [ ] `Rails`
- [ ] `RulePlanner`
- [ ] `RunInputs`
- [ ] `SBertMapper`
- [ ] `SbertEncoder`
- [ ] `SentenceTransformer`
- [ ] `SimpleContextAdapter`
- [ ] `Verify`
- [ ] `WordMapMapper`
- [ ] `apply_crop_bbox`
- [ ] `apply_pca_prior`
- [ ] `apply_recolor_map`
- [ ] `apply_rotate_k`
- [ ] `apply_translate`
- [ ] `arc_family_registry`
- [ ] `assert_deterministic`
- [ ] `build_term_vectors_and_protos`
- [ ] `check_hf`
- [ ] `check_ltv`
- [ ] `check_oracle`
- [ ] `circshift`
- [ ] `contextmanager`
- [ ] `dataclass`
- [ ] `decide_from_scores`
- [ ] `default_anchors`
- [ ] `defi_audit`
- [ ] `discover_context`
- [ ] `discover_policy`
- [ ] `encode_batch`
- [ ] `execute_arc`
- [ ] `extract_args`
- [ ] `extract_crop_bbox`
- [ ] `extract_recolor_map`
- [ ] `extract_rotate_k`
- [ ] `extract_translate`
- [ ] `f1_micro`
- [ ] `field`
- [ ] `from_centroids`
- [ ] `from_linear_coef`
- [ ] `get_audit_backend`
- [ ] `get_domain_runner`
- [ ] `halfmax_window`
- [ ] `import_module`
- [ ] `install`
- [ ] `l2_normalize`
- [ ] `load_domain_config`
- [ ] `load_pca_prior`
- [ ] `lru_cache`
- [ ] `main`
- [ ] `make_rng`
- [ ] `map_prompt_arc`
- [ ] `matched_filter_scores_all`
- [ ] `nxcorr`
- [ ] `pca_audit`
- [ ] `plan_arc`
- [ ] `register_arc`
- [ ] `replay`
- [ ] `rotate_ccw`
- [ ] `rotate_cw`
- [ ] `run_arc`
- [ ] `run_audit`
- [ ] `run_families`
- [ ] `run_wdd`
- [ ] `solve_arc_io`
- [ ] `span_similarity_max`
- [ ] `spans_all`
- [ ] `spans_gold_only`
- [ ] `synth_traces`
- [ ] `timed`
- [ ] `verify_action_local`
- [ ] `wdd_audit`
- [ ] `wdd_defi_audit`
- [ ] `zscore`

## 4) Modules with import errors


## 5) Highlights (Tier-2 critical paths)

### defi_families_wdd
```json
{
  "module": "micro_lm.domains.defi.families_wdd",
  "file": "/Users/ian_moore/repos/micro-lm/src/micro_lm/domains/defi/families_wdd.py",
  "public": [
    "FamilySpec",
    "defi_family_registry"
  ]
}
```
### legacy_imports_shim
```json
{
  "module": "micro_lm.compat.legacy_imports",
  "file": "/Users/ian_moore/repos/micro-lm/src/micro_lm/compat/legacy_imports.py",
  "public": [
    "install"
  ]
}
```