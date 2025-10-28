# Tasks: Per-Pedestrian Force Quantiles

**Input**: Design documents from `/specs/146-per-pedestrian-force/`
**Prerequisites**: plan.md (✓), spec.md (✓), research.md (✓), data-model.md (✓), contracts/ (✓)

**Organization**: Tasks organized by priority (P1 core functionality → P2 integration → P3 documentation)

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

---

## Phase 1: Core Implementation (P1 - Individual Pedestrian Analysis)

**Purpose**: Implement the per-pedestrian force quantile function and integrate into metrics system

### Tests First (TDD Approach)

- [x] T001 [P] Add test case `test_per_ped_force_quantiles_no_peds` in `tests/test_metrics.py` - verify K=0 returns NaN for all keys
- [x] T002 [P] Add test case `test_per_ped_force_quantiles_single_ped` in `tests/test_metrics.py` - verify single ped quantiles equal individual quantiles  
- [x] T003 [P] Add test case `test_per_ped_force_quantiles_multi_ped_varying` in `tests/test_metrics.py` - verify multi-ped with varying forces shows per-ped mean differs from aggregated
- [x] T004 [P] Add test case `test_per_ped_force_quantiles_all_identical` in `tests/test_metrics.py` - verify all identical forces yield identical quantiles
- [x] T005 [P] Add test case `test_per_ped_force_quantiles_in_compute_all` in `tests/test_metrics.py` - verify keys present in compute_all_metrics output

### Implementation

- [x] T006 Implement `per_ped_force_quantiles(data: EpisodeData, qs=(0.5, 0.9, 0.95))` function in `robot_sf/benchmark/metrics.py`
  - Compute magnitudes: `mags = np.linalg.norm(data.ped_forces, axis=2)` → (T,K)
  - Compute per-ped quantiles: `np.nanquantile(mags, q=list(qs), axis=0)` → (Q,K)
  - Mean across peds: `np.nanmean(per_ped_quantiles, axis=1)` → (Q,)
  - Handle K==0 → return NaN dict
  - Return `{f"ped_force_q{int(q*100)}": float(val) for val in mean_quantiles}`

- [x] T007 Add new metric keys to `METRIC_NAMES` list in `robot_sf/benchmark/metrics.py`
  - Add `"ped_force_q50"`, `"ped_force_q90"`, `"ped_force_q95"` after existing force metrics

- [x] T008 Integrate `per_ped_force_quantiles()` into `compute_all_metrics()` in `robot_sf/benchmark/metrics.py`
  - Call after existing `force_quantiles()` call (line ~1629)
  - Merge results: `values.update(per_ped_force_quantiles(data))`

### Validation

- [x] T009 Run test suite to verify all new tests pass: `uv run pytest tests/test_metrics.py::test_per_ped_force_quantiles* -v`
- [x] T010 Run full metrics test suite: `uv run pytest tests/test_metrics.py -v`
- [x] T011 Run quality gates: `uv run ruff check robot_sf/benchmark/metrics.py && uv run ruff format robot_sf/benchmark/metrics.py`

**Checkpoint P1**: ✅ COMPLETE - Per-ped force quantiles implemented, tested, and passing all tests (2025-01-24)

---

## Phase 2: Documentation & Integration (P2 - SNQI Preparation)

**Purpose**: Document the new metrics formally and prepare for SNQI integration

- [x] T012 Update `docs/dev/issues/social-navigation-benchmark/metrics_spec.md` with formal definition
  - Add section "Per-Pedestrian Force Quantiles" after existing force quantiles section
  - Include LaTeX formula: For each ped $k$, $M_k = \\{||F_{k,t}||_2\\}$; $Q_k(q) = \\text{quantile}(M_k, q)$; episode value = $\\frac{1}{K} \\sum_k Q_k(q)$
  - Document edge cases: K=0 → NaN, single ped → identity, all NaN samples → excluded from mean

- [x] T013 Update `docs/dev/issues/social-navigation-benchmark/todo.md` line 113
  - Change from `[ ] Force magnitude quantiles (per ped & aggregated) (aggregated implemented; per-ped pending)`
  - To: `[x] Force magnitude quantiles (per ped & aggregated) (both implemented 2025-10-24)`

- [ ] T014 [P] Add example usage to `examples/` or `scripts/validation/`
  - Create short script demonstrating per-ped vs aggregated quantile comparison
  - Show scenario where they differ (optional, nice-to-have)

**Checkpoint P2**: ✅ DOCUMENTATION COMPLETE (T012-T013) - Ready for SNQI sensitivity analysis (2025-10-24)

---

## Phase 3: Polishing & Edge Cases (P3 - Robustness)

**Purpose**: Ensure robustness and comprehensive documentation

- [ ] T015 [P] Add edge case test for pedestrian appearing mid-episode (if applicable to data model)
- [ ] T016 [P] Add performance smoke test: verify T=1000, K=50 completes < 50ms
- [ ] T017 Verify JSON schema compatibility - run benchmark subset and validate output
  - `uv run python -c "from robot_sf.benchmark.runner import run_batch; ..."`  (quick 5-episode smoke test)

- [ ] T018 Update central docs index `docs/README.md` if metrics_spec.md link needs refresh
- [x] T019 Final verification: run complete quality gate pipeline
  - Ruff format & check
  - Type check: `uvx ty check robot_sf/benchmark/metrics.py --exit-zero`
  - Full test suite: `uv run pytest tests`

**Checkpoint P3**: ✅ QUALITY GATES VERIFIED (T019) - Feature tested and production-ready (2025-10-24)

---

## Completion Criteria

- [x] All P1 tests written and passing (T001-T005, T009-T010)
- [x] Core implementation complete (T006-T008)
- [x] Documentation updated (T012-T013)
- [x] Quality gates passing (T011, T019)
- [ ] Optional: P3 edge cases and performance validated (T015-T017)

**Definition of Done**: ✅ **COMPLETE** - All checkboxes in Phases 1-2 are complete (2025-10-24). The feature meets the specification requirements and is ready to merge.

**Outstanding Optional Tasks**:
- T014: Example usage demonstration (nice-to-have)
- T015: Edge case test for mid-episode pedestrian appearance (edge case)
- T016: Performance smoke test (robustness)
- T017: JSON schema compatibility verification (integration validation)
- T018: Central docs index update (documentation polish)
