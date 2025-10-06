# Tasks: Preserve Algorithm Separation in Benchmark Aggregation

**Input**: Design documents from `/specs/142-aggregation-mixes-algorithms/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/, quickstart.md

## Execution Flow (main)
```
1. Load plan.md from feature directory
2. Load supporting design docs (research, data-model, contracts, quickstart)
3. Generate setup, test, implementation, integration, and polish tasks per feature scope
4. Enforce TDD ordering and constitution constraints
5. Assign [P] marker to tasks on independent files
6. Provide dependency guidance and parallel execution hints
```

## Phase 3.1: Setup
- [ ] **T001** Ensure benchmark dev environment ready: `uv sync`, submodules, and headless env vars (reference quickstart)  _(repo root)_
- [ ] **T002 [P]** Create custom exception skeleton `robot_sf/benchmark/errors.py::AggregationMetadataError` with docstring and export list  _(robot_sf/benchmark/errors.py)_

## Phase 3.2: Tests First (TDD)
- [ ] **T003 [P]** Contract test: orchestrator injects nested algo (`tests/benchmark/test_orchestrator_metadata.py::test_injects_nested_algo_metadata`)  
- [ ] **T004 [P]** Contract test: orchestrator raises on missing `algo` (`tests/benchmark/test_orchestrator_metadata.py::test_raises_on_missing_algo`)  
- [ ] **T005 [P]** Contract test: orchestrator logs warning on mismatch (`tests/benchmark/test_orchestrator_metadata.py::test_logs_warning_on_mismatch`)  
- [ ] **T006 [P]** Contract test: aggregation groups by nested key + fallback (`tests/benchmark/test_aggregation_algorithms.py::test_grouping_prefers_nested_algo`)  
- [ ] **T007 [P]** Contract test: aggregation warns on missing algorithms (`tests/benchmark/test_aggregation_algorithms.py::test_warns_and_flags_missing_algorithms`)  
- [ ] **T008 [P]** Contract test: aggregation raises `AggregationMetadataError` when metadata absent (`tests/benchmark/test_aggregation_algorithms.py::test_missing_algo_fields_raise`)  
- [ ] **T009 [P]** Integration smoke: quickstart workflow (`tests/integration/test_classic_benchmark_alg_grouping.py::test_smoke_aggregation_workflow`) verifying warning logging & `_meta`

## Phase 3.3: Core Implementation
- [ ] **T010** Implement `AggregationMetadataError` with helpful message + import in `robot_sf/benchmark/__init__.py`  
- [ ] **T011** Update orchestrator (`robot_sf/benchmark/full_classic/orchestrator.py`) to mirror `algo` into `scenario_params`, add structured logging, and raise `AggregationMetadataError` on invalid payloads  
- [ ] **T012** Ensure resume manifest handling skips duplicate injection (same file as T011)  
- [ ] **T013** Update aggregation fallback chain in `robot_sf/benchmark/aggregate.py` to prefer `scenario_params.algo`, then `algo`, else `fallback_group_by`; include `_meta.effective_group_key`  
- [ ] **T014** Add missing algorithm detection + `_meta.missing_algorithms` + warning emission in `robot_sf/benchmark/aggregate.py`  
- [ ] **T015** Extend `scripts/run_social_navigation_benchmark.py` aggregation path to pass expected algorithm set and propagate `_meta` diagnostics to output JSON

## Phase 3.4: Integration & Validation
- [ ] **T016** Update validation script (e.g., `scripts/validation/test_complete_simulation.sh` or relevant Python validation) to assert mirrored metadata exists  
- [ ] **T017 [P]** Add CHANGELOG entry documenting per-algorithm aggregation fix  
- [ ] **T018 [P]** Update `docs/benchmark.md` + related docs index references about algorithm grouping keys and warnings  
- [ ] **T019** Execute quickstart smoke steps (per quickstart.md) and capture results in progress notes  
- [ ] **T020** Run full pytest suite for touched modules (`uv run pytest tests/benchmark/test_orchestrator_metadata.py tests/benchmark/test_aggregation_algorithms.py tests/integration/test_classic_benchmark_alg_grouping.py`) and ensure green

## Phase 3.5: Polish
- [ ] **T021 [P]** Add regression guidance into `docs/dev/issues/142-aggregation-mixes-algorithms/design.md` or create summary note under feature docs (link from docs index)  
- [ ] **T022 [P]** Clean up any legacy JSONL fixtures or update test fixtures to include mirrored metadata  
- [ ] **T023** Final sweep: Loguru log level sanity, ensure no stray prints, and check lint (`uv run ruff check`)

## Dependencies & Ordering Notes
- T001 precedes all tasks.
- T002 (exception skeleton) required before aggregation/orchestrator tests compile; write tests (T003–T009) immediately after skeleton.
- Tests T003–T009 must exist and fail before implementing T010–T015.
- T010 feeds both orchestrator and aggregation implementations (T011–T015).
- Documentation/changelog tasks (T017, T018, T021) can run in parallel once functional changes are stable.
- Validation tasks (T019, T020) occur after implementations complete.
- Polish tasks (T021–T023) follow successful validation.

## Parallel Execution Examples
```
# After T001–T002, author contract tests in parallel:
/specs/.../tasks.md --run T003 T004 T005 T006 T007 T008 T009

# After implementation stabilizes, run docs tasks together:
/specs/.../tasks.md --run T017 T018 T021
```

## Validation Checklist
- [ ] All contract tests written before implementation
- [ ] Aggregation + orchestrator implementations respect new data model
- [ ] `_meta` diagnostics and warnings verified via integration smoke test
- [ ] Documentation and changelog updated
- [ ] Linting and pytest suites green
