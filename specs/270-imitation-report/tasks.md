# Tasks: Automated Research Reporting for Imitation Learning

**Feature**: 270-imitation-report  
**Branch**: `270-imitation-report`  
**Date**: 2025-11-21

**Input**: Design documents from `/specs/270-imitation-report/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Integration tests included for report generation; unit tests for core components.

**Organization**: Tasks grouped by user story to enable independent implementation and testing.

---

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

Following existing robot_sf structure:
- Library code: `robot_sf/research/`
- Scripts: `scripts/research/`, `scripts/tools/`
- Tests: `tests/research/`
- Output: `output/research_reports/`
- Fixtures: `tests/fixtures/minimal_manifests/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and module structure

- [X] T001 Create research module directory structure at `robot_sf/research/`
- [X] T002 Create `robot_sf/research/__init__.py` with public API exports
- [X] T003 [P] Create scripts directory at `scripts/research/`
- [X] T004 [P] Create tests directory at `tests/research/`
- [X] T005 [P] Create test fixtures directory at `tests/fixtures/minimal_manifests/`
- [X] T006 Add scipy, matplotlib, pandas to pyproject.toml dependencies if missing
- [X] T007 Update `docs/README.md` to link to research reporting documentation

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T008 Implement base exception classes in `robot_sf/research/exceptions.py` (ReportGenerationError, ValidationError)
- [ ] T009 [P] Create reproducibility metadata collector in `robot_sf/research/metadata.py` (git hash, packages, hardware)
- [ ] T010 [P] Implement artifact path helpers in `robot_sf/research/artifact_paths.py` (ensure_report_tree, get_output_paths)
- [ ] T011 Create base logger configuration in `robot_sf/research/logging_config.py` using Loguru
- [ ] T012 Implement schema loader utility in `robot_sf/research/schema_loader.py` for JSON schema validation
- [ ] T013 [P] Create test fixture generator in `tests/fixtures/minimal_manifests/generator.py` (baseline + pretrained manifests)
- [ ] T014 Add schema to `robot_sf/benchmark/schemas/report_metadata.schema.v1.json` from contracts/
- [ ] T015 Document report output structure in `docs/research_reporting.md` with examples

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Generate End-to-End Research Report (Priority: P1) 🎯 MVP

**Goal**: Automatically generate structured research report (Markdown + optional LaTeX) from pipeline runs with metrics, figures, and reproducibility metadata.

**Independent Test**: Execute full pipeline with multi-seed settings; verify report directory contains `report.md`, figures/, data/, metadata.json.

### Implementation for User Story 1

- [X] T016 [P] [US1] Implement metric aggregation engine in `robot_sf/research/aggregation.py` (aggregate_metrics, bootstrap_ci)
- [X] T017 [P] [US1] Implement statistical analyzer in `robot_sf/research/statistics.py` (paired_t_test, cohen_d, evaluate_hypothesis)
- [X] T018 [P] [US1] Implement figure generator in `robot_sf/research/figures.py` (learning_curve, sample_efficiency_bar, distribution_plots)
- [X] T019 [P] [US1] Create report template renderer in `robot_sf/research/report_template.py` (MarkdownReportRenderer class)
- [X] T020 [US1] Implement orchestrator in `robot_sf/research/orchestrator.py` (ReportOrchestrator.generate_report method)
- [X] T021 [US1] Create CLI script at `scripts/research/generate_report.py` with argparse interface
- [X] T022 [US1] Add LaTeX export support in `robot_sf/research/report_template.py` (export_latex method)
- [X] T023 [US1] Implement JSON/CSV data export in `robot_sf/research/aggregation.py` (export_metrics_json, export_metrics_csv)
- [X] T024 [US1] Add auto-populated Abstract section logic in `robot_sf/research/report_template.py` (_render_abstract method)
- [X] T025 [US1] Implement figure caption generator in `robot_sf/research/figures.py` (_generate_caption method)
- [X] T026 [US1] Add reproducibility metadata integration in `robot_sf/research/orchestrator.py` (collect_metadata step)
- [X] T027 [US1] Create validation script at `scripts/tools/validate_report.py` (check schema, files, completeness)

### Tests for User Story 1

- [X] T028 [P] [US1] Unit test metric aggregation in `tests/research/test_aggregation.py` (test_aggregate_metrics, test_bootstrap_ci)
- [X] T029 [P] [US1] Unit test statistical analysis in `tests/research/test_statistics.py` (test_paired_t_test, test_cohen_d, test_hypothesis_eval)
- [X] T030 [P] [US1] Unit test figure generation in `tests/research/test_figures.py` (test_learning_curve, test_sample_efficiency)
- [X] T031 [US1] Integration test end-to-end report in `tests/research/test_integration_report.py` (test_generate_full_report)
- [X] T032 [US1] Smoke test CLI invocation in `tests/research/test_cli.py` (test_generate_report_cli)

**Checkpoint**: User Story 1 complete - can generate research reports from pipeline runs

---

## Phase 4: User Story 2 - Multi-Seed Metrics Aggregation (Priority: P2)

**Goal**: Orchestrate multiple random seeds per condition and aggregate per-seed metrics into summary statistics with bootstrap CIs.

**Independent Test**: Run experiment with N seeds; verify aggregated output includes mean, median, p95, CI bounds for all metrics.

### Implementation for User Story 2

- [ ] T033 [P] [US2] Add seed orchestration logic in `robot_sf/research/orchestrator.py` (orchestrate_multi_seed method)
- [ ] T034 [US2] Implement per-seed metric extraction in `robot_sf/research/aggregation.py` (extract_seed_metrics from tracker manifests)
- [ ] T035 [US2] Add completeness scoring in `robot_sf/research/aggregation.py` (compute_completeness_score method)
- [ ] T036 [US2] Implement failed seed handling in `robot_sf/research/orchestrator.py` (handle_failed_seeds method)
- [ ] T037 [US2] Add seed failure logging in `robot_sf/research/logging_config.py` (log_seed_failure utility)
- [ ] T038 [US2] Create seed summary table renderer in `robot_sf/research/report_template.py` (_render_seed_summary method)
- [ ] T039 [US2] Add tracker manifest integration in `robot_sf/research/metadata.py` (parse_tracker_manifest method)

### Tests for User Story 2

- [ ] T040 [P] [US2] Unit test seed orchestration in `tests/research/test_orchestrator.py` (test_multi_seed_execution)
- [ ] T041 [P] [US2] Unit test completeness scoring in `tests/research/test_aggregation.py` (test_completeness_score)
- [ ] T042 [US2] Integration test multi-seed aggregation in `tests/research/test_integration_report.py` (test_multi_seed_report)
- [ ] T043 [US2] Test failed seed graceful degradation in `tests/research/test_orchestrator.py` (test_partial_seed_failure)

**Checkpoint**: User Story 2 complete - multi-seed experiments aggregate correctly with completeness tracking

---

## Phase 5: User Story 3 - Statistical Analysis & Figures (Priority: P3)

**Goal**: Generate standardized figures (learning curves, distributions, effect size summaries) and statistical tests (p-values, effect sizes, CIs).

**Independent Test**: Run figure generator on aggregated metrics; verify presence of required plots and stats table in report.

### Implementation for User Story 3

- [ ] T044 [P] [US3] Implement learning curve plot in `robot_sf/research/figures.py` (plot_learning_curve method)
- [ ] T045 [P] [US3] Implement sample efficiency bar chart in `robot_sf/research/figures.py` (plot_sample_efficiency method)
- [ ] T046 [P] [US3] Implement distribution comparison plots in `robot_sf/research/figures.py` (plot_distributions method)
- [ ] T047 [P] [US3] Implement effect size summary plot in `robot_sf/research/figures.py` (plot_effect_sizes method)
- [ ] T048 [P] [US3] Implement improvement summary plot in `robot_sf/research/figures.py` (plot_improvement_summary method)
- [ ] T049 [US3] Add matplotlib configuration in `robot_sf/research/figures.py` (configure_matplotlib_backend method)
- [ ] T050 [US3] Implement PDF + PNG dual export in `robot_sf/research/figures.py` (save_figure method)
- [ ] T051 [US3] Add statistical test result formatting in `robot_sf/research/statistics.py` (format_test_results method)
- [ ] T052 [US3] Create statistical summary table in `robot_sf/research/report_template.py` (_render_stats_table method)
- [ ] T053 [US3] Add graceful handling for <2 seeds in `robot_sf/research/statistics.py` (validate_sample_size method)

### Tests for User Story 3

- [ ] T054 [P] [US3] Unit test learning curve generation in `tests/research/test_figures.py` (test_learning_curve_plot)
- [ ] T055 [P] [US3] Unit test sample efficiency plot in `tests/research/test_figures.py` (test_sample_efficiency_plot)
- [ ] T056 [P] [US3] Unit test distribution plots in `tests/research/test_figures.py` (test_distribution_plots)
- [ ] T057 [P] [US3] Unit test effect size calculations in `tests/research/test_statistics.py` (test_effect_size_computation)
- [ ] T058 [US3] Integration test figure suite generation in `tests/research/test_integration_report.py` (test_full_figure_suite)
- [ ] T059 [US3] Test headless figure generation in `tests/research/test_figures.py` (test_headless_matplotlib)

**Checkpoint**: User Story 3 complete - publication-quality figures and statistical summaries generated

---

## Phase 6: User Story 4 - Ablation & Hypothesis Evaluation (Priority: P4)

**Goal**: Define ablation parameters and evaluate hypothesis that pre-training reduces PPO timesteps by target percentage.

**Independent Test**: Execute ablation matrix; verify report includes comparison table with pass/fail labels.

### Implementation for User Story 4

- [ ] T060 [P] [US4] Implement ablation matrix generator in `robot_sf/research/orchestrator.py` (AblationOrchestrator class)
- [ ] T061 [P] [US4] Implement hypothesis evaluator in `robot_sf/research/statistics.py` (HypothesisEvaluator class)
- [ ] T062 [US4] Add ablation config parser in `robot_sf/research/orchestrator.py` (parse_ablation_config method)
- [ ] T063 [US4] Create ablation comparison table renderer in `robot_sf/research/report_template.py` (_render_ablation_table method)
- [ ] T064 [US4] Implement sensitivity analysis plots in `robot_sf/research/figures.py` (plot_sensitivity method)
- [ ] T065 [US4] Add hypothesis result exporter in `robot_sf/research/statistics.py` (export_hypothesis_json method)
- [ ] T066 [US4] Create ablation CLI script at `scripts/research/compare_ablations.py` with argparse
- [ ] T067 [US4] Add incomplete variant handling in `robot_sf/research/orchestrator.py` (handle_incomplete_variants method)
- [ ] T068 [US4] Implement threshold comparison logic in `robot_sf/research/statistics.py` (compare_to_threshold method)

### Tests for User Story 4

- [ ] T069 [P] [US4] Unit test ablation matrix generation in `tests/research/test_orchestrator.py` (test_ablation_matrix)
- [ ] T070 [P] [US4] Unit test hypothesis evaluation in `tests/research/test_statistics.py` (test_hypothesis_evaluator)
- [ ] T071 [P] [US4] Unit test sensitivity plots in `tests/research/test_figures.py` (test_sensitivity_analysis)
- [ ] T072 [US4] Integration test ablation report in `tests/research/test_integration_report.py` (test_ablation_study_report)
- [ ] T073 [US4] Test incomplete variant handling in `tests/research/test_orchestrator.py` (test_incomplete_ablation)

**Checkpoint**: All user stories complete - full research reporting pipeline operational

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T074 [P] Add comprehensive docstrings to all `robot_sf/research/` modules following dev_guide.md standards
- [ ] T075 [P] Update `docs/research_reporting.md` with architecture diagrams (optional mermaid)
- [ ] T076 [P] Add examples to `examples/advanced/17_research_report_demo.py` demonstrating programmatic API
- [ ] T077 Add performance smoke test to `scripts/validation/performance_smoke_test.py` (target <120s for 3-seed)
- [ ] T078 [P] Add telemetry section rendering in `robot_sf/research/report_template.py` (_render_telemetry method)
- [ ] T079 Validate against success criteria SC-001 through SC-014 in `tests/research/test_success_criteria.py`
- [ ] T080 Run quickstart.md validation via `uv run python scripts/research/generate_report.py --help`
- [ ] T081 [P] Add CHANGELOG.md entry documenting new research reporting feature
- [ ] T082 Update `.github/copilot-instructions.md` Recent Changes section (if not auto-updated)
- [ ] T083 Create example ablation config at `configs/research/example_ablation.yaml`
- [ ] T084 [P] Add schema validation tests in `tests/research/test_schemas.py` (validate against JSON schemas)
- [ ] T085 Code quality pass: run Ruff, pylint, type check on `robot_sf/research/`
- [ ] T086 Create minimal end-to-end smoke script at `scripts/validation/test_research_report_smoke.sh`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phases 3-6)**: All depend on Foundational phase completion
  - User stories CAN proceed in parallel (if staffed appropriately)
  - OR sequentially in priority order (P1 → P2 → P3 → P4)
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories ✅ **MVP SCOPE**
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Integrates with US1 orchestrator but independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Uses US1 aggregation outputs but independently testable
- **User Story 4 (P4)**: Can start after Foundational (Phase 2) - Extends US1-US3 but independently testable

### Within Each User Story

- Implementation tasks before tests (tests validate implementation)
- Models/utilities before services
- Services before orchestration
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

**Setup Phase (Phase 1)**:
- T003, T004, T005 (directory creation) can run in parallel

**Foundational Phase (Phase 2)**:
- T009, T010, T013 can run in parallel (independent utilities)

**User Story 1 (Phase 3)**:
- T016, T017, T018, T019 can run in parallel (independent modules)
- T028, T029, T030 tests can run in parallel

**User Story 2 (Phase 4)**:
- T040, T041 tests can run in parallel

**User Story 3 (Phase 5)**:
- T044, T045, T046, T047, T048 (all figure types) can run in parallel
- T054, T055, T056, T057 tests can run in parallel

**User Story 4 (Phase 6)**:
- T060, T061 can run in parallel (orchestrator + evaluator)
- T069, T070, T071 tests can run in parallel

**Polish Phase (Phase 7)**:
- T074, T075, T076, T078, T081, T084 can run in parallel

---

## Parallel Execution Examples

### Example 1: User Story 1 Core Implementation (Parallel)

```bash
# Terminal 1: Implement aggregation
uv run python -c "from robot_sf.research import aggregation; print('Module skeleton ready')"

# Terminal 2: Implement statistics (simultaneously)
uv run python -c "from robot_sf.research import statistics; print('Module skeleton ready')"

# Terminal 3: Implement figures (simultaneously)
uv run python -c "from robot_sf.research import figures; print('Module skeleton ready')"

# After all complete: Implement orchestrator (depends on T016-T019)
uv run python -c "from robot_sf.research import orchestrator; print('Orchestrator ready')"
```

### Example 2: User Story 1 Testing (Parallel)

```bash
# Run all unit tests in parallel after implementation
uv run pytest tests/research/test_aggregation.py tests/research/test_statistics.py tests/research/test_figures.py -n auto
```

### Example 3: Multi-Story Parallel Development

```bash
# Developer 1: Focus on US1
uv run pytest tests/research/test_integration_report.py -k test_generate_full_report

# Developer 2: Focus on US2 (simultaneously)
uv run pytest tests/research/test_orchestrator.py -k test_multi_seed

# Developer 3: Focus on US3 (simultaneously)
uv run pytest tests/research/test_figures.py -k test_learning_curve
```

---

## Implementation Strategy

### MVP Scope (Minimum Viable Product)

**Recommended MVP**: User Story 1 (P1) ONLY

**Rationale**: Delivers core value proposition - transform pipeline runs into structured reports with metrics, figures, and metadata. Independently testable and deployable.

**MVP Tasks**: T001-T015 (Setup + Foundational) + T016-T032 (US1 Implementation + Tests) = **32 tasks**

**MVP Validation**:
```bash
# Run full pipeline
uv run python examples/advanced/16_imitation_learning_pipeline.py --enable-tracker

# Generate report
uv run python scripts/research/generate_report.py \
  --tracker-run <run_id> \
  --experiment-name "MVP Demo" \
  --output output/research_reports/mvp_demo

# Validate output
uv run python scripts/tools/validate_report.py output/research_reports/mvp_demo
```

### Incremental Delivery Roadmap

1. **Sprint 1**: Setup + Foundational (T001-T015) - Foundation ready
2. **Sprint 2**: User Story 1 Implementation (T016-T027) - Core reporting functional
3. **Sprint 3**: User Story 1 Tests + Validation (T028-T032) - MVP complete ✅
4. **Sprint 4**: User Story 2 (T033-T043) - Multi-seed robustness
5. **Sprint 5**: User Story 3 (T044-T059) - Publication-quality outputs
6. **Sprint 6**: User Story 4 (T060-T073) - Research depth (ablations)
7. **Sprint 7**: Polish (T074-T086) - Production-ready

### Success Metrics per Sprint

- **Sprint 1**: Foundation tests pass, directory structure created
- **Sprint 2**: Basic report generates from minimal fixture
- **Sprint 3**: SC-001, SC-003, SC-005 validated (MVP success criteria)
- **Sprint 4**: SC-009 validated (completeness scoring)
- **Sprint 5**: SC-006, SC-013 validated (statistical quality)
- **Sprint 6**: SC-007, SC-008 validated (ablation coverage)
- **Sprint 7**: All SC-001 through SC-014 validated; performance budget met

---

## Task Summary

- **Total Tasks**: 86
- **Setup**: 7 tasks
- **Foundational**: 7 tasks
- **User Story 1 (P1)**: 17 tasks (12 implementation + 5 tests)
- **User Story 2 (P2)**: 11 tasks (7 implementation + 4 tests)
- **User Story 3 (P3)**: 16 tasks (10 implementation + 6 tests)
- **User Story 4 (P4)**: 14 tasks (9 implementation + 5 tests)
- **Polish**: 13 tasks
- **Parallel Opportunities**: 28 tasks marked [P] (33% parallelizable)

---

## Format Validation

✅ All tasks follow checklist format: `- [ ] [ID] [P?] [Story?] Description with file path`  
✅ All user story tasks labeled with [US1], [US2], [US3], or [US4]  
✅ All tasks include specific file paths or validation commands  
✅ Dependencies clearly documented in execution order section  
✅ MVP scope identified (User Story 1 = 32 tasks)  
✅ Independent test criteria provided for each user story  
✅ Parallel opportunities marked with [P] tag
