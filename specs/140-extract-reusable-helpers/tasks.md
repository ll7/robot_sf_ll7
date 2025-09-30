# Tasks: Reusable Helper Consolidation

**Input**: Design documents from `/specs/140-extract-reusable-helpers/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/, quickstart.md

## Phase 3.1: Setup
- [X] T001 Document current helper usage across `examples/` and `scripts/` in `specs/140-extract-reusable-helpers/helper_inventory.yaml` using the two-pass audit criteria.
- [X] T002 Capture required regression commands in `specs/140-extract-reusable-helpers/regression_checks.yaml` (pytest suites + validation scripts) with descriptions per RegressionCheck data model.

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
- [ ] T003 [P] Add failing tests for benchmark helper catalog in `tests/benchmark/test_helper_catalog.py` covering `prepare_classic_env`, `load_trained_policy`, and `run_episodes_with_recording` (success + error paths).
- [ ] T004 [P] Add failing tests for render helper catalog in `tests/render/test_helper_catalog.py` validating `ensure_output_dir` and `capture_frames` behaviors (headless-friendly fixtures).
- [ ] T005 [P] Add failing tests for docs helper catalog in `tests/docs/test_helper_catalog.py` verifying `register_helper` updates the docs index manifest.
- [ ] T006 [P] Add integration test in `tests/integration/test_helper_orchestrator_flow.py` that exercises the quickstart flow: orchestrator imports helpers, runs a stub episode, and checks ExampleOrchestrator linkage metadata.

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [ ] T007 Implement helper registry data structures (`HelperCategory`, `HelperCapability`, `ExampleOrchestrator`, `OrchestratorUsage`, `RegressionCheck`) in a new module `robot_sf/benchmark/helper_registry.py` aligned with the data model (typed, documented).
- [ ] T008 Implement render helper catalog in `robot_sf/render/helper_catalog.py`, including frame capture utilities and directory management with Loguru instrumentation.
- [ ] T009 Implement benchmark helper catalog in `robot_sf/benchmark/helper_catalog.py`, extracting policy loading, environment preparation, episode execution, and recording logic from existing examples while reusing registry types.
- [ ] T010 Implement docs helper catalog in `robot_sf/docs/helper_catalog.py`, wiring `register_helper` to update `docs/README.md` linking structures and ensuring idempotency.
- [ ] T011 Update package exports (`robot_sf/benchmark/__init__.py`, `robot_sf/render/__init__.py`, `robot_sf/docs/__init__.py`) and any shared utilities to surface the new helper modules with complete docstrings.
- [ ] T012 Refactor `robot_sf/benchmark/utils.py` (and related modules) to reuse helper catalog functions and remove duplicate logic (e.g., `_load_policy`, recording warnings).

## Phase 3.4: Integration
- [ ] T013 [P] Refactor `examples/classic_interactions_pygame.py` to orchestrate exclusively through helper catalog functions and update `OrchestratorUsage` metadata in the inventory file.
- [ ] T014 [P] Refactor `examples/demo_full_classic_benchmark.py` to use helper catalog API for environment preparation and episode execution.
- [ ] T015 [P] Refactor `examples/demo_offensive.py` and `examples/demo_defensive.py` to delegate to helper catalog utilities, removing bespoke setup code.
- [ ] T016 [P] Refactor `examples/demo_pedestrian.py` and `examples/demo_pedestrian_updated.py` to rely on helper catalog for env/policy handling while preserving pedestrian-specific parameters.
- [ ] T017 [P] Refactor `examples/demo_aggregate.py` and `examples/demo_factory_options.py` to call helper catalog formatting and execution helpers.
- [ ] T018 [P] Refactor `examples/demo_robot_sf_bench_ppo.py` and `examples/demo_refactored_environments.py` to import helper catalog utilities, exposing only orchestration glue.
- [ ] T019 Refactor `scripts/run_classic_interactions.py` to use helper catalog functions and update CLI argument handling accordingly.
- [ ] T020 Refactor `scripts/run_social_navigation_benchmark.py` to reuse helper catalog orchestrators for batch execution.
- [ ] T021 Refactor `scripts/benchmark.py` and `scripts/benchmark02.py` to delegate environment/policy setup to helper catalog utilities, ensuring resume semantics remain intact.
- [ ] T022 Refactor `scripts/benchmark_repro_check.py` and `scripts/benchmark_workers.py` to consume helper catalog APIs, updating worker interfaces as needed.

## Phase 3.5: Polish
- [ ] T023 Add helper catalog documentation page at `docs/dev/helper_catalog.md` and link it from `docs/README.md` and any relevant section indices via `register_helper` automation.
- [ ] T024 Update `examples/README.md` (and per-example docstrings where applicable) to reference the helper catalog and quickstart usage.
- [ ] T025 Record helper-to-orchestrator mappings and regression commands in `specs/140-extract-reusable-helpers/helper_inventory.yaml` and `regression_checks.yaml`, confirming they reflect post-refactor state (update from setup placeholders).
- [ ] T026 Update `CHANGELOG.md` with a summary of the helper consolidation feature (public helper surface + example/script refactors).
- [ ] T027 Run formatting and linting (`uv run ruff check --fix .` followed by `uv run ruff check .`) to ensure style compliance.
- [ ] T028 Run unit tests (`uv run pytest tests`) and ensure newly added tests pass.
- [ ] T029 Run rendering/headless tests (`uv run pytest test_pygame`) to validate frame capture changes.
- [ ] T030 Execute validation scripts `scripts/validation/test_basic_environment.sh` and `scripts/validation/test_complete_simulation.sh`, comparing outputs against regression expectations.
- [ ] T031 Update `specs/140-extract-reusable-helpers/plan.md` Progress Tracking to reflect completed phases and note any follow-up work discovered during implementation.
- [ ] T032 Prepare implementation summary assets: helper catalog diff overview and example before/after snippets saved under `specs/140-extract-reusable-helpers/progress/` for PR context.

## Dependencies
- Phase 3.2 tasks must complete before Phase 3.3 (tests prior to implementation).
- T007–T012 depend on the new tests and registry definitions; ensure helper modules exist before refactoring orchestrators (T013–T022).
- Documentation and regression validation (T023–T030) depend on successful refactors.

## Parallel Example
```
# After completing T012, run the following helper-oriented refactors in parallel:
/tasks run T013
/tasks run T014
/tasks run T015
/tasks run T016
```
These tasks operate on distinct example files and can proceed concurrently once the helper modules are stable.
