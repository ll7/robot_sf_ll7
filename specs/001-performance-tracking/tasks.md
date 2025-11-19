# Tasks: Performance Tracking & Telemetry for Imitation Pipeline

**Input**: Design documents from `/specs/001-performance-tracking/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Repository standards expect pytest coverage plus validation scripts when functionality changes user-facing behavior. Each story below references its independent verification criteria; dedicated pytest tasks are marked where needed.

**Organization**: Tasks are grouped by user story to keep increments independently testable.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish scaffolding for telemetry code, CLI tooling, and tests so later phases plug into a consistent module layout.

- [X] T001 Create telemetry package entry point in `robot_sf/telemetry/__init__.py` and expose placeholder import from `robot_sf/__init__.py`.
- [X] T002 Add test scaffolding in `tests/test_tracking/__init__.py` and `tests/test_tracking/conftest.py` for temporary artifact roots.
- [X] T003 Scaffold CLI stub with argparse + placeholders in `scripts/tools/run_tracker_cli.py` matching quickstart commands (status/list/perf-tests).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the run-record data structures, artifact routing, and manifest writer that every user story depends on.

- [X] T004 Implement telemetry dataclasses (`PipelineRunRecord`, `StepExecutionEntry`, `TelemetrySnapshot`, etc.) in `robot_sf/telemetry/models.py`.
- [X] T005 Create `RunTrackerConfig` with artifact-root resolution + defaults in `robot_sf/telemetry/config.py` (reuse `robot_sf/common/artifact_paths.py`).
- [X] T006 Develop JSONL manifest writer with append/resume/rotation logic in `robot_sf/telemetry/manifest_writer.py`.
- [X] T007 Implement per-run lock + ID registry to ensure unique directories in `robot_sf/telemetry/run_registry.py`.
- [X] T008 Add pytest coverage for manifest writer + locking in `tests/test_tracking/test_manifest_writer.py` using temp dirs.
- [X] T009 Extend `robot_sf/common/artifact_paths.py` with `ensure_run_tracker_tree()` helper that prepares `output/run-tracker/<run_id>`.

**Checkpoint**: Run tracker can persist structured manifests safely; user-story work may now begin.

---

## Phase 3: User Story 1 – Monitor live pipeline progress (Priority: P1) 🎯 MVP

**Goal**: Operators see "step X/Y", elapsed time, and ETA directly from the imitation-learning pipeline.
**Independent Test**: Run `examples/advanced/16_imitation_learning_pipeline.py --enable-tracker` and verify console/log output announces each step start/end with updated ETA.

### Implementation & Validation

- [X] T010 [P] [US1] Implement `ProgressTracker` with step lifecycle + ETA smoothing in `robot_sf/telemetry/progress.py`.
- [X] T011 [US1] Emit step lifecycle events from `examples/advanced/16_imitation_learning_pipeline.py` using `ProgressTracker` hooks.
- [X] T012 [P] [US1] Add helper to derive enabled steps + ordinal metadata in `robot_sf/training/imitation_config.py` (or new runner module) for manifest consistency.
- [X] T013 [US1] Implement `status`/`watch` commands in `scripts/tools/run_tracker_cli.py` to display current step, elapsed time, and ETA from manifests.
- [X] T014 [P] [US1] Add ETA/progress pytest coverage in `tests/test_tracking/test_progress_tracker.py` (mock step durations + CLI rendering).
- [X] T015 [US1] Update `scripts/validation/run_examples_smoke.py` to toggle tracker mode and assert progress output so regressions surface in CI.

---

## Phase 4: User Story 2 – Review historical runs (Priority: P2)

**Goal**: Researchers can list at least the last 20 runs, filter by status/date, and inspect manifests even when runs failed.
**Independent Test**: Execute two tracker-enabled runs, then run `uv run python scripts/tools/run_tracker_cli.py list --status failed` to confirm filtering + manifest links; kill one run mid-step and confirm failure flush persists data.

### Implementation & Validation

- [X] T016 [P] [US2] Implement manifest index + filtering API (limit/status/since) in `robot_sf/telemetry/history.py`.
- [X] T017 [US2] Wire `list`/`show`/`export` commands in `scripts/tools/run_tracker_cli.py` to the history API with JSON + Markdown output support.
- [X] T018 [US2] Add failure-safe flush + signal handling in `robot_sf/telemetry/progress.py` so partial runs persist within 5s of abort.
- [X] T019 [P] [US2] Add pytest coverage for history filtering + failure flush in `tests/test_tracking/test_history.py` (use sample manifests in `tests/test_tracking/fixtures/`).
- [X] T020 [US2] Document run-summary workflow (filters, failure recovery) in `docs/dev_guide.md` and `docs/README.md`.

---

## Phase 5: User Story 3 – Telemetry & recommendations + perf tests (Priority: P3)

**Goal**: Capture resource telemetry, surface actionable recommendations, support optional TensorBoard, and expose scripted performance tests.
**Independent Test**: Enable telemetry on the pipeline, confirm telemetry snapshots + recommendations appear in the run summary, and run `uv run python scripts/telemetry/run_perf_tests.py` to record throughput + pass/fail status.

### Implementation & Validation

- [X] T021 [P] [US3] Build psutil-based telemetry sampler with configurable interval in `robot_sf/telemetry/sampler.py`.
- [X] T022 [P] [US3] Implement GPU metrics helper using NVML (guarded import) in `robot_sf/telemetry/gpu.py`.
- [X] T023 [US3] Integrate sampler loop + telemetry snapshot persistence into `examples/advanced/16_imitation_learning_pipeline.py` and related helpers.
- [X] T024 [US3] Implement rule-based recommendation engine with severity + suggested actions in `robot_sf/telemetry/recommendations.py`.
- [X] T025 [US3] Surface telemetry + recommendation summaries via `scripts/tools/run_tracker_cli.py summary` subcommand.
- [X] T026 [P] [US3] Add optional TensorBoard adapter in `robot_sf/telemetry/tensorboard_adapter.py` and CLI flag to enable mirroring.
- [X] T027 [US3] Create performance-test wrapper in `scripts/telemetry/run_perf_tests.py` that calls `scripts/validation/performance_smoke_test.py` and records results via manifest writer.
- [X] T028 [US3] Extend `scripts/validation/performance_smoke_test.py` to accept telemetry flags and emit recommendation-aware pass/fail statuses.
- [ ] T029 [P] [US3] Add pytest coverage for sampler fallbacks + recommendation triggers in `tests/test_tracking/test_telemetry.py`.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Align docs/CI artifacts, changelog, and quickstart instructions once all stories are functional.

- [ ] T030 Update `CHANGELOG.md` (Unreleased) with a summary of telemetry + run-tracking features.
- [ ] T031 [P] Sync quickstart/docs with final CLI flags by revising `specs/001-performance-tracking/quickstart.md`, `docs/dev_guide.md`, and `docs/README.md`.
- [ ] T032 Run quickstart validation end-to-end and capture artifact links in `specs/001-performance-tracking/spec.md` "Testing" notes.
- [ ] T033 Wire tracker smoke/perf tests into CI by updating `scripts/validation/run_examples_smoke.py` and `.github/workflows/ci.yml` to call the new perf wrapper.

---

## Dependencies & Execution Order

1. **Phase 1 → Phase 2**: Foundational work requires the telemetry package, CLI stub, and test scaffolding to exist.
2. **Phase 2 → All User Stories**: Data models, config, manifest writer, and artifact helpers power every story; no story work starts until T004–T009 pass.
3. **User Story Sequencing**: Stories can proceed sequentially by priority (US1 → US2 → US3) or in parallel once Foundational tasks finish, provided shared files (e.g., `scripts/tools/run_tracker_cli.py`) follow coordination notes.
4. **Polish Phase**: Executes after the desired user stories are feature-complete; CI wiring depends on perf wrapper from US3.

### Story Dependency Graph

- US1 (progress) only depends on Phase 2.
- US2 depends on Phase 2 plus US1's manifest schema (step entries) but can start once manifest writer stabilizes.
- US3 depends on Phase 2 and reuses US1's step hooks plus US2's manifest querying for summaries.

### Parallel Opportunities

- **Setup**: T001–T003 touch different files and can run concurrently.
- **Foundational**: T004/T005/T006/T007/T009 operate on distinct modules; tests (T008) start once writers exist.
- **US1**: T010 and T012 can progress in parallel, while T011/T013/T015 depend on them. T014 (tests) may run once T010 is drafted.
- **US2**: T016 and T018 are independent; T017 waits on T016, while T019 (tests) can run once fixtures exist.
- **US3**: T021, T022, and T026 can proceed simultaneously; T024 depends on telemetry metrics, while T027/T028 depend on both telemetry and recommendations. T029 covers sampler/recommendations after implementations.
- **Polish**: T030–T033 mostly touch docs/CI and can happen concurrently after core features stabilize.

## Implementation Strategy

1. **MVP (US1)**: Finish Phases 1–3 so the pipeline shows progress + ETA; this unlocks immediate operator value.
2. **Incremental Delivery**:
   - Deliver US1 as the MVP.
   - Layer US2 to provide historical manifests without waiting for telemetry work.
   - Ship US3 last, adding telemetry/recommendations/perf tests; optional TensorBoard adapter can trail behind sampler integration if needed.
3. **Validation Cadence**:
   - After each story, run the quickstart commands described in `quickstart.md` plus targeted pytest files.
   - Update docs + changelog only after behavior stabilizes to avoid churn.
4. **CI Integration**: Once US3 lands, wire the perf wrapper + tracker validation into CI so regressions block merges.
