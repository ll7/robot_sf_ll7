---

description: "Implementation task list for feature 245"
---

# Tasks: Organize and Categorize Example Files

**Input**: Design documents from `/specs/245-organize-examples/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Automated example smoke tests are introduced for the maintainer story (User Story 3). Other stories rely on documentation-driven validation as described in the specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish shared scaffolding (manifest stub, directory skeleton, test package) required by all user stories.

- [X] T001 Create initial manifest stub `examples/examples_manifest.yaml` enumerating all existing scripts with placeholder metadata.
- [X] T002 Create directory scaffold `examples/quickstart/`, `examples/advanced/`, `examples/benchmarks/`, `examples/plotting/`, `examples/_archived/` with `.gitkeep` files where needed.
- [X] T003 Add `tests/examples/__init__.py` to register the upcoming smoke test package.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core tooling and templates that all stories depend on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Implement manifest loader module `robot_sf/examples/manifest_loader.py` modeling ExampleCategory, ExampleScript, ExampleManifest entities.
- [X] T005 Create validation script `scripts/validation/validate_examples_manifest.py` that checks paths, category slugs, docstring alignment, CI flags, and ensures every script under `examples/` is represented in the manifest.
- [X] T006 Publish docstring template `docs/templates/example_docstring.md` describing required sections (purpose, usage, prerequisites, output, limitations).
- [X] T007 Add README generator script `scripts/validation/render_examples_readme.py` that converts the manifest into the section layout for `examples/README.md`.

**Checkpoint**: Manifest tooling, validation, and documentation templates are ready for user story implementation.

---

## Phase 3: User Story 1 - New User Discovers Starting Point (Priority: P1) 🎯 MVP

**Goal**: Provide a curated quickstart path (3–5 scripts) with clear instructions so new users can run their first example within minutes.

**Independent Test**: From `examples/README.md`, follow the quickstart section to run `examples/quickstart/01_basic_robot.py`, `02_trained_model.py`, and `03_custom_map.py` successfully.

### Implementation for User Story 1

- [X] T008 [P] [US1] Move `examples/demo_refactored_environments.py` to `examples/quickstart/01_basic_robot.py` and update docstring to match manifest template.
- [X] T009 [P] [US1] Move `examples/demo_robot_sf_bench_ppo.py` to `examples/quickstart/02_trained_model.py` with updated docstring and prerequisites.
- [X] T010 [P] [US1] Move `examples/simulate_with_svg_map.py` to `examples/quickstart/03_custom_map.py`, including new docstring instructions for map assets.
- [X] T011 [US1] Populate quickstart entries in `examples/examples_manifest.yaml` (summaries, prerequisites, tags).
- [X] T012 [US1] Regenerate `examples/README.md` using `render_examples_readme.py` and author the quickstart decision tree + 5-minute walkthrough.
- [X] T013 [US1] Update quickstart guidance in `docs/dev_guide.md` to reference the new `examples/quickstart/` scripts and execution commands.

**Checkpoint**: Quickstart directory, manifest entries, and docs enable a first-time user to complete the walkthrough independently.

---

## Phase 4: User Story 2 - Developer Finds Feature-Specific Examples (Priority: P1)

**Goal**: Organize advanced feature demos (backend swaps, feature extractors, image/pedestrian scenarios) so developers can locate targeted examples with clear prerequisites.

**Independent Test**: From `examples/README.md`, follow the advanced section to locate a feature-specific script, read its docstring, and run it without consulting other docs.

### Implementation for User Story 2

- [X] T014 [P] [US2] Move backend/feature demos (`examples/demo_backend_selection.py`, `examples/demo_feature_extractors.py`, `examples/demo_factory_options.py`, `examples/image_observation_demo.py`, `examples/demonstrate_2x_speed_fix.py`) into `examples/advanced/` with consistent naming.
- [ ] T015 [P] [US2] Move complex agent and scenario demos (`examples/demo_pedestrian_updated.py`, `examples/example_multi_pedestrian.py`, `examples/example_single_pedestrian.py`, `examples/demo_defensive.py`, `examples/demo_offensive.py`, `examples/ego_ped_example.py`, `examples/sfp_demo.py`, `examples/svg_map_example.py`, `examples/trajectory_demo.py`, `examples/view_recording.py`) into `examples/advanced/` and adjust imports.
- [ ] T016 [US2] Align docstrings in `examples/advanced/*.py` with the template (usage, prerequisites, limitations).
- [ ] T017 [US2] Update advanced section entries in `examples/examples_manifest.yaml`, including `tags` (e.g., `image`, `pedestrian`).
- [ ] T018 [US2] Extend the advanced section of `examples/README.md` with developer-focused decision tree paths and cross-links to docs.
- [ ] T019 [US2] Refresh `docs/dev_guide.md` feature sections to reference the reorganized `examples/advanced/` scripts.

**Checkpoint**: Developers can find and execute feature-specific demos with clear docstrings and documentation support.

---

## Phase 5: User Story 3 - Maintainer Identifies Obsolete Examples (Priority: P1)

**Goal**: Provide archival policy, migration notes, and automated smoke tests so maintainers can audit example health quickly.

**Independent Test**: Run `uv run pytest tests/examples/test_examples_run.py` to see all active examples pass and archived ones skipped with documented reasons.

### Implementation for User Story 3

- [ ] T020 [P] [US3] Relocate deprecated or interactive-only scripts (`examples/demo_pedestrian.py`, `examples/classic_interactions_pygame.py`, `examples/interactive_playback_demo.py`, other duplicates flagged during manifest review) to `examples/_archived/` with inline pointers to replacements.
- [ ] T021 [US3] Create `examples/_archived/README.md` documenting archival criteria and replacement mappings.
- [ ] T022 [US3] Mark archived items in `examples/examples_manifest.yaml` with `ci_enabled: false` and `ci_reason` messages.
- [ ] T023 [US3] Implement manifest-driven smoke test `tests/examples/test_examples_run.py` executing all `ci_enabled` scripts headlessly (configure DISPLAY/MPLBACKEND/SDL).
- [ ] T024 [US3] Add CLI wrapper `scripts/validation/run_examples_smoke.py` that delegates to the pytest module with optional dry-run flag.
- [ ] T025 [US3] Integrate the smoke test into CI by updating `scripts/validation/test_complete_simulation.sh` (and documentation of quality gates) to invoke `uv run pytest tests/examples/test_examples_run.py`.
- [ ] T026 [US3] Document maintainer workflow in `docs/dev_guide.md` (archival policy + smoke test usage).

**Checkpoint**: Maintainers have an archival directory, manifest flags, and automated harness ensuring example health.

---

## Phase 6: User Story 4 - Documentation Reader Finds Visual and Complex Examples (Priority: P2)

**Goal**: Surface plotting and benchmark workflows with clear outputs so documentation references align with the reorganized examples.

**Independent Test**: Follow documentation references to run a benchmark script from `examples/benchmarks/` and a plotting script from `examples/plotting/`, verifying their outputs align with expectations (JSONL results, generated figures).

### Implementation for User Story 4

- [ ] T027 [P] [US4] Move benchmark runners (`examples/demo_full_classic_benchmark.py`, `examples/demo_social_nav_scenarios.py`, `examples/demo_aggregate.py`, `examples/snqi_full_flow.py`) into `examples/benchmarks/` and update relative paths.
- [ ] T028 [P] [US4] Move plotting and analysis scripts (`examples/plot_force_field.py`, `examples/plot_force_field_normalized.py`, `examples/plot_force_field_save.py`, `examples/plot_pareto.py`, `examples/plot_pedestrian_position_kde.py`, `examples/snqi_figures_example.py`, `examples/coverage_example.py`, `examples/data_analysis_example.py`) into `examples/plotting/` with consistent naming.
- [ ] T029 [US4] Refresh docstrings in `examples/benchmarks/*.py` and `examples/plotting/*.py` to describe outputs and required assets.
- [ ] T030 [US4] Update manifest entries for benchmarks/plotting with tags (`benchmark`, `visualization`) and `ci_enabled` overrides for long-running scripts.
- [ ] T031 [US4] Expand `examples/README.md` decision tree with benchmark/plotting branches and document expected outputs.
- [ ] T032 [US4] Update documentation (`docs/benchmark.md`, `docs/benchmark_visuals.md`, `docs/distribution_plots.md`, `docs/README.md`, `README.md`) to reference new file paths and link prominently to `examples/README.md`.

**Checkpoint**: Documentation references align with reorganized benchmark/plotting examples, and outputs are clearly described.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Repository-wide cleanup, validation, and release notes after all user stories are complete.

- [ ] T033 Run `scripts/validation/validate_examples_manifest.py` and `scripts/validation/run_examples_smoke.py --dry-run` to ensure manifest/doc sync.
- [ ] T034 Update `CHANGELOG.md` with a summary of the example reorganization and smoke test addition.
- [ ] T035 Execute full validation suite (`uv run pytest tests/examples/test_examples_run.py` and `uv run pytest tests`) and fix any regressions.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)** → prerequisite for all subsequent work.
- **Phase 2 (Foundational)** → depends on Phase 1; blocks all user stories.
- **Phase 3 (US1)**, **Phase 4 (US2)**, **Phase 5 (US3)**, **Phase 6 (US4)** → each depends on Phase 2 completion and can proceed in priority order (US1/US2/US3 first, US4 afterward).
- **Phase 7 (Polish)** → final once all targeted user stories are complete.

### User Story Dependencies

- **US1** and **US2** (both P1) start immediately after Phase 2 and can run concurrently once shared files (manifest, README) changes are coordinated.
- **US3** (P1) should begin after manifest updates from US1/US2 stabilize, as it integrates automation and archival policy.
- **US4** (P2) starts once benchmark/plotting scripts are free of overlaps with quickstart/advanced changes.

### Within Each User Story

- Tasks touching `examples/examples_manifest.yaml` (T011, T017, T022, T030) should be sequenced to avoid conflicts.
- README regeneration tasks (T012, T018, T031) require manifest updates to be in place before running the generator.
- Automation tasks (T023–T025) depend on manifest CI flags from T022/T030.

---

## Parallel Opportunities

- [P] tasks within each phase (T008–T010, T014–T015, T020, T027–T028) involve distinct files and can be divided among contributors.
- After Phase 2, US1 and US2 can progress in parallel as long as `examples/examples_manifest.yaml` merges are coordinated.
- Once US3 adds the smoke harness, US4 tasks that only relocate scripts (`examples/plotting`, `examples/benchmarks`) can execute in parallel with harness documentation updates.

**Parallel Execution Examples**

```bash
# Example: Run quickstart and advanced reorganizations in parallel branches
# Developer A
git checkout -b feat/quickstart-us1
# work on T008–T013

# Developer B
git checkout -b feat/advanced-us2
# work on T014–T019
```

```bash
# Example: Parallelize benchmark and plotting migrations after US3 harness lands
# Developer A handles benchmarks, Developer B handles plotting
```

---

## Implementation Strategy

### MVP First (User Story 1)

1. Complete Phases 1–2 to establish manifest tooling.
2. Execute Phase 3 (US1) to deliver the quickstart path and README walkthrough.
3. Validate by running the quickstart scripts per README instructions.

### Incremental Delivery

1. Add US2 (advanced features) once quickstart is stable.
2. Add US3 (maintenance harness + archival policy) to ensure ongoing health.
3. Add US4 (benchmarks/plotting) to align documentation references.

### Parallel Team Strategy

- Assign separate maintainers to US1 and US2 after foundational work.
- Start US3 once manifest edits settle; this story introduces testing infrastructure.
- US4 can run in parallel with US3 once the smoke harness is available to verify longer-running examples.

---

## Summary Metrics

- **Total Tasks**: 35
- **Tasks per User Story**:
  - US1: 6 tasks (T008–T013)
  - US2: 6 tasks (T014–T019)
  - US3: 7 tasks (T020–T026)
  - US4: 6 tasks (T027–T032)
- **Parallel Tasks Identified**: 10 (T008, T009, T010, T014, T015, T020, T027, T028, plus parallelizable documentation updates once prerequisites complete)
- **Independent Tests**: Defined per user story (quickstart walkthrough, developer docstring validation, smoke harness run, benchmark/plot output review)
- **Suggested MVP Scope**: Complete Phases 1–3 (US1) before proceeding to other stories.
