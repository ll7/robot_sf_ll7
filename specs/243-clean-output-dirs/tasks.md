# Tasks: Clean Root Output Directories

**Input**: Design documents from `/specs/243-clean-output-dirs/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Test tasks are included where enforcement or migration correctness must be verified.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare repository ignore rules and tooling skeleton for artifact consolidation.

- [x] T001 Update `.gitignore` to ignore the new `output/` hierarchy and remove redundant top-level artifact entries.
- [x] T002 Create `scripts/tools/__init__.py` with a module docstring to establish the tooling package namespace.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core helpers required by every user story.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T003 Refactor `robot_sf/common/artifact_paths.py` to expose canonical `output/` root helpers, category metadata, and an `ensure_canonical_tree` utility.
- [x] T004 Update `robot_sf/common/__init__.py` to re-export the new artifact helper functions for downstream imports.
- [x] T005 Integrate the canonical tree helper into `tests/conftest.py` so pytest sessions create and tear down the expected artifact layout.
- [x] T006 Add unit coverage for artifact helpers in `tests/test_guard/test_artifact_paths.py` (default root, override handling, legacy path detection).

**Checkpoint**: Foundation ready – artifact root helpers available to all stories.

---

## Phase 3: User Story 1 – Maintainer sees clean root (Priority: P1) 🎯 MVP

**Goal**: Local workflows migrate legacy directories into `output/` and default scripts write to structured subdirectories.

**Independent Test**: Run `uv run python scripts/tools/migrate_artifacts.py --dry-run` followed by `uv run python scripts/tools/check_artifact_root.py`; confirm the migration report lists all legacy paths and the guard reports zero violations after migration.

### Implementation for User Story 1

- [x] T007 [US1] Implement `scripts/tools/migrate_artifacts.py` with dry-run, summary report, and legacy path relocation into `output/`.
- [x] T008 [P] [US1] Register a console entry point in `pyproject.toml` (e.g., `robot-sf-migrate-artifacts`) that invokes the migration script.
- [x] T009 [US1] Update `scripts/benchmark02.py` to default JSON outputs into `output/benchmarks/` via the artifact helper.
- [x] T010 [P] [US1] Update `scripts/validation/performance_smoke_test.py` to persist logs under `output/benchmarks/` and remove root-level `results/` usage.
- [x] T011 [P] [US1] Update `scripts/run_classic_interactions.py` and related CLI defaults to write episode manifests under `output/benchmarks/`.
- [x] T012 [P] [US1] Update `scripts/play_recordings.py` to read from and list `output/recordings/` while honoring the override.
- [x] T013 [US1] Create migration regression tests in `tests/test_guard/test_migrate_artifacts.py` covering dry-run, actual relocation, and idempotency.

**Checkpoint**: Maintainers running local workflows produce no stray root directories.

---

## Phase 4: User Story 2 – CI artifacts stay organized (Priority: P2)

**Goal**: Automated pipelines enforce the artifact policy and publish outputs from the canonical root only.

**Independent Test**: Execute `uv run python scripts/tools/check_artifact_root.py` after a CI-equivalent run (pytest + benchmarks); verify zero violations and confirm `.github/workflows/ci.yml` uploads artifacts from `output/` paths exclusively.

### Implementation for User Story 2

- [x] T014 [US2] Implement `scripts/tools/check_artifact_root.py` to detect unapproved top-level files/directories and emit actionable failures.
- [x] T015 [P] [US2] Add guard coverage in `tests/test_guard/test_check_artifact_root.py`, including positive (clean) and negative (legacy path) scenarios.
- [x] T016 [US2] Wire the guard script into `.github/workflows/ci.yml` so every job fails fast when legacy directories appear.
- [x] T017 [P] [US2] Update `scripts/coverage/open_coverage_report.py` and related tooling to target `output/coverage/` (HTML + JSON paths).
- [x] T018 [P] [US2] Revise `scripts/validation/test_coverage_collection.sh` and any coverage utilities to look under `output/coverage/`.
- [x] T019 [US2] Adjust CI artifact upload steps in `.github/workflows/ci.yml` to publish `output/coverage/`, `output/benchmarks/`, and `output/recordings/` directories.

**Checkpoint**: Guard enforcement active in CI; artifacts published from canonical paths.

---

## Phase 5: User Story 3 – Documentation highlights artifact policy (Priority: P3)

**Goal**: Contributors learn the new artifact policy, migration steps, and override workflow from the core docs.

**Independent Test**: Follow updated instructions in `docs/dev_guide.md` and `README.md` to run tests, inspect artifacts under `output/`, and override the root; confirm directions match actual behavior.

### Implementation for User Story 3

- [x] T020 [US3] Update `docs/dev_guide.md` sections on artifact handling, coverage, and validation to reference the `output/` hierarchy and migration script.
- [x] T021 [P] [US3] Refresh root `README.md` to summarize the artifact policy and link to the quickstart.
- [x] T022 [P] [US3] Update `docs/coverage_guide.md` with the new coverage paths and guard workflow.
- [x] T023 [US3] Add cross-links (e.g., `docs/README.md` and `docs/dev/issues/*`) pointing to the quickstart and guard scripts for discoverability.

**Checkpoint**: Documentation accurately reflects artifact management expectations.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanup once all user stories are complete.

- [ ] T024 [P] Add CHANGELOG entry in `CHANGELOG.md` summarizing the artifact root overhaul and migration guidance.
- [ ] T025 Run `specs/243-clean-output-dirs/quickstart.md` end-to-end and capture adjustments in `docs/dev_guide.md` if discrepancies are found.
- [ ] T026 Verify `.github/workflows/ci.yml` and scripts/tooling adhere to Loguru logging standards, adding notes where guard output needs structured logging.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)** → blocks Phase 2
- **Foundational (Phase 2)** → blocks all user stories
- **User Story 1 (Phase 3)** → becomes MVP once complete
- **User Story 2 (Phase 4)** → depends on Phase 2 (shares helpers) but can start after Phase 3 begins if migration script is available
- **User Story 3 (Phase 5)** → depends on Phase 3 outputs so docs reference actual tools
- **Polish (Phase 6)** → runs after targeted user stories finish

### User Story Dependencies

- **US1 (P1)**: First deliverable after foundational tasks; no dependency on other stories.
- **US2 (P2)**: Depends on US1 migration tooling to supply guard inputs; otherwise independent.
- **US3 (P3)**: Depends on US1+US2 to document accurate behavior.

### Within Each User Story

1. Validate helpers/tests (from Phase 2) are green.
2. Implement scripts/services.
3. Update documentation or CI as required.
4. Run the independent test scenario before closing the story.

### Parallel Opportunities

- T008, T010, T011, T012, T015, T017, T018, T021, T022, T023, T024 marked `[P]` can run concurrently once their prerequisite tasks complete.
- After Phase 2, different developers can split User Stories: e.g., one tackles migration script updates (US1) while another prepares guard enforcement (US2).

---

## Parallel Example: User Story 1

```bash
# Run migration tooling in parallel once helpers exist
uv run python scripts/tools/migrate_artifacts.py --dry-run
uv run python scripts/tools/migrate_artifacts.py

# Update producers simultaneously (different files)
Apply T010 to scripts/validation/performance_smoke_test.py
Apply T011 to scripts/run_classic_interactions.py
Apply T012 to scripts/play_recordings.py
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Setup (Phase 1) and Foundational (Phase 2).
2. Implement migration tooling and producer updates (Phase 3).
3. Run the independent test and ensure guard reports zero violations.

### Incremental Delivery

1. Deliver US1 (migration + local cleanliness) as MVP.
2. Layer US2 (CI enforcement + coverage path updates).
3. Finish with US3 (documentation polish) and final cross-cutting tasks.

### Parallel Team Strategy

- Developer A: Foundations → US1 migration tooling.
- Developer B: Guard script and CI wiring (US2) once helpers exist.
- Developer C: Documentation updates (US3) after scripts are ready.

---

## Notes

- `[P]` tasks operate on distinct files and may run concurrently when dependencies allow.
- Each user story is independently testable per the defined acceptance checks.
- Ensure guard and migration scripts log via Loguru and honor `ROBOT_SF_ARTIFACT_ROOT` overrides.
- Commit after each task or logical group; stop at checkpoints to verify story completion.
