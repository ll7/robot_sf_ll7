---

description: "Task list for feature 242-reorganize-docs"
---

# Tasks: Reorganize Docs

**Input**: Design documents from `/specs/242-reorganize-docs/`
**Prerequisites**: plan.md (required), spec.md (user stories derived from plan for this docs-only feature), research.md, data-model.md, contracts/

**Tests**: Not requested explicitly; we’ll use independent test criteria per user story (manual verification acceptable for this docs-only change).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Paths below are absolute as required

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare working artifacts and backups for safe docs reorganization

- [X] T001 [P] Create backup of central index at docs/README.backup.md (copy of docs/README.md)
- [X] T002 [P] Create audit notes file at specs/242-reorganize-docs/audit_notes.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Ensure required design artifacts exist to guide reorganization

- [X] T003 [P] Verify quickstart exists at specs/242-reorganize-docs/quickstart.md (create if missing)
- [X] T004 [P] Verify data model exists at specs/242-reorganize-docs/data-model.md (create if missing)
- [X] T005 [P] Verify contracts placeholder exists at specs/242-reorganize-docs/contracts/openapi.yaml (create if missing)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Central Docs Index Update (Priority: P1) 🎯 MVP

**Goal**: A clear, categorized `docs/README.md` that links to all primary guides, enabling two-click navigation to key content.

**Independent Test**: Open docs/README.md and verify each top-level section exists and links resolve locally.

### Implementation for User Story 1

- [X] T010 [US1] Add "Getting Started" section linking to docs/dev_guide.md and docs/ENVIRONMENT.md in docs/README.md
- [X] T011 [US1] Add "Benchmarking & Metrics" section linking to docs/benchmark.md, docs/benchmark_full_classic.md, docs/benchmark_visuals.md in docs/README.md
- [X] T012 [US1] Add "Tooling" section linking to docs/snqi-weight-tools/README.md, docs/pyreverse.md, docs/DATA_ANALYSIS.md in docs/README.md
- [X] T013 [US1] Add "Architecture & Refactoring" section linking to docs/refactoring/ (directory index), docs/SUBTREE_MIGRATION.md, docs/UV_MIGRATION.md in docs/README.md
- [X] T014 [US1] Add "Simulation & UI" section linking to docs/SIM_VIEW.md and docs/SVG_MAP_EDITOR.md in docs/README.md
- [X] T015 [US1] Add "Figures & Visualization" section linking to docs/trajectory_visualization.md, docs/force_field_visualization.md, docs/pareto_plotting.md, docs/force_field_heatmap.md in docs/README.md
- [X] T016 [US1] Add "Performance & CI" section linking to docs/performance_notes.md and docs/coverage_guide.md in docs/README.md
- [X] T017 [US1] Add "Hardware & Environment" section linking to docs/GPU_SETUP.md and docs/ENVIRONMENT.md in docs/README.md
- [X] T018 [US1] Verify all links in docs/README.md resolve locally; update anchors/paths if needed

**Checkpoint**: User Story 1 is fully functional; central index present and working

---

## Phase 4: User Story 2 - Normalize Headings & Purpose (Priority: P2)

**Goal**: Each key doc starts with a proper H1 and a one-line purpose statement.

**Independent Test**: Open each file and verify H1 and a clear purpose sentence appear near the top.

### Implementation for User Story 2

- [X] T020 [P] [US2] Ensure H1 + purpose in docs/dev_guide.md
- [X] T021 [P] [US2] Ensure H1 + purpose in docs/ENVIRONMENT.md
- [X] T022 [P] [US2] Ensure H1 + purpose in docs/SIM_VIEW.md
- [X] T023 [P] [US2] Ensure H1 + purpose in docs/benchmark.md
- [X] T024 [P] [US2] Ensure H1 + purpose in docs/snqi-weight-tools/README.md
- [X] T025 [P] [US2] Ensure H1 + purpose page exists at docs/refactoring/README.md (create minimal if missing)
- [X] T026 [P] [US2] Ensure H1 + purpose in docs/UV_MIGRATION.md
- [X] T027 [P] [US2] Ensure H1 + purpose in docs/GPU_SETUP.md
- [X] T028 [P] [US2] Ensure H1 + purpose in docs/DATA_ANALYSIS.md

**Checkpoint**: User Story 2 is complete; headings and purposes normalized

---

## Phase 5: User Story 3 - Cross-links for Navigation (Priority: P3)

**Goal**: Provide short-path navigation between core guides and back to index.

**Independent Test**: From any core page, reach the index or another core page in ≤2 clicks.

### Implementation for User Story 3

- [X] T030 [P] [US3] Add "Back to Docs Index" link at top of docs/dev_guide.md
- [X] T031 [P] [US3] Add "Back to Docs Index" link at top of docs/ENVIRONMENT.md
- [X] T032 [P] [US3] Add cross-links in specs/242-reorganize-docs/quickstart.md to docs/dev_guide.md and docs/ENVIRONMENT.md
- [X] T033 [P] [US3] Add "See also" in docs/benchmark.md linking to docs/snqi-weight-tools/README.md and docs/distribution_plots.md
- [X] T034 [P] [US3] Add cross-links in docs/SIM_VIEW.md to docs/ENVIRONMENT.md and docs/video/ (directory)

**Checkpoint**: User Story 3 complete; cross-linking reduces navigation clicks

---

## Phase 6: User Story 4 - Orphan Scan & Missing Links (Priority: P3)

**Goal**: Identify unreferenced docs and link them from the index.

**Independent Test**: All `.md` files under `docs/` are reachable within two clicks from the index.

### Implementation for User Story 4

- [X] T040 [US4] Add Baselines docs entry in docs/README.md linking to docs/baselines/social_force.md
- [X] T041 [US4] Add missing index entry for docs/GPU_SETUP.md in docs/README.md
- [X] T043 [P] [US4] Create orphans report at specs/242-reorganize-docs/orphans.md listing any `.md` under docs/ not referenced by docs/README.md

**Checkpoint**: User Story 4 complete; orphaned docs addressed or tracked

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T050 [P] Documentation updates in docs/
- [X] T051 [P] Code cleanup and refactoring
- [X] T052 Run manual link validation and record results in specs/242-reorganize-docs/audit_notes.md

---

## Dependencies & Execution Order

### Phase Dependencies

- Setup (Phase 1): No dependencies - can start immediately
- Foundational (Phase 2): Depends on Setup completion - BLOCKS all user stories
- User Stories (Phase 3+): Prefer order US1 → US2 → (US3 || US4 in parallel)
- Polish (Final Phase): After desired user stories complete

### User Story Dependencies

- User Story 1 (P1): No dependency on other stories; enables navigation skeleton
- User Story 2 (P2): Can start after US1 to align headings with index structure
- User Story 3 (P3): Can run in parallel after US1; independent of US2
- User Story 4 (P3): Can run in parallel after US1; independent of US2/US3

### Within Each User Story

- For US1, consolidate index edits serially (avoid parallel conflicts in docs/README.md)
- For US2/US3, tasks marked [P] can proceed in parallel (different files)

### Parallel Opportunities

- Setup and Foundational tasks marked [P]
- US2 and US3 tasks (different files) can run in parallel
- US4 T043 can run in parallel with other non-index edits
- Polish T050 and T051 can run in parallel (different files)

---

## Parallel Example: User Story 2

```bash
# Parallelizable edits (different files):
Task: "Ensure H1 + purpose in docs/dev_guide.md"
Task: "Ensure H1 + purpose in docs/ENVIRONMENT.md"
Task: "Ensure H1 + purpose in docs/SIM_VIEW.md"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1 (central index)
4. STOP and VALIDATE: Verify links resolve and sections are clear

### Incremental Delivery

1. Foundation ready → US1 (index)
2. US2 (headings/purpose) in parallel with US3 (cross-links)
3. US4 (orphan scan) to wrap up
4. Polish and CHANGELOG update

---

## Summary & Validation

- Total tasks: 34
- Per user story: US1=9, US2=9, US3=5, US4=3
- Parallel opportunities: Setup, Foundational, US2, US3, US4(T043), Polish(T050,T051)
- Independent tests:
  - US1: All links in docs/README.md resolve locally
  - US2: Each file has H1 and purpose line
  - US3: Reach index or a related core page in ≤2 clicks from any core page
  - US4: Orphans report empty or all listed pages linked in index
- MVP scope: User Story 1 (central index) only

- Format validation: All tasks follow `- [ ] T### [P?] [US?] Description with absolute file path` format.
