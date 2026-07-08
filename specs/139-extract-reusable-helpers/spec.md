# Feature Specification: Extract reusable helpers from examples/classic_interactions_pygame.py

**Feature Branch**: `139-extract-reusable-helpers`  
**Created**: 2025-09-29  
**Status**: Draft  
**Input**: User description: "Extract reusable helpers from examples/classic_interactions_pygame.py into robot_sf/benchmark modules: move frame/recording helpers, plotting/overlay helpers, demo/test helpers, model/env utilities, and summary/validation helpers into appropriate benchmark modules (visualization.py, utils.py, model_utils.py, map_utils.py). Include tests, docs/design note, and a migration plan."

## Execution Flow (main)
```
1. Parse user description from Input
   → If empty: ERROR "No feature description provided"
2. Extract key concepts from description
   → Identify: helper function groups, target modules, tests, docs
3. For each helper candidate:
   → Determine dependencies (imports, data shapes, side effects)
4. For each unclear aspect:
   → Mark with [NEEDS CLARIFICATION: specific question]
5. Create a design doc and TODO in `docs/dev/issues/classic-interactions-refactor/`
6. Implement small, well-scoped moves in sequence (frame helpers → table/formatting → model/loaders → map utilities)
7. Add unit tests for each moved helper and update example imports to the new locations
8. Run quality gates: Ruff, ty, pytest; iterate on failures
9. Return: SUCCESS (spec ready for implementation)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT to extract and WHY (reusability, testability, reduced duplication)
- ❌ Avoid large refactors that change public behavior; prefer small, test-covered moves
- 👥 Write a short design note and a migration plan so reviewers can follow the change

### Section Requirements
- **Mandatory sections**: User Scenarios & Testing, Functional Requirements, Migration Plan
- **Optional sections**: Performance expectations, Back-compat notes

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a developer working on benchmark tooling, I want common demo helpers extracted into the `robot_sf/benchmark` package so that examples remain thin, behavior is shared across tools, and unit tests can target small pure helpers.

### Acceptance Scenarios
1. **Given** the `examples/classic_interactions_pygame.py` demo, **When** helpers are moved into `robot_sf/benchmark/*`, **Then** the demo imports the helpers from the new modules and behavior is identical (example output / artifacts unchanged).
2. **Given** the moved helpers, **When** unit tests for those helpers run, **Then** they pass deterministically in CI (no example-level side-effects required).

### Edge Cases
- Helpers that close over module-level demo state (e.g., env instances, global caches) must be refactored into pure functions or small adapters that accept explicit parameters.
- Any helper that relies on optional deps (moviepy, SimulationView) must gracefully degrade and be covered by unit tests that monkeypatch dependency presence/absence.

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: Provide a design note at `docs/dev/issues/classic-interactions-refactor/design.md` describing which helpers move and target modules.
- **FR-002**: Create `specs/139-extract-reusable-helpers/spec.md` (this file) and track progress in `docs/dev/issues/classic-interactions-refactor/todo.md`.
- **FR-003**: Move the following helper groups to `robot_sf/benchmark` submodules and ensure public API stable:
  - frame/recording helpers → `robot_sf/benchmark/visualization.py`
  - plotting/overlay helpers → `robot_sf/benchmark/visualization.py`
  - demo/test helpers (fast-mode, scenario prep) → `robot_sf/benchmark/utils.py`
  - model loading & policy utilities → `robot_sf/benchmark/model_utils.py`
  - map loading / conversion helpers → `robot_sf/benchmark/map_utils.py`
- **FR-004**: Add unit tests in `tests/unit/benchmark/` covering happy path + one edge case per helper.
- **FR-005**: Preserve behavior of the example: running `examples/classic_interactions_pygame.py` on the same inputs the same outputs should be created.

### Non-functional Requirements
- **NFR-001**: Changes must be covered by unit tests and the existing CI pipeline (Ruff + ty + pytest). No new lint errors.
- **NFR-002**: Avoid reintroducing heavy optional deps at import time; use lazy imports or try/except to keep top-level imports fast.

## Key Entities
- **Helper**: Small pure or small-adapter function (frame size, overlay text, record flag) designed to be testable in isolation.
- **Benchmark module**: `robot_sf/benchmark/*` where helpers will live; modules must have docstrings and tests.

---

## Clarifications

### Session 2025-09-29

- Q: Which helper groups should be included in Phase A (the first, smallest-risk extraction)? → A: Option A (Frame/recording helpers + plotting/overlay helpers + formatting/table helpers).

Applied: Phase A scope is clarified. Migration Plan step 2 has been updated to explicitly define Phase A as the extraction of frame/recording, plotting/overlay, and formatting/table helpers (pure helpers, lowest risk).


## Migration Plan (step-by-step)
1. Create design doc + todo in `docs/dev/issues/classic-interactions-refactor/` (this is the coordination artifact).
2. Phase A (first extraction): Move frame/recording helpers + plotting/overlay helpers + formatting/table helpers (lowest-risk, pure helpers):
   - Files to edit: `examples/classic_interactions_pygame.py` (remove helpers, import new functions), `robot_sf/benchmark/visualization.py` (add functions), `robot_sf/benchmark/utils.py` (formatting/table helpers), `tests/unit/benchmark/test_visualization_helpers.py` (new tests for visualization helpers) and `tests/unit/benchmark/test_utils_helpers.py` (new tests for formatting helpers).
3. Move formatting/table helpers into `robot_sf/benchmark/utils.py` and add tests.
4. Move model/load helpers into `robot_sf/benchmark/model_utils.py`.
5. Move map loaders into `robot_sf/benchmark/map_utils.py`.
6. Run quality gates (Ruff, ty, pytest); fix issues.
7. Add a one-line entry in `CHANGELOG.md` and link the design note.
8. Open PR targeting `main` from `139-extract-reusable-helpers` with clear description, list of moved functions, and verification steps.

## Testing Plan
- Unit tests per helper in `tests/unit/benchmark/` (pytest fixtures + small inputs).
- Smoke test: run `python examples/classic_interactions_pygame.py --dry-run` to validate example-level behavior.
- CI: Ensure Ruff and `uvx ty check . --exit-zero` pass; run `uv run pytest tests`.

## Backwards Compatibility & Notes
- Example import paths will change; keep compatibility shim (deprecated import path) for one release if necessary: `robot_sf.benchmark.visualization.import_from_examples()`.
- Avoid changing external CLI flags or JSON schema during this refactor.

## Review & Acceptance Checklist
- [ ] Design doc added at `docs/dev/issues/classic-interactions-refactor/design.md`.
- [ ] TODO tracker added at `docs/dev/issues/classic-interactions-refactor/todo.md` with tasks for each helper group.
- [ ] Frame/recording helpers moved and unit-tested.
- [ ] Plotting/overlay helpers moved and unit-tested.
- [ ] Demo/test helpers moved and unit-tested.
- [ ] Model & map helpers moved and unit-tested.
- [ ] Examples updated to import from new locations and smoke-tested.
- [ ] PR opened with description, small changelog, and verification instructions.

---

## Execution Status
- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked (none blocking)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist pending

---

This spec is ready for implementation. The branch `139-extract-reusable-helpers` was created and checked out; implement the changes in small commits and run the CI gate locally before opening the PR.
