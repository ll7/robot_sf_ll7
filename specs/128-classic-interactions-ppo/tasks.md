# Tasks: Classic Interactions PPO Pygame Visualization (Feature 128)

Input: `spec.md`, `plan.md` in this directory (MVP FR-001..FR-021, optional FR-022..FR-023)
Prerequisites: plan.md (done), spec.md (done)

This task list enforces: tests-first (TDD), deterministic & reproducible demo, no CLI (constants), safe optional recording, structured episode summaries, documentation + changelog.

Legend:
- Format: `[ID] [P?] Description (→ FR-xxx[, FR-yyy])`
- [P] = can execute in parallel (different files, no dependency chain)
- MVP scope = FR-001..FR-021. Optional backlog = FR-022..FR-023 (clarification needed).

Directory conventions (existing / proposed):
- Demo script: `examples/classic_interactions_pygame.py`
- Loader: `robot_sf/benchmark/classic_interactions_loader.py`
- New helpers (if needed): `robot_sf/benchmark/classic_interactions_runner.py` (future optional)
- Tests: `tests/classic_interactions/`
- Docs (feature issue folder): `docs/dev/issues/classic-interactions-ppo/`

## Phase 1: Setup (Foundations)
- [X] T001 Ensure tests package folder `tests/classic_interactions/` exists (mkdir, `__init__.py`). (Infra)
- [X] T002 [P] Create docs folder `docs/dev/issues/classic-interactions-ppo/` with `README.md` (purpose, link spec/plan). (Doc infra)
- [X] T003 [P] Add feature entry stub to `CHANGELOG.md` under Added (will finalize after MVP). (Meta)
- [X] T004 [P] Add link in `docs/README.md` referencing new feature docs folder. (Doc integration)

## Phase 2: Tests First (TDD) – MUST FAIL INITIALLY
Smoke & integration tests encode FR requirements before code adjustments.

- [ ] T005 [P] Smoke test: run `run_demo(dry_run=True, enable_recording=False)` returns list of episode dicts length >=1 in `tests/classic_interactions/test_smoke_demo.py` (→ FR-001 basic run, FR-004 dry run fast path, FR-010 summaries present).
- [ ] T006 [P] Deterministic seeds test: two successive `run_demo` invocations with same constants produce identical ordered `seed` list and reproducible primary outcome fields in `tests/classic_interactions/test_seed_determinism.py` (→ FR-002 deterministic ordering, FR-003 reproducibility).
- [ ] T007 [P] Recording-skip test: monkeypatch absence/failure of video backend (simulate missing `moviepy`) ensures no exception and summary marks `recording=False` in `tests/classic_interactions/test_recording_skip_graceful.py` (→ FR-008 graceful recording handling).
- [ ] T008 [P] Scenario selection test: selecting scenario constant that exists in matrix returns non-empty episode list; invalid name raises clear ValueError in `tests/classic_interactions/test_scenario_selection.py` (→ FR-005 scenario validation, FR-006 explicit error text).
- [ ] T009 [P] Outcome summary schema test: each episode dict has keys {"scenario","seed","steps","success","collision","timeout"} in `tests/classic_interactions/test_summary_schema.py` (→ FR-010 structured summary, FR-011 consistent keys).
- [ ] T010 [P] Headless safety test: set `SDL_VIDEODRIVER=dummy` env and run one episode (dry) without raising (→ FR-012 headless support).
- [ ] T011 [P] Model path failure test: corrupt/absent model path triggers clear exception or fallback message without crash in `tests/classic_interactions/test_model_path_failure.py` (→ FR-007 model load error clarity).
- [ ] T012 [P] Reward fallback integration test: create env via demo path verifying reward present & non-None (→ FR-013 reward safety integration; complements existing unit fallback test).
- [ ] T013 [P] Logging toggle test (if logging constants introduced) verifying controlled verbosity (→ FR-014 logging control).
- [ ] T014 [P] Recording success test (if dependency present) ensures file created under `results/` when `ENABLE_RECORDING=True` (skipped if dependency missing) (→ FR-009 optional video artifact path).
- [ ] T015 [P] Performance smoke test: single dry episode completes < 3s (threshold configurable) (→ FR-015 performance baseline).

## Phase 3: Core Implementation (Only after T005–T015 are authored & initially failing)
- [ ] T016 Refine `examples/classic_interactions_pygame.py`: add `dry_run` arg path to `run_demo` ensuring no rendering loop when True (if not already) (→ FR-004, FR-015).
- [ ] T017 Add deterministic seed iteration enforcement (explicit sorting, documented) inside loader or demo (→ FR-002, FR-003).
- [ ] T018 Add scenario validation raising ValueError with list of available names (→ FR-005, FR-006).
- [ ] T019 Add structured episode summary builder w/ required keys (→ FR-010, FR-011).
- [ ] T020 Integrate graceful recording guard (import try/except, flag in summary) (→ FR-008, FR-009).
- [ ] T021 Add model load error messaging & fallback guidance (→ FR-007).
- [ ] T022 Integrate reward fallback reference in demo logging (no code duplication) (→ FR-013).
- [ ] T023 Add controlled logging verbosity constant and conditional logging usage (→ FR-014).
- [ ] T024 Headless env compatibility adjustments (no unconditional display calls) (→ FR-012).
- [ ] T025 Performance micro-optimizations: skip rendering surfaces in dry_run, minimal per-step allocations (→ FR-015).
- [ ] T026 Add optional overlay info function (steps, success flag) (→ FR-016 UI clarity).
- [ ] T027 Add summary printing helper returning aligned text block (→ FR-017 human-readable output).
- [ ] T028 Provide programmatic API docstring for `run_demo` enumerating return schema (→ FR-018 documentation contract).
- [ ] T029 Add constants validation (assert directories exist or auto-create output dir) (→ FR-019 robustness).
- [ ] T030 Add explicit return type annotations + TypedDict / dataclass for episode summary (→ FR-020 typing clarity).
- [ ] T031 Update `CHANGELOG.md` with completed feature details (→ FR-021 release note).

## Phase 4: Integration & Polishing
- [ ] T032 Add additional unit tests for overlay and logging toggles (→ FR-016, FR-014).
- [ ] T033 Add type checking enhancements (mypy/ty) for new TypedDict/dataclass (→ FR-020).
- [ ] T034 Update root `README.md` short section referencing demo (→ FR-021 discoverability).
- [ ] T035 Add feature quickstart doc `docs/dev/issues/classic-interactions-ppo/quickstart.md` (→ FR-018 extended docs).
- [ ] T036 Add data-model doc `docs/dev/issues/classic-interactions-ppo/data-model.md` describing EpisodeSummary schema (→ FR-020, FR-018).
- [ ] T037 Add usage example snippet to `docs/ENVIRONMENT.md` linking to demo (→ FR-018 cross-link).
- [ ] T038 Add performance notes subsection referencing dry_run optimization (→ FR-015 documentation).
- [ ] T039 Ruff & type check clean sweep; fix new warnings (Quality gate) (→ Quality).
- [ ] T040 Final smoke: run deterministic seeds test twice ensuring identical output hash (→ Regression guard).

## Phase 5: Optional / Backlog (Post-MVP; Clarification Needed)
Tracked but not required for initial acceptance.
- [ ] T041 (FR-022) Multi-scenario chaining: extend demo to iterate a scenario list; design doc update & new test.
- [ ] T042 (FR-022) Add command-level constant `SCENARIO_LIST` override + test ensuring order preserved.
- [ ] T043 (FR-023) Frame sampling / decimation: introduce `FRAME_STRIDE` constant controlling recorded frames.
- [ ] T044 (FR-023) Test ensuring output video shorter & frame count approx original/stride.
- [ ] T045 (FR-023) Documentation update for frame sampling trade-offs.
- [ ] T046 Benchmark with frame stride to confirm performance gain annotation.

## Dependencies Summary
Tests-first gating: T005–T015 must exist (and initially fail if implementation absent) before starting T016.

Key chains:
- T005–T015 → T016–T031 (core build)
- T016 (dry_run behavior) → T025 (performance optimization)
- T017 (seed determinism) → T040 (final deterministic hash smoke)
- T019 (summary builder) → T027 (print helper), T030 (typing), T036 (data-model doc)
- T020 (recording guard) → T014 (recording success test) may be un-skipped once implemented
- T023 (logging verbosity) → T032 (overlay/log tests)
- T031 (CHANGELOG final) after all core tasks complete

Optional backlog (T041–T046) depends on completion of T016–T031 baseline features.

## Parallel Execution Notes
Example initial parallel batch (after T001–T004):
T005, T006, T007, T008, T009, T010, T011, T012, T013, T014, T015 (all different new test files)

During core implementation, parallel groups (distinct files):
- Group A: T016 (examples script adjustments) & T017 (loader if separate) should be sequential (same conceptual area) → not parallel.
- Group B (can parallel after T019 ready): T026 overlay (examples script), T027 summary printer (could be same file; make sequential if editing same region). If editing same file, remove [P] (already omitted to enforce sequence).

## Parallel Suitability Flags
Explicit [P] markers only on tasks guaranteed separate files. If combining tasks into same file during implementation, remove [P] at execution time to avoid merge conflicts.

## Validation Checklist
- [ ] All FR-001..FR-021 mapped to at least one task
- [ ] Tests precede implementation tasks covering same FRs
- [ ] Each [P] task isolated by file path
- [ ] Optional tasks clearly segregated
- [ ] Episode summary schema documented & typed

## FR ↔ Task Mapping (MVP)
(High level – detailed one-to-many handled in descriptions)
- FR-001 Basic run: T005, T016
- FR-002 Deterministic ordering: T006, T017
- FR-003 Reproducibility: T006, T017, T040
- FR-004 Dry run fast path: T005, T016, T025
- FR-005 Scenario validation: T008, T018
- FR-006 Error messaging invalid scenario: T008, T018
- FR-007 Model load failure clarity: T011, T021
- FR-008 Graceful recording absence: T007, T020
- FR-009 Recording artifact generation: T014, T020
- FR-010 Structured episode summary: T005, T009, T019
- FR-011 Consistent keys: T009, T019, T030
- FR-012 Headless safety: T010, T024
- FR-013 Reward safety integration: T012, T022
- FR-014 Logging control: T013, T023, T032
- FR-015 Performance baseline: T015, T025, T038
- FR-016 Overlay clarity: T026, T032
- FR-017 Human-readable summary output: T027
- FR-018 API & docs: T028, T035, T036, T037
- FR-019 Robust constants validation: T029
- FR-020 Typing clarity: T030, T033, T036
- FR-021 Changelog / discoverability: T031, T034

Optional:
- FR-022 Multi-scenario chaining: T041, T042
- FR-023 Frame sampling: T043–T046

## Completion Definition (MVP)
MVP done when: T001–T040 completed, FR-001..FR-021 satisfied, tests green, ruff & type checks clean, docs & changelog updated.

---
Notes:
- Adjust [P] markers dynamically if implementation collocates tasks.
- Keep commits small: one task per commit where feasible.
- If FR clarifications alter scope (esp. FR-022/FR-023) update this file with delta section.
