# Tasks: Improve Environment Factory Ergonomics

Feature Dir: `specs/130-improve-environment-factory`  
Branch: `130-improve-environment-factory`

Legend: `[P]` = Can run in parallel with other `[P]` tasks (different files / no shared mutable code).  
Dependencies explicitly noted where sequencing required.  
All tasks follow Constitution Principles II, IV, VII, XII.

## Ordering Strategy
1. Baseline data capture (tests first where feasible).  
2. Introduce option dataclasses & contracts with failing tests (TDD).  
3. Implement deprecation + normalization layer.  
4. Update factories incrementally.  
5. Add docs & migration.  
6. Performance + final validation.  

## Task List

### Setup & Baseline
**T001** [X]: Create baseline timing helper script (`scripts/perf/baseline_factory_creation.py`) to measure current env creation (30 iterations each) → output JSON to `results/factory_perf_baseline.json`.  
Deps: None.  
Accept: File created; runs and prints mean & p95; no imports fail.

**T002 [P] [X]**: Add test `tests/factories/test_current_factory_signatures.py` asserting existing signatures (pre-change snapshot) to detect unintended drift during refactor.  
Deps: None.  
Accept: Test passes and fails meaningfully if signature changes prematurely.

**T003 [P] [X]**: Grep usage of `make_*_env` across repo; generate `specs/130-improve-environment-factory/deprecation_map_raw.txt` listing legacy kwargs & frequency.  
Deps: None.  
Accept: File contains frequency table.

### Option Dataclasses & Contracts (Failing Tests First)
**T004 [X]**: Create `robot_sf/gym_env/options.py` module with placeholder (unimplemented) `RenderOptions`, `RecordingOptions` dataclasses (slots, type hints) – minimal stub.  
Deps: T002.  
Accept: Module imports; attributes docstring skeleton present.

**T005 [X]**: Add tests `tests/factories/test_option_dataclasses_api.py` expecting fields (from data-model). Initially failing until full implementation.  
Deps: T004.  
Accept: Test fails on missing validation logic initially.

**T006 [X]**: Implement full `RenderOptions` & `RecordingOptions` with validation (value checks, repr) in `options.py`; add factory methods `from_bool_and_path(record_video, video_path, existing)` for normalization.  
Deps: T005.  
Accept: Prior failing test passes; type hints clean.

### Deprecation & Normalization Layer
**T007 [X]**: Create `robot_sf/gym_env/_factory_compat.py` implementing `DeprecationMap` and function `apply_legacy_kwargs(kwargs, strict: bool)` returning (normalized_kwargs, warnings_emitted).  
Deps: T006, T003.  
Accept: Unit tests (next task) can import and manipulate.

**T008 [X]**: Add tests `tests/factories/test_deprecation_mapping.py` covering: mapped kw, unmapped strict error, permissive mode via env var, warning messages (Loguru capture).  
Deps: T007.  
Accept: Tests fail until factories integrate layer.

### Factory Refactor (Incremental)
**T009 [X]**: Refactor `environment_factory.py` to accept new `render_options` / `recording_options` parameters (without yet removing old paths); integrate call to `apply_legacy_kwargs`.  
Deps: T008.  
Accept: Code compiles; legacy usage still works; new params accepted (no behavior change yet).

**T010 [X]**: Implement normalization logic: precedence rules (explicit options over booleans) with logging; boolean convenience path auto-constructs RecordingOptions.  
Deps: T009.  
Accept: Add/extend tests `tests/factories/test_normalization.py` verifying precedence & warning emission.

**T011 [X]**: Add validation for incompatible combos (e.g., `record_video=True` + `recording_options.record=False` → enforce True + warning).  
Deps: T010.  
Accept: Extend normalization test or new `test_incompatible_combinations.py`.

**T012 [P] [X]**: Update pedestrian factory path (`make_pedestrian_env`) to mirror normalization logic; add test for required `robot_model` presence unaffected.  
Deps: T010.

**T013 [P] [X]**: (Optional placeholder) Insert stub for prospective `make_multi_robot_env` raising NotImplementedError with docstring referencing future extension—documented but not exposed publicly yet.  
Deps: T010.

### Logging & Diagnostics
**T014 [X]**: Add Loguru logging points (INFO creation, WARNING deprecation, WARNING precedence overrides) inside factories; unit test capturing log messages with `caplog` or Loguru sink.  
Deps: T011.

### Tests Enhancements & Performance Guard
**T015 [X]**: Add performance regression test `tests/perf/test_factory_creation_perf.py` comparing new creation time vs baseline JSON (< +5% mean). Skip if baseline file missing.  
Deps: T001, T011.

**T016 [P]**: Add (failing) RNG seed determinism test `tests/factories/test_seed_determinism.py` ensuring two env creations with same forthcoming `seed` param produce identical initial observations (and first sampled action where applicable). Fails until T030 implements seeding.  
Deps: T011.  
Accept: Test fails meaningfully until seed support exists; passes once implemented.

**T017 [P] [X]**: Add frame recording integration test reusing existing rendering test harness verifying `record_video=True` path still records frames with new options.  
Deps: T011.

### Documentation & Migration
**T018**: Create migration guide `docs/dev/issues/130-improve-environment-factory/migration.md` with before/after table and deprecation window timeline.  
Deps: T011.

**T019 [P]**: Update `docs/ENVIRONMENT.md` adding new parameter tables & option object examples.  
Deps: T018.

**T020 [P]**: Update `examples/` or add new example `examples/demo_factory_options.py` demonstrating combined render + recording options.  
Deps: T011.

**T021**: Update `CHANGELOG.md` (pending release) summarizing ergonomic additions, deprecation notice.  
Deps: T018.

### Quality & Cleanup
**T022 [P]**: Run and fix linters & type checks after refactor (Ruff + ty), adjust type hints if needed.  
Deps: T021.

**T023 [P]**: Consolidate tests ensuring no redundancy; remove obsolete signature snapshot test (T002) once new baseline accepted; update references.  
Deps: T022.

**T024**: Final performance re-run of baseline script (now post-change) writing `results/factory_perf_post.json`; compare and store diff into `specs/130-improve-environment-factory/perf_diff.md`.  
Deps: T015, T022.

**T025**: Review spec vs implementation: verify FR-001..FR-021 coverage; add `specs/130-improve-environment-factory/coverage_checklist.md` documenting each FR with file references.  
Deps: T024.

**T026**: Prepare PR description draft (`specs/130-improve-environment-factory/pr_outline.md`) including risks, migration summary, performance diff.  
Deps: T025.

### Optional / Stretch
**T027 [P]**: Prototype docstring auto-generation helper for factories reducing duplication (may defer if time constrained).  
Deps: T010.

**T028 [P]**: Add telemetry hook stub (no-op) for future usage analytics (documented off by default).  
Deps: T014.

### Remediation & Alignment (Post Analysis)
**T029 (CRITICAL)**: Reinstate legacy kwargs shim usage in all public factories (`apply_legacy_kwargs` invocation early). Honor env vars `ROBOT_SF_FACTORY_LEGACY` (permissive warn) & `ROBOT_SF_FACTORY_STRICT` (strict error). Extend tests for mapped rename, unmapped permissive warning, strict error.  
Deps: T008, T011.  
Accept: Tests assert correct warnings/errors; legacy params mapped.

**T030 (CRITICAL)**: Implement `seed: Optional[int]` param across factories; seed Python `random`, NumPy, env RNG; store `seed_applied` attribute. Update quickstart & docstrings.  
Deps: T016 (failing test), T011.  
Accept: T016 passes; signatures updated; docs adjusted.

**T031 (HIGH)**: Tighten performance guard to +5% mean (spec compliance). Update perf test constant; add `specs/130-improve-environment-factory/perf_diff.md` summarizing baseline vs current metrics.  
Deps: T015, T022.  
Accept: Perf test passes under new threshold; diff file committed.

**T032 (HIGH) [X]**: Expand docstrings (factories & option dataclasses) with param tables, precedence rules, pedestrian divergence, seeding sequence, performance note.  
Deps: T030.  
Accept: Docstrings present; lint/type checks clean.

**T033 (MEDIUM) [X]**: Edge-case tests: (1) headless + debug + recording interplay; (2) recording without video_path warns & buffers; (3) pedestrian explicit opt-out respected with seed.  
Deps: T030.  
Accept: Tests added & pass.

**T034 (MEDIUM) [X]**: Create `coverage_checklist.md` mapping FR-001..FR-021 → code/tests/status (Done/Deferred). Mark `max_episode_steps` deferred.  
Deps: T029, T030.  
Accept: Checklist complete.

**T035 (HIGH) [X]**: Migration guide `docs/dev/issues/130-improve-environment-factory/migration.md` (before/after, env vars, precedence divergence, seeding examples). Link from `docs/README.md` & quickstart.  
Deps: T029, T030.  
Accept: Links valid; guide thorough.

**T036 (LOW) [X]**: Import purity test `tests/factories/test_factory_import_purity.py` ensuring no side-effects (prints/file writes) on import.  
Deps: T029.  
Accept: Test passes; fails if side-effects appear.

**T037 (MEDIUM) [X]**: Logging enforcement test scanning `environment_factory.py` for stray `print(` excluding docstrings/comments.  
Deps: T032.  
Accept: Test fails if print introduced.

**T038 (LOW) [X]**: Update spec & quickstart: integrate seed param; mark `max_episode_steps` deferred; remove outdated references.  
Deps: T030.  
Accept: Spec & quickstart updated consistently.

## Parallel Execution Guidance
- Group 1 `[P]`: T002, T003 after T001.
- Group 2 `[P]`: T012, T013 after T010.
- Group 3 `[P]`: T016, T017 after T011.
- Group 4 `[P]`: T019, T020 after T018.
- Group 5 `[P]`: T022, T023 after T021.
- Group 6 `[P]`: T027, T028 (optional) after their deps.
- Group 7 `[P]`: T032 parallel with T031 post T030.
- Group 8 `[P]`: T033, T036, T037 parallel after T029 & T030.

## Requirement Coverage Matrix (Updated)
| FR | Tasks |
|----|-------|
| FR-001 | T009, T010, T011 |
| FR-002 | T014, T018, T019, T032 |
| FR-003 | T004, T006, T010 |
| FR-004 | T011, T014, T033 |
| FR-005 | T007, T008, T014, T029 |
| FR-006 | T007, T008, T029 |
| FR-007 | T002, T009 |
| FR-008 | T016 (failing test), T030 (implementation) |
| FR-009 | T010, T017, T033 |
| FR-010 | T018, T019, T032 |
| FR-011 | T018, T035 |
| FR-012 | T019, T020 |
| FR-013 | T005, T008, T010, T011, T016, T017, T015, T033 |
| FR-014 | T009, T036 |
| FR-015 | T014, T037 |
| FR-016 | T022 |
| FR-017 | T001, T015, T024, T031 |
| FR-018 | T006, T032 |
| FR-019 | T018, T035, T034 |
| FR-020 | T019, T032 |
| FR-021 | T007, T008, T029 |

## Exit Criteria
- All mandatory FRs implemented & verified.
- Performance diff within budget (<+5% mean creation time).
- Migration & docs updated and linked in central docs index.
- Deprecation warnings present for legacy params; strict mode works.
- Coverage checklist complete.
