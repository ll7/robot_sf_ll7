# Tasks: Enhance Benchmark Visual Artifacts (SimulationView Replay, Encoding, Schema Validation)

Branch: `127-enhance-benchmark-visual`
Spec: `specs/127-enhance-benchmark-visual/spec.md`
Plan: `specs/127-enhance-benchmark-visual/plan.md`
Design Artifacts: research.md, data-model.md, contracts/*.schema.json, quickstart.md

Legend:
- [P] = Can run in parallel with other [P] tasks (different files / no ordering dependency)
- Status columns to be updated during execution (NOT started / In progress / Done)
- Keep TDD ordering: write failing tests first where feasible

- [Ordering Rationale](#ordering-rationale)
- [Task List](#task-list)
  - [Setup \& Constants](#setup--constants)
  - [Schema \& Determinism Tests (Pre-Implementation)](#schema--determinism-tests-pre-implementation)
  - [Replay Adapter \& Data Structures](#replay-adapter--data-structures)
  - [SimulationView \& Encoding Pipeline](#simulationview--encoding-pipeline)
  - [SimulationView Enhancement Delta (Added 2025-09-22)](#simulationview-enhancement-delta-added-2025-09-22)
  - [Performance \& Budget Instrumentation](#performance--budget-instrumentation)
  - [Tests for Core Paths (Post Implementation Core)](#tests-for-core-paths-post-implementation-core)
  - [Documentation \& Demo Updates](#documentation--demo-updates)
  - [Final Polish \& Validation](#final-polish--validation)
- [Parallel Execution Guidance](#parallel-execution-guidance)
- [Environment / Flags](#environment--flags)
- [Acceptance Mapping](#acceptance-mapping)
- [Notes](#notes)


## Ordering Rationale
1. Setup & constants foundation
2. Tests for schemas, skip codes, and selection determinism (to drive implementation)
3. Replay adapter & detection utilities (enablers for SimulationView path)
4. SimulationView renderer + encoding pipeline (core feature)
5. Performance & memory instrumentation
6. Integration into existing visuals orchestration
7. Documentation & demo updates
8. Final validation & polish

## Task List

### Setup & Constants
| ID | Task | Details / File Paths | Depends | Parallel | Status |
|----|------|----------------------|---------|----------|--------|
| T001 | Create constants module | `robot_sf/benchmark/full_classic/visual_constants.py` with note enums & renderer names | - | [P] | Done |
| T002 | Add import + usage placeholder in visuals orchestrator | Update `robot_sf/benchmark/full_classic/visuals.py` to reference constants (no behavior change) | T001 |  | Done |
| T003 | Add optional deps detection util | `robot_sf/benchmark/full_classic/visual_deps.py` functions: `has_pygame()`, `has_moviepy()`, `moviepy_ready()` (ffmpeg check attempt) | - | [P] | Done |

### Schema & Determinism Tests (Pre-Implementation)
| ID | Task | Details / File Paths | Depends | Parallel | Status |
|----|------|----------------------|---------|----------|--------|
| T010 | Test: video schema validity | New `tests/visuals/test_video_schema_validation.py`: load schema + sample good/bad objects | contracts schema | [P] | Done |
| T011 | Test: plot schema validity | `tests/visuals/test_plot_schema_validation.py` | contracts schema | [P] | Done |
| T012 | Test: performance schema validity | `tests/visuals/test_performance_schema_validation.py` | contracts schema | [P] | Done |
| T013 | Test: deterministic episode selection | `tests/visuals/test_deterministic_selection.py` simulate list ordering + selection logic (mock) | T001 | [P] | Done |
| T014 | Test: skip note constants stable | `tests/visuals/test_skip_notes.py` assert exported values set membership | T001 | [P] | Done |

### Replay Adapter & Data Structures
| ID | Task | Details / File Paths | Depends | Parallel | Status |
|----|------|----------------------|---------|----------|--------|
| T020 | Implement ReplayState dataclass | `robot_sf/benchmark/visuals/replay.py` with validation helper | T001 |  | Done |
| T021 | Hook capture in benchmark loop (if missing) | Identify episode data source; modify `robot_sf/benchmark/orchestrator.py` or related collector to store minimal replay arrays (guarded by flag) | T020 |  | Done |
| T022 | Adapter: extract replay states post-run | Extend `visuals.py` to build list of `ReplayState` objects from stored episode info | T021 |  | Done |
| T023 | Test: insufficient replay state skip | `tests/visuals/test_insufficient_replay_skip.py` create malformed ReplayState -> expect skip note | T022 | [P] | Done |

### SimulationView & Encoding Pipeline
| ID | Task | Details / File Paths | Depends | Parallel | Status |
|----|------|----------------------|---------|----------|--------|
| T030 | Utility: simulation view availability probe | In `deps.py` implement probe caching result | T003 |  | Done |
| T031 | Implement SimulationView frame generator | `visuals/render_sim_view.py` generate frame (numpy array) from ReplayState | T030 T022 |  | Done |
| T032 | Implement synthetic fallback path parity refactor | Ensure existing synthetic code moved/cleaned to `visuals/render_synthetic.py` | T001 | [P] | Done (2025-09-22) |
| T033 | Implement moviepy encoding wrapper (streaming) | `visuals/encode.py` function with generator + memory sampling hook | T031 |  | Done |
| T034 | Memory sampler (optional psutil) | Helper in `encode.py` or `perf.py` thread sampling RSS; returns peak MB | T033 |  | Done |
| T035 | Integrate encoding + renderer selection | Update `visuals.py` main flow: choose simulation_view vs synthetic vs skip; write artifact entries | T033 T032 |  | Done (2025-09-22) |
| T036 | Failure cleanup logic | Ensure partial file removal + status=failed note path implemented | T035 |  | Done |

### SimulationView Enhancement Delta (Added 2025-09-22)
| ID | Task | Details / File Paths | Depends | Parallel | Status |
|----|------|----------------------|---------|----------|--------|
| T037 | VisualizableSimState builder adapter | New helper `robot_sf/benchmark/full_classic/state_builder.py` mapping ReplayEpisode -> minimal VisualizableSimState (poses, ped positions if available) | T022 T031 |  | Done (2025-09-22) |
| T038 | Invoke real render loop | Update `render_sim_view.py` to construct state and call `_sim_view.render(state)` (guard missing fields -> skip note) | T037 |  | Done (2025-09-22) |
| T039 | Replay enrichment (ped + robot trajectories) | Extend capture to store robot pose, pedestrian positions, actions per step; update `replay.py` schema | T021 |  | Not started |
| T039 | Replay enrichment (ped + robot trajectories) | Extend capture to store robot pose, pedestrian positions, actions per step; update `replay.py` schema | T021 |  | Done (2025-09-22) |
| T040A | (Split) Timing instrumentation (moved) | Refine T040: separate render timing (ms/frame) from encode timing; add to performance manifest | T040 | [P] | Not started |
| T041A | Cleanup & exit handling | Ensure `_sim_view.exit_simulation()` invoked after generator exhaustion; add try/finally in `generate_frames` | T038 | [P] | Done (2025-09-22) |
| T050A | Test: SimulationView placeholder guard | Add test asserting placeholder (pre-state-builder) produces synthetic frames with note indicating 'placeholder' until T038 done | T031 | [P] | Not started |
| T050B | Test: SimulationView real render | After T038/T037, assert frames differ from flat gradient and `_sim_view.render` was invoked (mock/spies) | T038 |  | Not started |
| T052A | Test: renderer toggle flag | Add CLI/config flag `--renderer=synthetic|sim-view` ensuring selection logic respects explicit choice | T035 | [P] | Not started |
| T060A | Docs: video rendering lifecycle | Section in `docs/benchmark_visuals.md` describing synthetic vs sim-view pipeline, state requirements, fallback ladder | T060 |  | Not started |
| T063A | Dependency matrix update | Add SimulationView row specifying additional replay fields required | T063 | [P] | Not started |
| T074A | Perf benchmark (render) | Measure render+encode split timings (N=50 frames) and document baseline numbers; store under `results/visual_perf.json` | T074 |  | Not started |


### Performance & Budget Instrumentation
| ID | Task | Details / File Paths | Depends | Parallel | Status |
|----|------|----------------------|---------|----------|--------|
| T040 | Timing instrumentation for plots & first video | Amend `visuals.py` measuring durations | T035 |  | Done (2025-09-22) |
| T041 | Memory over-budget flag logic | Add logic after first video encode to set `memory_over_budget` if peak >100 | T034 T035 |  | Done (2025-09-22) |
| T042 | Schema validation helper | `visuals/validation.py` (conditional jsonschema usage) | T010 T011 T012 | [P] | Done (2025-09-22) |
| T043 | Integrate validation call (dev/test mode) | Invoke after manifests written with env var toggle | T042 T035 |  | Done (2025-09-22) |

### Tests for Core Paths (Post Implementation Core)
| ID | Task | Details / File Paths | Depends | Parallel | Status |
|----|------|----------------------|---------|----------|--------|
| T050 | Test: simulation_view success path | `tests/visuals/test_simulation_view_success.py` (skip if pygame/moviepy missing) verifies renderer field and mp4 exists | T035 |  | |
| T051 | Test: synthetic fallback path | `tests/visuals/test_synthetic_fallback.py` simulate missing pygame; expect renderer synthetic | T035 | [P] | |
| T052 | Test: moviepy-missing skip | `tests/visuals/test_moviepy_missing.py` monkeypatch detection to False; expect skip notes | T033 | [P] | |
| T052A | Test: renderer toggle flag | Added `tests/test_full_classic_visuals_renderer_toggle.py` verifying forced synthetic and sim-view modes (implemented 2025-09-22) | T035 | [P] | Done (2025-09-22) |
| T053 | Test: performance flags | `tests/visuals/test_performance_flags.py` mock timings to trigger over_budget booleans | T040 | [P] | |
| T054 | Test: memory sampling over-budget flag | `tests/visuals/test_memory_over_budget.py` simulate peak >100MB via monkeypatch sampler | T041 | [P] | |
| T055 | Test: validation errors raise in dev mode | `tests/visuals/test_schema_validation_mode.py` corrupt manifest then assert validation exception/log | T043 | [P] | |

### Documentation & Demo Updates
| ID | Task | Details / File Paths | Depends | Parallel | Status |
|----|------|----------------------|---------|----------|--------|
| T060 | Add documentation section | Update `docs/benchmark_full_classic.md` or create new `docs/benchmark_visuals.md` + link in `docs/README.md` | T035 T040 |  | |
| T061 | Update demo script docstring | `examples/demo_full_classic_benchmark.py` reflect SimulationView path & fallback table | T050 | [P] | |
| T062 | Update CHANGELOG.md | Add enhancement entry referencing feature branch & key capabilities | T050 |  | |
| T063 | Add dependency matrix table | Ensure matrix included in new docs page (pygame/moviepy/jsonschema/psutil) | T060 |  | |

### Final Polish & Validation
| ID | Task | Details / File Paths | Depends | Parallel | Status |
|----|------|----------------------|---------|----------|--------|
| T070 | Ruff + format pass | Run lint/format; resolve issues | All impl | [P] | |
| T071 | Type checking adjustments | Add/ refine type hints for new modules; fix ty check warnings | All impl | [P] | |
| T072 | Run test suites (core + visuals) | Ensure new tests pass headless; mark skips appropriately | T050 T051 T052 T053 T054 T055 |  | |
| T073 | Validation scripts smoke run | Run existing validation scripts verifying no regression | T072 |  | |
| T074 | Performance sanity benchmark | Optional: measure single encode time & memory; document in PR | T072 | [P] | |
| T075 | Update plan progress & close feature | Mark tasks completed in plan & add summary section to spec or PR | T073 |  | |

## Parallel Execution Guidance
- Safe early parallel group: T001, T003
- Schema test group: T010–T014 can run together once constants file exists
- Renderer/encoding core (T031–T036) must remain sequential
- Post-core test group: T051, T052, T053, T054, T055 in parallel after their deps
- Polish group: T070, T071, T074 can parallelize after implementation

## Environment / Flags
- Env var `ROBOT_SF_VALIDATE_VISUALS=1` triggers schema validation at runtime
- Headless pygame: set `SDL_VIDEODRIVER=dummy`

## Acceptance Mapping
| FR | Coverage Tasks |
|----|----------------|
| FR-001 | T031 T033 T035 T050 |
| FR-002 | T032 T035 T051 |
| FR-003 | T013 T035 |
| FR-004 | T035 T042 T043 T050–T055 |
| FR-005 | Existing plots + T040 T042 T043 |
| FR-006 | T042 T043 T055 |
| FR-007 | T033 T035 T052 |
| FR-008 | T020 T022 T023 T035 |
| FR-009 | T040 T043 T053 |
| FR-010 | T040 T053 |
| FR-011 | T040 T053 |
| FR-012 | T033 T034 T041 T054 |
| FR-013 | T001 T014 T035 T052 T023 |
| FR-014 | T035 T050 |
| FR-015 | T050 |
| FR-016 | T051 |
| FR-017 | T052 |
| FR-018 | T060 T063 |
| FR-019 | T061 |
| FR-020 | (No schema change) Verified by absence of episode schema edits (review in PR) |

## Notes
- Keep synthetic path behavior unchanged except relocation/cleanup.
- Ensure all frames generation is lazy; avoid retaining full frame list.
- Use ultrafast preset to respect performance soft budget.
- Memory sampling optional if psutil missing (artifact fields omitted) — tests should account for conditional presence.

---
Generated on 2025-09-20.
