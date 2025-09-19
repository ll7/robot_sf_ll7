# Tasks: Social Navigation Benchmark Platform Foundations

**Input**: Design documents from `specs/120-social-navigation-benchmark-plan/`  
**Prerequisites**: plan.md (present), research.md (present), data-model.md (present), contracts/ (present - schemas pending creation)

## Execution Flow Mapping
Plan + research + data model parsed → entities & required schemas enumerated → tasks ordered: Setup → Contract Tests (failing first) → Core Models → Services/Computation Modules → CLI / Runner Integration → Aggregation & Visualization → Resume & Provenance → SNQI Tooling → Polish.

## Legend
Format: `[ID] [P?] Description (File(s))`
- [P] = Parallelizable (different files, no dependency conflicts)
- No [P] = Sequential or depends on prior artifacts in same file
- TDD: All contract + integration tests precede implementation tasks they validate

---
## Phase 3.1: Setup
- [ ] T001 Ensure fast-pysf submodule initialized guard check (add import-time error message) (`robot_sf/sim/__init__.py`)
- [ ] T002 Add episode schema version constant `EPISODE_SCHEMA_VERSION = "v1"` (`robot_sf/benchmark/constants.py`)
- [ ] T003 [P] Add new package module scaffolds: `robot_sf/benchmark/schemas/`, `robot_sf/benchmark/identity/`, `robot_sf/benchmark/snqi/` (empty `__init__.py`)
- [ ] T004 [P] Add docs index link section "Social Navigation Benchmark" (`docs/README.md`)
- [ ] T005 Define central collision & near-miss threshold constants (`robot_sf/benchmark/constants.py`)
- [ ] T006 Define force comfort threshold constant and expose via metrics module (`robot_sf/benchmark/constants.py`)

## Phase 3.2: Contract & Integration Tests (Must FAIL initially)
Schema contract test files target non-existent schema definitions; they must fail before implementation.
- [ ] T010 [P] Contract test: episode schema validation (`tests/contract/test_episode_schema.py`)
- [ ] T011 [P] Contract test: scenario matrix schema validation (`tests/contract/test_scenario_matrix_schema.py`)
- [ ] T012 [P] Contract test: aggregate summary schema (`tests/contract/test_aggregate_schema.py`)
- [ ] T013 [P] Contract test: SNQI weights schema (`tests/contract/test_snqi_weights_schema.py`)
- [ ] T014 [P] Contract test: resume manifest schema (`tests/contract/test_resume_manifest_schema.py`)
- [ ] T015 Integration test: run single scenario → produce 1 episode line (`tests/integration/test_single_episode_run.py`)
- [ ] T016 Integration test: resume run skips existing episode ids (`tests/integration/test_resume_behavior.py`)
- [ ] T017 Integration test: aggregation with bootstrap outputs CI keys (`tests/integration/test_aggregation_bootstrap.py`)
- [ ] T018 Integration test: SNQI recompute + application populates snqi metric (`tests/integration/test_snqi_recompute_apply.py`)
- [ ] T019 Integration test: figure orchestrator writes expected artifacts set (`tests/integration/test_figure_orchestrator.py`)
- [ ] T020 Integration test: identity hash stable across runs (`tests/integration/test_episode_identity_stability.py`)

## Phase 3.3: Core Schemas & Identity
- [ ] T030 Implement `episode.schema.v1.json` (`specs/120-social-navigation-benchmark-plan/contracts/episode.schema.v1.json` & copy to `robot_sf/benchmark/schemas/episode.schema.v1.json`)
- [ ] T031 Implement `scenario-matrix.schema.v1.json` (same twin locations) 
- [ ] T032 Implement `aggregate.schema.v1.json`
- [ ] T033 Implement `snqi-weights.schema.v1.json`
- [ ] T034 Implement `resume-manifest.schema.v1.json`
- [ ] T035 Identity hash helper (canonical JSON serialization, stable ordering) (`robot_sf/benchmark/identity/hash_utils.py`)
- [ ] T036 Episode ID generation integration in runner (`robot_sf/benchmark/runner.py`)

## Phase 3.4: Models / Data Structures
- [ ] T040 [P] Dataclass / Typed structures: ScenarioSpec (`robot_sf/benchmark/types.py`)
- [ ] T041 [P] Dataclass: EpisodeRecord skeleton (`robot_sf/benchmark/types.py`)
- [ ] T042 [P] Dataclass: MetricsBundle view / accessor (`robot_sf/benchmark/metrics/types.py`)
- [ ] T043 [P] Dataclass: SNQIWeights (`robot_sf/benchmark/snqi/types.py`)
- [ ] T044 [P] Dataclass: ResumeManifest (`robot_sf/benchmark/types.py`)
- [ ] T045 Metrics constants export & validation helpers (`robot_sf/benchmark/metrics/constants.py`)

## Phase 3.5: Metrics Computation Layer
- [ ] T050 Implement metrics computation functions (collisions, near-misses, path efficiency, force stats, smoothness, energy) (`robot_sf/benchmark/metrics/compute.py`)
- [ ] T051 Add unit tests for edge cases (zero-length trajectory, no pedestrians) (`tests/unit/test_metrics_edge_cases.py`)
- [ ] T052 Implement SNQI computation function (weights applied, missing weights => skip) (`robot_sf/benchmark/snqi/compute.py`)
- [ ] T053 Unit test SNQI weighting & sensitivity with synthetic metrics (`tests/unit/test_snqi_weights.py`)

## Phase 3.6: Runner & Resume
- [ ] T060 Integrate schema validation & identity hashing into batch runner (`robot_sf/benchmark/runner.py`)
- [ ] T061 Implement resume manifest creation & update logic (`robot_sf/benchmark/resume/manifest.py`)
- [ ] T062 Unit test manifest invalidation triggers (`tests/unit/test_resume_manifest.py`)
- [ ] T063 Ensure parallel workers parent-only write enforcement (`robot_sf/benchmark/runner.py`)

## Phase 3.7: Aggregation & Bootstrap
- [ ] T070 Aggregation module: compute mean/median/p95 per metric (`robot_sf/benchmark/aggregate.py`)
- [ ] T071 Bootstrap CI logic (resampling) (`robot_sf/benchmark/bootstrap.py`)
- [ ] T072 Integrate aggregation + bootstrap CLI entry points (`robot_sf/benchmark/cli.py`)
- [ ] T073 Unit test bootstrap statistical shape (CI array length=2, ordering) (`tests/unit/test_bootstrap_shapes.py`)

## Phase 3.8: SNQI Tooling
- [ ] T080 Implement weight recompute CLI (read baseline stats → produce weight file) (`robot_sf/benchmark/snqi/cli.py`)
- [ ] T081 Implement ablation CLI (remove each component → ranking deltas) (`robot_sf/benchmark/snqi/cli.py`)
- [ ] T082 Unit test recompute deterministic with seed (`tests/unit/test_snqi_recompute_determinism.py`)

## Phase 3.9: Figure Orchestrator & Visualization
- [ ] T090 Implement figure orchestrator script enhancements (auto out dir naming) (`scripts/generate_figures.py`)
- [ ] T091 Force-field figure generation integration (existing logic adapt to orchestrator) (`scripts/generate_figures.py` + `robot_sf/benchmark/figures/force_field.py`)
- [ ] T092 Scenario thumbnails montage builder module (`robot_sf/benchmark/figures/thumbnails.py`)
- [ ] T093 Unit test orchestrator required artifact list generation (`tests/unit/test_figure_orchestrator_requirements.py`)

## Phase 3.10: CLI Enhancements
- [ ] T100 Add `validate-config` subcommand hooking scenario matrix schema (`robot_sf/benchmark/cli.py`)
- [ ] T101 Add `list-scenarios` subcommand (`robot_sf/benchmark/cli.py`)
- [ ] T102 Add `baseline` subcommand (stats computation) (`robot_sf/benchmark/cli.py`)
- [ ] T103 Add `aggregate` subcommand (with CI flags) (`robot_sf/benchmark/cli.py`)
- [ ] T104 Add `seed-variance` diagnostic subcommand (`robot_sf/benchmark/cli.py`)

## Phase 3.11: Baseline Planner Interface
- [ ] T110 Define planner interface protocol (step(obs)->action) (`robot_sf/baselines/interface.py`)
- [ ] T111 Adapt SocialForce baseline to interface (`robot_sf/baselines/social_force_planner.py`)
- [ ] T112 Adapt PPO model loader baseline (`robot_sf/baselines/ppo_planner.py`)
- [ ] T113 Implement Random planner (`robot_sf/baselines/random_planner.py`)
- [ ] T114 Unit tests: planner action shape & type (`tests/unit/test_planner_interface.py`)

## Phase 3.12: Documentation & Quickstart Alignment
- [ ] T120 Add benchmark section to `docs/README.md` (if not already covered) (`docs/README.md`)
- [ ] T121 Add metrics specification doc (`docs/ped_metrics/metrics_spec.md`)
- [ ] T122 Add SNQI weights artifact provenance doc (`docs/snqi-weight-tools/weights_provenance.md`)
- [ ] T123 Add figure naming scheme doc & link (`docs/dev/issues/figures-naming/design.md` update + link)
- [ ] T124 Update quickstart if CLI flag names differ post-implementation (`specs/120-social-navigation-benchmark-plan/quickstart.md`)

## Phase 3.13: Polish & Quality Gates
- [ ] T130 Ruff + formatting pass (ensure repo clean) (root)
- [ ] T131 Type checking pass; refine type hints in new modules (root)
- [ ] T132 Add smoke validation script references to docs (update `docs/dev_guide.md` if needed)
- [ ] T133 Performance smoke test: measure steps/sec baseline; record in docs (`docs/performance_notes.md`)
- [ ] T134 Final reproducibility script: end-to-end run & figure regeneration (`scripts/benchmark_repro_check.py`)
- [ ] T135 CI: add small benchmark smoke job configuration (`.github/workflows/ci.yml`)

## Phase 3.14: Optional / Deferred
- [ ] T140 (Deferred) ORCA baseline stub & decision record (`docs/dev/issues/adding_orca.md`)
- [ ] T141 (Deferred) Real-data calibrated scenario variant (extend scenario matrix)

---
## Dependencies Summary
- T010–T020 must run (and fail) before T030+ (schemas / implementation).
- Schemas (T030–T034) must precede identity + runner integration (T035–T036).
- Metrics computations (T050) before SNQI computation (T052) and aggregation (T070+ depends on metrics existing in episodes).
- Resume manifest (T061) depends on identity (T035) and runner base (T036).
- Aggregation (T070–T072) depends on episode records (T036) and metrics (T050).
- SNQI tooling (T080–T082) depends on aggregation outputs & weights data model (T043, T052).
- Figure orchestrator tasks (T090–T093) depend on episodes + aggregation + metrics.
- Baseline planner tests (T114) depend on interface + planner implementations (T110–T113).

## Parallelizable Groups Examples
Group A (post-setup, contract tests): T010–T014 concurrently.  
Group B (integration tests): T015–T020 concurrently (some may share runner; ensure isolated temp outputs).  
Group C (dataclasses): T040–T044 concurrently.  
Group D (docs batch at end): T121–T124 concurrently.  

## Validation Checklist
- [ ] All schema contracts have tests (T010–T014)
- [ ] All entities have dataclass tasks (T040–T044)
- [ ] Tests precede implementation (ordering honored)
- [ ] Identity & resume implemented before aggregation
- [ ] SNQI optimization & ablation tasks defined
- [ ] Documentation tasks cover metrics, provenance, figures naming
- [ ] CI integration task present

---
Generated: 2025-09-19
