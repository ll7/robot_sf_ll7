---
description: "Task list for Architectural decoupling and consistency overhaul"
---

# Tasks: Architectural decoupling and consistency overhaul

**Input**: Design documents from `/specs/149-architectural-coupling-and/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Optional. The feature spec defines independent test criteria; this plan does not include explicit test tasks unless later requested.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: User story label (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path conventions
- Library code: `robot_sf/` and `fast-pysf/` (subtree)
- Feature docs: `specs/149-architectural-coupling-and/`

---

## Phase 1: Setup (Shared infrastructure)

Purpose: Ensure docs are linked and developer ergonomics are in place.

- [x] T001 [P] Link new docs in `docs/README.md` under an "Architecture" or "Design features" section.
- [x] T002 [P] Add pointers in `CHANGELOG.md` (Unreleased) about upcoming facade/registry additions (no breaking changes).
- [x] T003 [P] Add anchors from `specs/149-architectural-coupling-and/quickstart.md` to `docs/README.md`.

---

## Phase 2: Foundational (Blocking prerequisites)

Purpose: Core infrastructure that MUST be complete before ANY user story work.

- [ ] T004 [US1] Define Simulator Facade contract in code: `robot_sf/sim/facade.py` (create/reset/step/get_state) matching `contracts/simulator-facade.md`.
- [x] T005 [P] [US1] Add simulator backend registry: `robot_sf/sim/registry.py` with `register_backend(key, factory)` and `get_backend(key)`.
- [x] T006 [US1] Implement Fast-PySF adapter backend: `robot_sf/sim/backends/fast_pysf_backend.py` that conforms to the facade and wraps current usage.
- [x] T007 [P] [US1] Provide a tiny "dummy" simulator backend for smoke: `robot_sf/sim/backends/dummy_backend.py` (constant positions, deterministic seed handling).
- [x] T008 [US1] Wire env factories to facade: update `robot_sf/gym_env/environment_factory.py` to resolve backend via unified config key using the registry (preserve existing signatures).
- [x] T009 [US2] Define Sensor base interface and registry: `robot_sf/sensor/base.py` and `robot_sf/sensor/registry.py` per `contracts/sensor-registry.md`.
- [x] T010 [P] [US2] Ensure existing `robot_sf/sensor/sensor_fusion.py` accepts a list of sensors from config without hardcoding types; add a thin adapter if needed (new module `robot_sf/sensor/fusion_adapter.py`).
- [x] T011 [US3] Establish error policy helpers: `robot_sf/common/errors.py` with helpers to raise fatal errors with remediation and to log soft-degrade warnings.
- [x] T012 [US4] Extend unified config schema for backend/sensors: update `robot_sf/gym_env/unified_config.py` (or the existing config module) to include `backend: str` and `sensors: list[dict]`; add defaults.
- [x] T013 [P] [US4] Add config validation and conflict detection function: `robot_sf/gym_env/config_validation.py` (unknown keys, mutually exclusive options).
- [x] T014 [US4] Add resolved-config dump (loggable representation) at env creation for reproducibility.

Checkpoint: Facade + registries + config validation are in place; user story phases can start.

---

## Phase 3: User Story 1 — Swap simulation backends without touching env code (Priority: P1) 🎯 MVP

Goal: Select simulator backend by config; environment code remains unchanged.

Independent Test: Provide two minimal backends (fast-pysf and dummy) implementing the facade; switch via config and run env reset→step smoke without editing env code.

### Implementation

- [x] T015 [US1] Register Fast-PySF backend in `robot_sf/sim/registry.py` during module import or dedicated setup function.
- [x] T016 [P] [US1] Register Dummy backend in `robot_sf/sim/registry.py`.
- [x] T017 [US1] Update `robot_sf/gym_env/environment_factory.py` to read `config.backend` and instantiate the facade-selected backend, replacing direct imports.
- [x] T018 [P] [US1] Update `docs/dev_guide.md` (brief backend selection note with example snippet).
- [x] T019 [US1] Add a minimal example under `examples/demo_refactored_environments.py` (or update existing) to show backend toggle.

Checkpoint: Backend swap works with 0 env code changes beyond the factory.

---

## Phase 4: User Story 2 — Add a new sensor without editing fusion or simulator (Priority: P1)

Goal: Implement a new sensor by conforming to a small interface and register via config; no changes to SensorFusion or simulator internals.

Independent Test: Implement a dummy sensor returning constants; enable via config; fusion consumes it and env produces a stable observation dict.

### Implementation

- [x] T020 [US2] Implement `robot_sf/sensor/dummy_constant.py` exposing Sensor base API; register with `sensor/registry.py`.
- [x] T021 [P] [US2] Extend `robot_sf/gym_env/env_util.py` or factory wiring to build sensor instances from `config.sensors` using the registry and inject into SensorFusion without code changes to fusion internals.
- [ ] T022 [US2] Update `robot_sf/sensor/sensor_fusion.py` only if necessary to accept abstract Sensor inputs (avoid hardcoding specific types; prefer adapter layer from T010).
- [ ] T023 [P] [US2] Document sensor registration in `specs/149-architectural-coupling-and/quickstart.md` with a second example (new sensor).

Checkpoint: New sensor can be added via config; fusion/simulator code unchanged.

---

## Phase 5: User Story 3 — Graceful error handling with actionable messages (Priority: P2)

Goal: Fatal for required resources with remediation; warnings and soft-degrade for optional ones.

Independent Test: Remove a required asset → RuntimeError with path and quick fix. Optional component missing → WARNING and continue.

### Implementation

- [ ] T024 [US3] Apply error policy in env factories: wrap missing map/model loads with clear remediation messages (edit `robot_sf/gym_env/environment_factory.py`).
- [ ] T025 [P] [US3] Apply policy in simulator adapters: map loader failures produce actionable errors (edit `robot_sf/sim/backends/fast_pysf_backend.py`).
- [ ] T026 [US3] Apply policy in sensor initialization: unknown sensor names raise KeyError listing known names (implement via `sensor/registry.py`).
- [ ] T027 [P] [US3] Audit `fast-pysf/pysocialforce/` loading points; where applicable, translate exceptions at integration boundaries to actionable messages in our adapter.

Checkpoint: Errors are consistent and actionable across env/sim/sensors.

---

## Phase 6: User Story 4 — Consistent configuration with validation (Priority: P2)

Goal: Unified, schema-validated config with defaults and conflict detection; resolved-config dump.

Independent Test: Invalid backend/sensor names rejected with allowed alternatives; minimal config auto-defaults; resolved config logged.

### Implementation

- [ ] T028 [US4] Implement strict validation: unknown keys rejected; supply message with valid keys (edit `robot_sf/gym_env/config_validation.py`).
- [ ] T029 [P] [US4] Implement conflict checks (e.g., mutually exclusive image vs non-image stacks) in `config_validation.py`.
- [ ] T030 [US4] Integrate validation into env factory path before instantiation; on failure raise concise validation errors.
- [ ] T031 [P] [US4] Add resolved-config serializer (dict) and log at env creation.

Checkpoint: Unified config is enforced with friendly validation and a resolved dump.

---

## Phase N: Polish & cross-cutting concerns

- [ ] T032 [P] Documentation refresh across `docs/` and examples.
- [ ] T033 Code cleanup and dead import pruning in areas touched by adapters.
- [ ] T034 [P] Optional: small perf smoke to confirm no step-rate regressions (wire to existing scripts in `scripts/validation/`).
- [ ] T035 Security/robustness: ensure registries don’t allow re-registering under same key without explicit override flag.
- [ ] T036 [P] Link this feature’s docs from `docs/README.md` central point.

---

## Dependencies & execution order

### Phase dependencies
- Setup → Foundational → User Stories (3–6) → Polish
- All user stories depend on Foundational being complete
- US1 is MVP and should be delivered first; US2 can start after Foundational (in parallel if team capacity allows)

### User story dependencies
- US1 (P1): None (after Foundational)
- US2 (P1): None (after Foundational); integrates with US1 via env factory but independently testable
- US3 (P2): None (after Foundational); applies policy across components
- US4 (P2): None (after Foundational); validation hooks called by env factory

### Parallel opportunities
- Registry implementations (T005, T007, T010, T013) are parallelizable
- Backend adapters (T006) and env factory wiring (T008) can be developed in parallel with sensor registry (T009/T010) as long as interfaces are respected
- Error policy application (T024–T027) can be split by component owners
- Config validation (T028–T031) can be split between checks and wiring

---

## Implementation strategy

### MVP first (User Story 1 only)
1. Complete Foundational tasks T004–T008
2. Implement US1 tasks T015–T019
3. Validate backend toggle via quick example

### Incremental delivery
1. Deliver US1 (MVP)
2. Deliver US2 (new sensor via registry)
3. Deliver US3 (error policy)
4. Deliver US4 (validation + resolved config)

---

## Independent test criteria (from spec)
- US1: Two backends selectable by config; env reset→step smoke without env code edits
- US2: Add dummy sensor via config; fusion consumes it and outputs stable observation dict
- US3: Missing required asset → clear RuntimeError with remediation; optional missing → WARNING and continue
- US4: Invalid config rejected with allowed alternatives; minimal config defaults applied; resolved config logged
