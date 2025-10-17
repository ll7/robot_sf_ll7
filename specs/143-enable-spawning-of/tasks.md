# Tasks: Single Pedestrian Spawning and Control

**Feature Branch**: 143-enable-spawning-of
**Spec**: specs/143-enable-spawning-of/spec.md
**Plan**: specs/143-enable-spawning-of/plan.md

---

## Phase 1: Setup Tasks

- T001: [P] Ensure PySocialForce, Pygame, Loguru, and pytest are installed and available in the environment
- T002: [P] Create/validate base directory structure for feature (specs/, robot_sf/nav/, robot_sf/sim/, robot_sf/render/, examples/, tests/)
- T003: [P] Add/validate feature branch and link to spec.md and plan.md

---

## Phase 2: Foundational Tasks

- T004: Define SinglePedestrianDefinition dataclass in robot_sf/nav/map_config.py
- T005: Extend MapDefinition to include single_pedestrians: List[SinglePedestrianDefinition]
- T006: Add validation logic for unique IDs, mutually exclusive goal/trajectory, and coordinate checks
- T007: Update/validate unified config schema to support single pedestrian definitions

---

## Phase 3: User Story 1 (P1) - Spawn Single Pedestrian with Fixed Goal

- T008: [US1] Extend SVG/JSON map parsing in robot_sf/nav/svg_map_parser.py to detect and load single pedestrian definitions
- T009: [US1] Update simulator logic in robot_sf/sim/simulator.py to spawn single pedestrians with start and goal positions
- T010: [US1] Integrate single pedestrian into PySocialForce simulation (goal-based navigation)
- T011: [US1] Add/validate unit test: test_simulator_spawns_single_pedestrian_correctly (tests/test_single_pedestrian.py)
- T012: [US1] Add/validate unit test: test_pedestrian_moves_toward_goal (tests/test_single_pedestrian.py)
- T013: [US1] Add/validate example: example_single_pedestrian.py (examples/)
- T014: [US1] Add/validate error handling for invalid pedestrian definitions (missing fields, bad coordinates)
- T015: [US1] Add/validate documentation: update docs/maps.md with "Defining Single Pedestrians in SVG Maps"

---

## Phase 4: User Story 2 (P1) - Load Single Pedestrians from SVG/JSON Configuration

- T016: [US2] Extend SVG/JSON parser to support both goal and trajectory definitions
- T017: [US2] Add/validate error handling for invalid/ambiguous definitions (conflicting goal/trajectory)
- T018: [US2] Add/validate unit test: test_load_single_pedestrian_from_svg (tests/test_single_pedestrian.py)
- T019: [US2] Add/validate unit test: test_mapdefinition_contains_single_pedestrians (tests/test_single_pedestrian.py)
- T020: [US2] Add/validate example: example_multi_pedestrian.py (examples/)

---

## Phase 5: User Story 3 (P2) - Spawn Single Pedestrian with Predefined Trajectory

- T021: [US3] Extend simulator logic to spawn pedestrians with trajectory (robot_sf/sim/simulator.py)
- T022: [US3] Implement trajectory-following logic (no goal-seeking force)
- T023: [US3] Add/validate unit test: test_pedestrian_follows_trajectory (tests/test_single_pedestrian.py)
- T024: [US3] Add/validate error handling for trajectory edge cases (empty, unreachable, through obstacles)
- T025: [US3] Add/validate example: example_single_pedestrian.py (trajectory variant)

---

## Phase 6: User Story 4 (P2) - Visualize Single Pedestrian Elements

- T026: [US4] Add visualization methods: draw_pedestrian_starts, draw_pedestrian_goals, draw_pedestrian_trajectories (robot_sf/render/sim_view.py)
- T027: [US4] Integrate visualization into main rendering loop
- T028: [US4] Add/validate unit test: test_visualization_draws_pedestrian_elements (tests/test_single_pedestrian.py)
- T029: [US4] Add/validate example: example_multi_pedestrian.py (visualization demo)

---

## Phase 7: User Story 5 (P3) - Define Multiple Single Pedestrians

- T030: [US5] Extend simulator and map parsing to support up to 4 single pedestrians
- T031: [US5] Add/validate Social Force interactions between single and zone-spawned pedestrians
- T032: [US5] Add/validate unit test: test_multi_pedestrian_interaction (tests/test_single_pedestrian.py)
- T033: [US5] Add/validate example: example_multi_pedestrian.py (4 pedestrians)

---

## Phase 8: Polish & Cross-Cutting Concerns

- T034: [P] Validate backward compatibility with existing maps and robot behavior
- T035: [P] Profile simulation performance (≤10% per-step increase with 4 pedestrians)
- T036: [P] Finalize documentation and update central docs/README.md and docs/maps.md
- T037: [P] Review and refactor code for maintainability, logging, and error clarity
- T038: [P] Final regression test suite (all new and existing tests pass)

---

## Dependencies & Execution Order

- Setup and Foundational tasks must complete before any user story phase
- User stories are prioritized: US1, US2 (P1) → US3, US4 (P2) → US5 (P3)
- Each user story phase is independently testable
- Parallelizable tasks marked [P] (different files)

---

## Parallel Execution Examples

- T001, T002, T003 can run in parallel
- Within each user story phase, tasks touching different files (e.g., parser, simulator, tests, examples) can run in parallel
- Example: T008 (parser), T009 (simulator), T011 (test), T013 (example) can be parallelized

---

## Implementation Strategy

- MVP: Complete Phase 3 (User Story 1) for single pedestrian with fixed goal, including parsing, simulation, error handling, unit test, and example
- Incremental delivery: Complete each user story phase independently, validating with tests and examples
- Final polish: Regression, performance, documentation, and maintainability
