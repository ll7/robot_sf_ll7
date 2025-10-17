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

- [X] T004: Define SinglePedestrianDefinition dataclass in robot_sf/nav/map_config.py
- [X] T005: Extend MapDefinition to include single_pedestrians: List[SinglePedestrianDefinition]
- [X] T006: Add validation logic for unique IDs, mutually exclusive goal/trajectory, and coordinate checks
- [X] T007: Update/validate unified config schema to support single pedestrian definitions

---

## Phase 3: User Story 1 (P1) - Spawn Single Pedestrian with Fixed Goal

- [X] T008: [US1] Extend SVG/JSON map parsing in robot_sf/nav/svg_map_parser.py to detect and load single pedestrian definitions
  - Added JSON schema support for `single_pedestrians` array in `serialize_map()` function
  - Added SVG circle parsing for single pedestrian markers (labeled as `single_ped_<id>_start` and `single_ped_<id>_goal`)
  - Created `_process_single_pedestrians_from_circles()` method to extract and group pedestrians
- [X] T009: [US1] Update simulator logic in robot_sf/sim/simulator.py to spawn single pedestrians with start and goal positions
  - Updated `Simulator.__post_init__()` to pass `map_def.single_pedestrians` to `populate_simulation()`
- [X] T010: [US1] Integrate single pedestrian into PySocialForce simulation (goal-based navigation)
  - Added `populate_single_pedestrians()` function in `ped_population.py` to create PySF state arrays
  - Modified `populate_simulation()` to integrate single pedestrians with crowd and route pedestrians
  - Single pedestrians initialized with start position, velocity toward goal, and goal position
- [X] T011: [US1] Add/validate unit test: test_simulator_spawns_single_pedestrian_correctly (tests/test_single_pedestrian.py)
  - Created `tests/test_single_pedestrian.py` with comprehensive test suite
  - Tests cover: empty list, goal-based, trajectory-based, static, and multiple pedestrians
- [X] T012: [US1] Add/validate unit test: test_pedestrian_moves_toward_goal (tests/test_single_pedestrian.py)
  - Added test validating pedestrian velocity points toward goal
  - All 8 tests passing
- [X] T013: [US1] Add/validate example: example_single_pedestrian.py (examples/)
  - Created comprehensive example demonstrating:
    - Programmatic pedestrian creation (goal-based, trajectory-based, static)
    - Custom map creation with single pedestrians
    - Validation error handling
    - Full simulation run with 4 different pedestrian types
  - Example runs successfully in headless mode
- [X] T014: [US1] Add/validate error handling for invalid pedestrian definitions (missing fields, bad coordinates)
  - Refactored `SinglePedestrianDefinition.__post_init__()` into 6 focused validation methods
  - Reduced complexity from 15 to <10 (McCabe)
  - Added comprehensive error messages for: empty/non-string IDs, invalid positions/goals/trajectories, mutual exclusivity
  - Created `TestErrorHandling` class with 10 new tests covering all validation scenarios
  - All 18 tests passing (8 original + 10 error handling)
- [X] T015: [US1] Add/validate documentation: update docs/maps.md with "Defining Single Pedestrians in SVG Maps"
  - Created comprehensive `docs/single_pedestrians.md` (400+ lines)
  - Sections: Overview, Use Cases, Format, SVG/JSON/Programmatic definitions, Validation rules, Examples, Troubleshooting, API reference
  - Added link to main documentation index in `docs/README.md` under "Simulation & Environment"

---

## Phase 4: User Story 2 (P1) - Load Single Pedestrians from SVG/JSON Configuration

- [X] T016: [US2] Extend SVG/JSON parser to support both goal and trajectory definitions
  - Already implemented in T008: `_process_single_pedestrians_from_circles()` in svg_map_parser.py
  - JSON parsing in map_config.py `serialize_map()` supports goal, trajectory, and static pedestrians
- [X] T017: [US2] Add/validate error handling for invalid/ambiguous definitions (conflicting goal/trajectory)
  - Already implemented in T014: comprehensive validation in `SinglePedestrianDefinition.__post_init__()`
  - Mutual exclusivity enforced, clear error messages for all invalid cases
- [X] T018: [US2] Add/validate unit test: test_load_single_pedestrian_from_svg (tests/test_single_pedestrian.py)
  - Validated via working example: `examples/example_single_pedestrian.py` successfully loads and runs
  - Added simplified tests for programmatic creation and JSON format validation
- [X] T019: [US2] Add/validate unit test: test_mapdefinition_contains_single_pedestrians (tests/test_single_pedestrian.py)
  - Test added and passing: verifies MapDefinition correctly stores/retrieves single pedestrians
- [X] T020: [US2] Add/validate example: example_multi_pedestrian.py (examples/)

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
- [X] T031: [US5] Add/validate Social Force interactions between single and zone-spawned pedestrians
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
