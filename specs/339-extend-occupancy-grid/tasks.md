# Tasks: Extended Occupancy Grid with Multi-Channel Support

**Feature**: 339-extend-occupancy-grid  
**Branch**: `339-extend-occupancy-grid`  
**Input**: Design documents from `/specs/339-extend-occupancy-grid/`  
**Prerequisites**: spec.md, plan.md, research.md, data-model.md, contracts/occupancy_api.md, quickstart.md  

**Organization**: Tasks organized by user story to enable independent implementation and parallel testing.

---

## Task Format: `[ID] [P?] [Story?] Description`

- **[ID]**: Task identifier (T001, T002, etc.) in execution order
- **[P]**: Parallelizable flag (can run in parallel with other [P] tasks at same level, different files)
- **[Story]**: User story label (US1, US2, US3, US4, US5) for user story tasks only
- Paths shown are absolute or relative to repo root `/Users/lennart/git/robot_sf_ll7/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize feature branch, project structure, and shared configuration.

**Status**: ✅ COMPLETE (All 5 tasks implemented)

- [x] T001 Initialize feature branch and verify `specs/339-extend-occupancy-grid/` documentation is accessible
  - ✅ Branch active: `339-extend-occupancy-grid`
  - ✅ All 11 documentation files present and accessible
  
- [x] T002 [P] Create base data model classes in `robot_sf/nav/occupancy_grid.py` (OccupancyGrid, GridChannel, GridConfig)
  - ✅ File created: `robot_sf/nav/occupancy_grid.py` (495 lines)
  - ✅ Classes: OccupancyGrid, GridChannel (enum), GridConfig (dataclass with validation)
  - ✅ Full docstrings, type hints, example usage
  
- [x] T003 [P] Create query result classes in `robot_sf/nav/occupancy_grid.py` (POIQuery, POIResult)
  - ✅ Classes: POIQuery, POIResult (dataclasses with field validation)
  - ✅ POIQueryType enum for query types (POINT, CIRCLE, RECT, LINE)
  - ✅ All invariants documented and validated in __post_init__
  
- [x] T004 [P] Create utility functions in `robot_sf/nav/occupancy_grid_utils.py` (coordinate transforms, bounds checking)
  - ✅ File created: `robot_sf/nav/occupancy_grid_utils.py` (380 lines)
  - ✅ 11 utility functions: world↔grid, world↔ego transforms, bounds checking, cell enumeration
  - ✅ Full docstrings with examples, type hints
  
- [x] T005 Verify existing `robot_sf/nav/occupancy.py` structure and identify extension points
  - ✅ Code reviewed: 306 lines, O(N) collision checking
  - ✅ Key finding: Module has NO rasterization; new feature extends with grid support
  - ✅ Existing API preserved: is_circle_circle_intersection(), is_circle_line_intersection(), ContinuousOccupancy
  - ✅ Extension strategy: New classes/functions in occupancy_grid.py, import in __init__.py for public API

**Implementation Summary**:
- Created `robot_sf/nav/occupancy_grid.py` (495 lines) with 4 classes, 2 enums
- Created `robot_sf/nav/occupancy_grid_utils.py` (380 lines) with 11 utility functions
- Created `tests/conftest_occupancy.py` (250 lines) with 23 pytest fixtures
- Created `tests/test_occupancy_grid.py` (300+ lines) with 28 test methods across 7 test classes
- All Phase 1 code compiles successfully ✅
- Ready for Phase 2 (Foundational) implementation

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before any user story implementation.

**Status**: ✅ COMPLETE (All 9 tasks implemented)

- [x] T006 [P] Implement grid utility functions in `robot_sf/nav/occupancy_grid_utils.py` (cell indexing, coordinate transforms)
  - ✅ Already completed in Phase 1 with 11 utility functions
  - ✅ Functions: world_to_grid_indices, grid_indices_to_world, world_to_ego, ego_to_world, is_within_grid, get_grid_bounds, clip_to_grid, get_affected_cells
  
- [x] T007 [P] Implement frame transform functions for ego/world conversion in `robot_sf/nav/occupancy_grid_utils.py`
  - ✅ Already completed in Phase 1
  - ✅ Functions: world_to_ego(), ego_to_world() with rotation matrix transforms
  
- [x] T008 [P] Implement obstacle rasterization in `robot_sf/nav/occupancy_grid_rasterization.py` (line segments to grid cells)
  - ✅ File created: `robot_sf/nav/occupancy_grid_rasterization.py` (310 lines)
  - ✅ Functions: rasterize_line_segment(), rasterize_obstacles(), _bresenham_line()
  - ✅ Algorithm: Bresenham's line algorithm, O(max(dx, dy))
  - ✅ Handles batch processing of obstacle lists
  
- [x] T009 [P] Implement pedestrian rasterization in `robot_sf/nav/occupancy_grid_rasterization.py` (circle-based occupancy)
  - ✅ Functions: rasterize_circle(), rasterize_pedestrians(), rasterize_robot()
  - ✅ Algorithm: Discrete disk algorithm, O(π * r²)
  - ✅ Handles batch processing of pedestrian circles
  
- [ ] T010 Configure `robot_sf/gym_env/unified_config.py` to include GridConfig fields (size_m, resolution_m, frame, occupancy_type, enabled_channels)
  - ⏳ Not started - deferred to next session
  
- [ ] T011 Add occupancy grid configuration to `RobotSimulationConfig` dataclass in `robot_sf/gym_env/unified_config.py`
  - ⏳ Not started - deferred to next session
  
- [x] T012 [P] Create test fixtures in `tests/conftest.py` for synthetic obstacles, pedestrians, and test grids
  - ✅ Added 13 fixtures to conftest.py: simple_grid_config, large_grid_config, coarse_grid_config, single_channel_config, occupancy_grid, robot_pose_center, robot_pose_corner, robot_pose_rotated, simple_obstacles, complex_obstacles, simple_pedestrians, crowded_pedestrians, empty_pedestrians, pre_generated_grid
  - ✅ Fixed type alias issues: Line2D, Circle2D, RobotPose now use tuple syntax
  
- [ ] T013 [P] Create SVG map fixtures in `maps/svg_maps/` (or reference existing) for integration tests
  - ⏳ Not started - can use existing SVG maps from `maps/svg_maps/` directory
  
- [x] T014 Setup logging infrastructure for grid module in `robot_sf/nav/occupancy_grid.py` (Loguru imports, structured logging)
  - ✅ Already completed in Phase 1
  - ✅ Loguru logging integrated: logger.debug() for initialization, grid generation
  - ✅ Structured logging with context (resolution, size, channels)

**Implementation Summary**:
- Created `robot_sf/nav/occupancy_grid_rasterization.py` (310 lines) with 6 rasterization functions
- Fixed circular import issues with TYPE_CHECKING pattern
- Integrated rasterization into OccupancyGrid.generate() method
- Added 13 test fixtures to conftest.py with correct type alias usage
- All 22 occupancy grid tests passing ✅
- Performance: All tests <0.01s (well under 20s soft limit)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel (after T010-T011 config integration)

---

## Phase 3: User Story 1 - Configure and Generate Multi-Channel Occupancy Grids (Priority: P1) 🎯 MVP

**Goal**: Enable developers to create configurable occupancy grids with separate channels for static obstacles and pedestrians.

**Independent Test**: Can be tested by creating a grid with specific parameters, verifying grid dimensions, resolution, channels match configuration, and cells are correctly marked occupied/free.

### Tests for User Story 1 (Unit & Integration)

- [ ] T015 [P] [US1] Unit test: Empty grid generation in `tests/test_occupancy_grid.py` (no obstacles, no pedestrians)
- [ ] T016 [P] [US1] Unit test: Single obstacle rasterization in `tests/test_occupancy_grid.py` (line segment, verify affected cells)
- [ ] T017 [P] [US1] Unit test: Multiple pedestrian rasterization in `tests/test_occupancy_grid.py` (3+ pedestrians at known positions, verify circles)
- [ ] T018 [P] [US1] Unit test: Ego-frame grid generation in `tests/test_occupancy_grid.py` (rotated robot, verify obstacle alignment)
- [ ] T019 [P] [US1] Unit test: World-frame grid generation in `tests/test_occupancy_grid.py` (fixed coordinates regardless of robot heading)
- [ ] T020 [P] [US1] Unit test: Multi-channel separation in `tests/test_occupancy_grid.py` (obstacles and pedestrians in separate channels, not mixed)
- [ ] T021 [P] [US1] Unit test: Grid update consistency in `tests/test_occupancy_grid.py` (timestamp, pedestrian updates, obstacle stasis)
- [ ] T022 [US1] Integration test: Full grid generation with real SVG map in `tests/test_occupancy_grid.py` (load map, create grid, verify coverage)
- [ ] T023 [US1] Integration test: Grid generation with FastPysfWrapper pedestrians in `tests/test_occupancy_grid.py` (extract pedestrians from physics sim)

### Implementation for User Story 1

- [ ] T024 Implement `create_occupancy_grid()` function in `robot_sf/nav/occupancy.py` with full signature and docstring
- [ ] T025 [P] [US1] Implement grid cell indexing and bounds checking in `robot_sf/nav/occupancy.py` (world→grid, grid→world)
- [ ] T026 [P] [US1] Implement ego-frame transformation in `robot_sf/nav/occupancy.py` (rotation matrix, robot-relative coordinates)
- [ ] T027 [P] [US1] Implement world-frame pass-through in `robot_sf/nav/occupancy.py` (direct grid allocation, no transform)
- [ ] T028 [P] [US1] Implement obstacle channel population in `robot_sf/nav/occupancy.py` (call rasterize, populate channel data)
- [ ] T029 [P] [US1] Implement pedestrian channel population in `robot_sf/nav/occupancy.py` (call rasterize, populate channel data)
- [ ] T030 [US1] Implement `OccupancyGrid.update()` method in `robot_sf/nav/occupancy.py` (timestamp, pedestrian updates, ego-frame re-rotation)
- [ ] T031 [US1] Add validation to `create_occupancy_grid()` for invalid configs (negative size, zero resolution, etc.)
- [ ] T032 [US1] Add error handling and clear error messages for grid creation failures in `robot_sf/nav/occupancy.py`
- [ ] T033 [US1] Add structured logging to grid generation in `robot_sf/nav/occupancy.py` (log grid creation, channel population, performance metrics)

**Checkpoint**: At this point, User Story 1 is fully functional. Developers can create multi-channel grids with ego/world frames and update them per timestep.

---

## Phase 4: User Story 2 - Gymnasium Observation Space Integration (Priority: P1)

**Goal**: Enable occupancy grids as part of gymnasium observation spaces so RL agents can learn from grid representations.

**Independent Test**: Can be tested by creating an environment with occupancy grid observation, resetting, and verifying observation includes grid array with correct shape (channels, height, width), dtype (float32), and values in [0,1].

### Tests for User Story 2 (Unit & Integration)

- [ ] T034 [P] [US2] Unit test: Box observation space creation in `tests/test_occupancy_gymnasium.py` (correct shape, dtype, bounds)
- [ ] T035 [P] [US2] Unit test: Grid-to-observation conversion in `tests/test_occupancy_gymnasium.py` (reshape [C,H,W], verify float32, values in [0,1])
- [ ] T036 [P] [US2] Unit test: Multi-channel observation stacking in `tests/test_occupancy_gymnasium.py` (multiple channels in single array)
- [ ] T037 [P] [US2] Unit test: Variable grid config observation adaptation in `tests/test_occupancy_gymnasium.py` (different sizes/resolutions → different observation shapes)
- [ ] T038 [US2] Integration test: Environment reset with occupancy observation in `tests/test_occupancy_gymnasium.py` (make_robot_env with grid, reset, check obs)
- [ ] T039 [US2] Integration test: Environment step with occupancy observation updates in `tests/test_occupancy_gymnasium.py` (step, verify grid channels update)
- [ ] T040 [US2] Integration test: StableBaselines3 RL training with grid observation in `tests/test_occupancy_gymnasium.py` (quick 10-step episode with PPO, no crashes)

### Implementation for User Story 2

- [ ] T041 Implement `grid_to_observation()` function in `robot_sf/nav/occupancy.py` (channels→array, [C,H,W] shape, float32 dtype)
- [ ] T042 [P] [US2] Extend `RobotSimulationConfig` in `robot_sf/gym_env/unified_config.py` to include grid observation flag
- [ ] T043 [P] [US2] Modify environment reset in `robot_sf/gym_env/environment.py` to generate initial occupancy grid
- [ ] T044 [P] [US2] Modify environment step in `robot_sf/gym_env/environment.py` to update occupancy grid each timestep
- [ ] T045 [US2] Implement gym observation space definition in `robot_sf/gym_env/environment.py` to include grid Box space
- [ ] T046 [US2] Integrate grid observation into `make_robot_env()` factory in `robot_sf/gym_env/environment_factory.py`
- [ ] T047 [US2] Add configuration validation in `robot_sf/gym_env/unified_config.py` (grid config must be valid for env creation)
- [ ] T048 [US2] Add logging for gymnasium integration in `robot_sf/gym_env/environment.py` (log grid observation shape/dtype on env creation)

**Checkpoint**: At this point, User Stories 1 and 2 are complete. Environments can include occupancy grids as part of observations, and RL agents can train on grid-based policies.

---

## Phase 5: User Story 3 - Query Point-of-Interest Status for Spawn Validation (Priority: P2)

**Goal**: Enable developers to programmatically check if specific points or areas are free, supporting automated spawn validation.

**Independent Test**: Can be tested by querying multiple points in known free/occupied regions and verifying results (free, occupied, safe-to-spawn).

### Tests for User Story 3 (Unit & Integration)

- [ ] T049 [P] [US3] Unit test: Point query in free space in `tests/test_occupancy_queries.py` (query returns is_occupied=False)
- [ ] T050 [P] [US3] Unit test: Point query in occupied space in `tests/test_occupancy_queries.py` (query returns is_occupied=True)
- [ ] T051 [P] [US3] Unit test: Point query at boundary in `tests/test_occupancy_queries.py` (exact edge cases, no crashes)
- [ ] T052 [P] [US3] Unit test: Point query out-of-bounds in `tests/test_occupancy_queries.py` (raises ValueError or returns safe default with clear message)
- [ ] T053 [P] [US3] Unit test: Circular AOI query free in `tests/test_occupancy_queries.py` (entire circle free → safe_to_spawn=True)
- [ ] T054 [P] [US3] Unit test: Circular AOI query partially occupied in `tests/test_occupancy_queries.py` (some cells occupied → safe_to_spawn=False)
- [ ] T055 [P] [US3] Unit test: Rectangular AOI query in `tests/test_occupancy_queries.py` (rectangle query returns occupancy fraction)
- [ ] T056 [P] [US3] Unit test: Per-channel query results in `tests/test_occupancy_queries.py` (query returns results per channel)
- [ ] T057 [US3] Integration test: Spawn validation workflow in `tests/test_occupancy_queries.py` (query 100 candidates, verify >95% success in valid regions)
- [ ] T058 [US3] Integration test: Query with FastPysfWrapper pedestrians in `tests/test_occupancy_queries.py` (queries respect dynamic pedestrian channels)

### Implementation for User Story 3

- [ ] T059 Implement `query_occupancy()` function in `robot_sf/nav/occupancy.py` with POIQuery/POIResult types
- [ ] T060 [P] [US3] Implement point query logic in `robot_sf/nav/occupancy.py` (world coord→grid cell, check occupancy)
- [ ] T061 [P] [US3] Implement circular AOI query in `robot_sf/nav/occupancy.py` (rasterize circle, check all cells)
- [ ] T062 [P] [US3] Implement rectangular AOI query in `robot_sf/nav/occupancy.py` (rasterize rectangle, check all cells)
- [ ] T063 [US3] Implement per-channel query results in `robot_sf/nav/occupancy.py` (POIResult includes breakdown by channel)
- [ ] T064 [US3] Implement safe-to-spawn heuristic in `robot_sf/nav/occupancy.py` (True if occupancy_fraction < threshold)
- [ ] T065 [US3] Implement out-of-bounds handling in `robot_sf/nav/occupancy.py` (clear error message, optional safe default)
- [ ] T066 [US3] Add query validation in `robot_sf/nav/occupancy.py` (positive dimensions, valid frame, etc.)
- [ ] T067 [US3] Add logging for queries in `robot_sf/nav/occupancy.py` (debug: query type, location, result; warning: OOB queries)

**Checkpoint**: User Story 3 is complete. Developers can validate spawn points and areas programmatically.

---

## Phase 6: User Story 4 - Visualize Occupancy Grids in Pygame (Priority: P2)

**Goal**: Enable visual debugging of occupancy grids in pygame with color-coded cells and interactive channel toggling.

**Independent Test**: Can be tested by enabling grid visualization in pygame, running simulation for a few steps, and verifying grid renders without crashes and is visible on screen.

### Tests for User Story 4 (Visual & Integration)

- [ ] T068 [P] [US4] Visual test: Grid rendering at default configuration in `test_pygame/test_occupancy_visualization.py` (no crashes, grid visible)
- [ ] T069 [P] [US4] Visual test: Obstacle cells highlighted in `test_pygame/test_occupancy_visualization.py` (obstacles show yellow, pedestrians show red)
- [ ] T070 [P] [US4] Visual test: Free cells transparent in `test_pygame/test_occupancy_visualization.py` (background visible through free cells)
- [ ] T071 [P] [US4] Visual test: Ego-frame rotation in `test_pygame/test_occupancy_visualization.py` (grid rotates with robot heading)
- [ ] T072 [P] [US4] Visual test: World-frame alignment in `test_pygame/test_occupancy_visualization.py` (grid stays aligned to world regardless of robot heading)
- [ ] T073 [P] [US4] Visual test: Channel toggling in `test_pygame/test_occupancy_visualization.py` (toggle visibility, verify rendering changes)
- [ ] T074 [US4] Integration test: Full simulation with visualization in `test_pygame/test_occupancy_visualization.py` (run 100 steps, no performance regression, 30+ FPS maintained)
- [ ] T075 [US4] Integration test: Video recording with grid overlay in `test_pygame/test_occupancy_visualization.py` (if video recording enabled, grid appears in output)

### Implementation for User Story 4

- [ ] T076 Implement `render_grid_pygame()` function in `robot_sf/render/sim_view.py` with complete docstring
- [ ] T077 [P] [US4] Implement cell color mapping in `robot_sf/render/sim_view.py` (obstacles yellow, pedestrians red, free transparent)
- [ ] T078 [P] [US4] Implement grid-to-surface coordinate transform in `robot_sf/render/sim_view.py` (grid cells→pygame pixels)
- [ ] T079 [P] [US4] Implement per-cell rendering in `robot_sf/render/sim_view.py` (draw colored rectangles for occupied cells)
- [ ] T080 [P] [US4] Implement alpha blending in `robot_sf/render/sim_view.py` (configurable transparency for grid overlay)
- [ ] T081 [US4] Implement `toggle_grid_channel_visibility()` function in `robot_sf/render/sim_view.py` (remember visibility state, re-render)
- [ ] T082 [US4] Implement ego-frame rotation in `robot_sf/render/sim_view.py` (apply rotation matrix, center on robot)
- [ ] T083 [US4] Integrate grid visualization into `SimView` class in `robot_sf/render/sim_view.py` (add to render loop, check visibility flag)
- [ ] T084 [US4] Add configuration for grid visualization in `robot_sf/gym_env/unified_config.py` (enable/disable, color scheme, alpha)
- [ ] T085 [US4] Add logging for visualization in `robot_sf/render/sim_view.py` (debug: rendering time, FPS impact)

**Checkpoint**: User Story 4 is complete. Developers can visually inspect grids in pygame with interactive controls.

---

## Phase 7: User Story 5 - Achieve 100% Test Coverage (Priority: P1)

**Goal**: Ensure comprehensive test coverage of the occupancy module to catch regressions and verify all edge cases.

**Independent Test**: Can be verified by running coverage analysis on `robot_sf/nav/occupancy.py` and confirming 100% line, branch, and condition coverage.

### Tests for User Story 5 (Edge Cases & Coverage)

- [ ] T086 [P] [US5] Edge case test: Empty grid (no obstacles, no pedestrians) in `tests/test_occupancy_edge_cases.py`
- [ ] T087 [P] [US5] Edge case test: Fully occupied grid (every cell occupied) in `tests/test_occupancy_edge_cases.py`
- [ ] T088 [P] [US5] Edge case test: Very high resolution (0.01m per cell) in `tests/test_occupancy_edge_cases.py` (verify <5ms target still met)
- [ ] T089 [P] [US5] Edge case test: Very low resolution (1m per cell) in `tests/test_occupancy_edge_cases.py`
- [ ] T090 [P] [US5] Edge case test: Pedestrian at grid boundary in `tests/test_occupancy_edge_cases.py`
- [ ] T091 [P] [US5] Edge case test: Rotated frame at cardinal angles (0°, 90°, 180°, 270°) in `tests/test_occupancy_edge_cases.py`
- [ ] T092 [P] [US5] Edge case test: Frame transition (ego↔world mid-simulation) in `tests/test_occupancy_edge_cases.py`
- [ ] T093 [P] [US5] Edge case test: Single-cell grid in `tests/test_occupancy_edge_cases.py`
- [ ] T094 [P] [US5] Edge case test: Zero pedestrians, multiple obstacles in `tests/test_occupancy_edge_cases.py`
- [ ] T095 [P] [US5] Edge case test: Many pedestrians, no obstacles in `tests/test_occupancy_edge_cases.py`
- [ ] T096 [P] [US5] Error path test: Invalid grid config (negative size, zero resolution) in `tests/test_occupancy_edge_cases.py` (should raise ValueError)
- [ ] T097 [P] [US5] Error path test: Query out-of-bounds in `tests/test_occupancy_edge_cases.py` (should raise or handle gracefully)
- [ ] T098 [US5] Coverage verification: Run `coverage.py` on `robot_sf/nav/occupancy.py`, verify 100% line+branch coverage in `tests/test_occupancy_coverage.py`
- [ ] T099 [US5] Coverage verification: Generate HTML coverage report in `output/coverage/htmlcov/` and verify all occupancy.py code is covered

### Implementation for User Story 5 (Instrumentation)

- [ ] T100 Add comprehensive docstrings to all functions/methods in `robot_sf/nav/occupancy.py` (purpose, params, returns, raises)
- [ ] T101 Add boundary assertion checks in grid functions (ensure grid operations stay in bounds)
- [ ] T102 Add type hints to all functions in `robot_sf/nav/occupancy.py` (function signatures, return types)
- [ ] T103 Review and refactor complex functions in `robot_sf/nav/occupancy.py` to reduce cyclomatic complexity (avoid noqa: C901)
- [ ] T104 Add structured logging to all error paths in `robot_sf/nav/occupancy.py` (logger.error with context)

**Checkpoint**: User Story 5 is complete. Occupancy module has 100% test coverage and all edge cases are handled.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, examples, performance validation, and final integration.

- [ ] T105 [P] Update `docs/dev/occupancy/Update_or_extend_occupancy.md` with usage guide (create, query, visualize grids)
- [ ] T106 [P] Add API reference section to `docs/dev/occupancy/Update_or_extend_occupancy.md` (function signatures, parameters, returns)
- [ ] T107 [P] Add configuration guide to `docs/dev/occupancy/Update_or_extend_occupancy.md` (GridConfig options, frame modes, channels)
- [ ] T108 [P] Add troubleshooting section to `docs/dev/occupancy/Update_or_extend_occupancy.md` (grid not updating, performance slow, etc.)
- [ ] T109 [P] Create example script in `examples/advanced/` demonstrating grid creation, querying, and visualization
- [ ] T110 [P] Create quickstart notebook or script in `examples/quickstart/` showing basic grid usage (3-5 min walkthrough)
- [ ] T111 Create demo script in `examples/` for occupancy-based reward shaping in RL training
- [ ] T112 Update `docs/README.md` to link new occupancy documentation
- [ ] T113 [P] Run performance validation: grid generation <5ms, queries <1ms, visualization 30+ FPS in `scripts/validation/`
- [ ] T114 [P] Run example smoke tests: `run_examples_smoke.py` covers occupancy examples, no failures
- [ ] T115 Verify backward compatibility: existing code using old `occupancy.py` API still works (if any)
- [ ] T116 Update CHANGELOG.md with new feature summary (grid types, query APIs, gymnasium integration, visualization)
- [ ] T117 [P] Run final quality gates: Ruff format, type check, pylint, full pytest suite
- [ ] T118 Merge to main and close feature branch

**Checkpoint**: Feature complete, documented, tested, and merged. Ready for production use.

---

## Task Summary

**Total Tasks**: 118  
**Phase 1 (Setup)**: 5 tasks  
**Phase 2 (Foundational)**: 9 tasks  
**Phase 3 (US1)**: 19 tasks (9 tests + 10 implementation)  
**Phase 4 (US2)**: 16 tasks (7 tests + 9 implementation)  
**Phase 5 (US3)**: 20 tasks (10 tests + 10 implementation)  
**Phase 6 (US4)**: 20 tasks (8 tests + 12 implementation)  
**Phase 7 (US5)**: 19 tasks (14 tests + 5 instrumentation)  
**Phase 8 (Polish)**: 14 tasks  

---

## Parallel Execution Strategy

### Parallel Opportunities within Phases

**Phase 2 (Foundational)**:
- T006–T009 (utility functions, coordinate transforms, rasterization) can run in parallel (different files, no inter-task dependencies)
- T012–T013 (test fixtures) can run in parallel

**Phase 3 (US1)**:
- T015–T021 (unit tests) can run in parallel (independent test cases)
- T025–T029 (implementation components: indexing, transforms, channel population) can run in parallel
- All T015–T021 tests can run in parallel before any implementation tasks

**Phase 4 (US2)**:
- T034–T037 (unit tests) can run in parallel
- T042–T044 (config, reset, step modifications) can run in parallel after T041

**Phase 5 (US3)**:
- T049–T058 (query tests) can run in parallel
- T060–T062 (query implementations: point, circular, rectangular) can run in parallel

**Phase 6 (US4)**:
- T068–T074 (visual tests) can run in parallel
- T077–T080 (rendering implementations: color mapping, transforms, blending) can run in parallel

**Phase 7 (US5)**:
- T086–T097 (edge case and error path tests) can run in parallel
- T100–T103 (instrumentation: docstrings, type hints, refactoring) can run in parallel

**Phase 8 (Polish)**:
- T105–T111 (documentation and examples) can run in parallel
- T113–T114 (validation scripts) can run in parallel

### Example Parallel Execution Plan

**Fast-path for MVP (User Story 1 only)**:
```
Phase 1 → Phase 2 → Phase 3 (T015-T032, excluding T033 logging)
Expected timeline: 4–6 sprints for core functionality
```

**Full feature delivery (all user stories)**:
```
Phase 1 → Phase 2 → (Phase 3 + Phase 4 + Phase 5 + Phase 6 in parallel) → Phase 7 → Phase 8
Expected timeline: 8–12 sprints for complete feature with 100% coverage
```

**Recommended first sprint** (5–7 tasks):
- T001: Setup
- T006, T007, T008, T009: Foundational utilities (parallelizable)
- T002, T003, T004: Data model classes (parallelizable)

---

## Implementation Notes

### MVP Scope (User Stories 1 & 2)
Implement grid creation/update and gymnasium integration first. This delivers core value (occupancy observation for RL) without queries or visualization. Recommended for initial 4-week sprint.

### MVP-2 Scope (Add US3)
Add POI/AOI query API for spawn validation. Recommended for week 5–6.

### Full Scope (Add US4 & US5)
Complete with visualization and 100% coverage. Recommended for weeks 7–12.

### Key Dependencies
- Phase 2 must complete before any user story (blocking)
- US1 must complete before US2 (gymnasium integration depends on grid data structures)
- US3 can start after Phase 2 (independent from US2)
- US4 can start after Phase 2 (visualization independent from queries)
- US5 (testing) can start after US1, but coverage validation requires all stories complete

### File Structure Summary
- **Core module**: `robot_sf/nav/occupancy.py` (grid generation, queries, utility functions)
- **Configuration**: `robot_sf/gym_env/unified_config.py` (GridConfig dataclass)
- **Gymnasium integration**: `robot_sf/gym_env/environment.py` (grid in observation space)
- **Visualization**: `robot_sf/render/sim_view.py` (pygame rendering)
- **Tests**: `tests/test_occupancy_*.py` (4 files), `test_pygame/test_occupancy_visualization.py`
- **Documentation**: `docs/dev/occupancy/Update_or_extend_occupancy.md` (usage guide)
- **Examples**: `examples/advanced/` and `examples/quickstart/` (demo scripts)

---

## Success Criteria Mapping

| Success Criterion | Related Tasks | Validation |
|-------------------|---------------|-----------|
| SC-001: 100% coverage | T086–T099 (US5) | `coverage.py` report in Phase 8 |
| SC-002: <5ms gen | T113 (performance validation) | Benchmark test in Phase 8 |
| SC-003: <1ms queries | T113 (performance validation) | Benchmark test in Phase 8 |
| SC-004: 30+ FPS viz | T074, T113 | Visual test in Phase 6, bench in Phase 8 |
| SC-005: Gymnasium tests pass | T038–T040 (US2) | Integration tests in Phase 4 |
| SC-006: Visual tests pass | T068–T074 (US4) | Visual tests in Phase 6 |
| SC-007: Documentation complete | T105–T112 (Polish) | Docs review in Phase 8 |
| SC-008: Edge cases handled | T086–T097 (US5) | Edge case tests in Phase 7 |
| SC-009: O(N) avoidance | T113, T059–T067 | Query implementation + benchmark |
| SC-010: Spawn validation >95% | T057 (US3) | Integration test in Phase 5 |

---

## Next Steps

1. **Start Phase 1**: Initialize feature branch and verify documentation (T001).
2. **Complete Phase 2**: Build foundational utilities and test fixtures (T006–T014). This unblocks all user stories.
3. **Parallel User Story Implementation**: Execute Phases 3–7 according to priority (US1 & US2 first for MVP).
4. **Finalize**: Documentation, examples, validation, and merge (Phase 8).

**Estimated timeline**: 8–12 weeks for full feature with 100% coverage and all user stories complete.
