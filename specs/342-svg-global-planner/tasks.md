# Tasks: SVG-Based Global Planner

**Input**: Design documents from `/specs/342-svg-global-planner/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/global_planner_api.md

**Tests**: Tests are included as this is a core library feature requiring high coverage (90%+ target per SC-007)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Project follows robot_sf structure:
- Library code: `robot_sf/planner/` (new module)
- Extensions: `robot_sf/nav/`, `robot_sf/gym_env/`
- Tests: `tests/test_planner/`
- Examples: `examples/advanced/`
- Validation: `scripts/validation/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic module structure

- [ ] T001 Create robot_sf/planner/ module directory with __init__.py
- [ ] T002 Add pyvisgraph>=0.2.1 and networkx>=3.6 to pyproject.toml dependencies
- [ ] T003 [P] Create tests/test_planner/ directory structure with __init__.py
- [ ] T004 [P] Create tests/fixtures/test_maps/ directory for test SVG files
- [ ] T005 [P] Update docs/README.md to add planner documentation link placeholder

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data structures that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T006 Implement PlannerConfig dataclass with validation in robot_sf/planner/global_planner.py
- [ ] T007 Implement PlanningFailedError exception in robot_sf/planner/global_planner.py
- [ ] T008 [P] Create test_planner_config.py unit tests in tests/test_planner/
- [ ] T009 Extend MapDefinition with poi_positions and poi_labels fields in robot_sf/nav/map_config.py
- [ ] T010 [P] Create simple_corridor.svg test map in tests/fixtures/test_maps/
- [ ] T011 [P] Create narrow_passage.svg test map in tests/fixtures/test_maps/
- [ ] T012 [P] Create no_path.svg test map in tests/fixtures/test_maps/
- [ ] T013 [P] Create complex_warehouse.svg test map in tests/fixtures/test_maps/

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Basic Path Generation (Priority: P1) 🎯 MVP

**Goal**: Generate collision-free paths between spawn and goal zones using visibility graphs

**Independent Test**: Load a map with obstacles, request path from spawn to goal, verify waypoints avoid obstacles and connect start to goal

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T014 [P] [US1] Create test_global_planner.py with basic path generation tests in tests/test_planner/
- [ ] T015 [P] [US1] Add test for clearance verification (SC-002) in tests/test_planner/test_global_planner.py
- [ ] T016 [P] [US1] Add test for straight-line path on empty map (edge case) in tests/test_planner/test_global_planner.py
- [ ] T017 [P] [US1] Add test for PlanningFailedError when no path exists in tests/test_planner/test_global_planner.py

### Implementation for User Story 1

- [ ] T018 [US1] Implement _inflate_obstacles() method using Shapely buffering in robot_sf/planner/global_planner.py
- [ ] T019 [US1] Create visibility_graph.py module with pyvisgraph wrapper in robot_sf/planner/
- [ ] T020 [US1] Implement VisibilityGraph.build() using pyvisgraph in robot_sf/planner/visibility_graph.py
- [ ] T021 [US1] Implement VisibilityGraph.shortest_path() with Dijkstra via NetworkX in robot_sf/planner/visibility_graph.py
- [ ] T022 [US1] Implement GlobalPlanner.__init__() with lazy graph initialization in robot_sf/planner/global_planner.py
- [ ] T023 [US1] Implement GlobalPlanner.plan() basic method (no caching, no smoothing) in robot_sf/planner/global_planner.py
- [ ] T024 [US1] Add boundary validation for start/goal positions in robot_sf/planner/global_planner.py
- [ ] T025 [US1] Handle start/goal inside obstacles with projection to free space in robot_sf/planner/global_planner.py
- [ ] T026 [US1] Add fallback_on_failure logic for straight-line paths in robot_sf/planner/global_planner.py
- [ ] T027 [US1] Add Loguru logging for planning operations (graph build, failures, warnings) in robot_sf/planner/global_planner.py
- [ ] T028 [US1] Run unit tests and verify all US1 tests pass

**Checkpoint**: Basic path generation working - planner generates collision-free paths for simple maps

---

## Phase 4: User Story 2 - Map Integration and Compatibility (Priority: P2)

**Goal**: Ensure planner works seamlessly with existing SVG maps and RouteNavigator

**Independent Test**: Use planner output with RouteNavigator, verify it tracks waypoints correctly

### Tests for User Story 2

- [ ] T029 [P] [US2] Create test_map_integration.py for SVG parser integration in tests/test_planner/
- [ ] T030 [P] [US2] Add test verifying planner output works with RouteNavigator in tests/test_planner/test_map_integration.py
- [ ] T031 [P] [US2] Add tests for all existing example maps generating valid paths (SC-003) in tests/test_planner/test_map_integration.py
- [ ] T032 [P] [US2] Add test for POI parsing from SVG in tests/test_planner/test_map_integration.py

### Implementation for User Story 2

- [ ] T033 [US2] Extend svg_map_parser.py to parse <circle class="poi"> elements in robot_sf/nav/svg_map_parser.py
- [ ] T034 [US2] Add POI extraction to MapDefinition.poi_positions during parsing in robot_sf/nav/svg_map_parser.py
- [ ] T035 [US2] Add POI label mapping to MapDefinition.poi_labels in robot_sf/nav/svg_map_parser.py
- [ ] T036 [US2] Implement MapDefinition.get_poi_by_label() helper method in robot_sf/nav/map_config.py
- [ ] T037 [US2] Add backward compatibility handling for maps without POIs in robot_sf/nav/svg_map_parser.py
- [ ] T038 [US2] Update existing example maps with POI annotations in maps/svg_maps/ (select 2-3 maps)
- [ ] T039 [US2] Verify RouteNavigator compatibility with planner-generated paths (manual validation)
- [ ] T040 [US2] Run integration tests and verify all US2 tests pass

**Checkpoint**: Planner fully integrated with existing map system and navigation components

---

## Phase 5: User Story 3 - Flexible Start/Goal Selection (Priority: P3)

**Goal**: Support various start/goal combinations including zone sampling and POI routing

**Independent Test**: Sample multiple start/goal pairs from zones, verify planner generates valid paths for each

### Tests for User Story 3

- [ ] T041 [P] [US3] Create test_poi_sampler.py for POI sampling strategies in tests/test_planner/
- [ ] T042 [P] [US3] Add test for random POI sampling with seed reproducibility in tests/test_planner/test_poi_sampler.py
- [ ] T043 [P] [US3] Add test for nearest/farthest POI selection strategies in tests/test_planner/test_poi_sampler.py
- [ ] T044 [P] [US3] Add test for via_pois routing in GlobalPlanner.plan() in tests/test_planner/test_global_planner.py
- [ ] T045 [P] [US3] Add test for dynamic start positions (non-zone) in tests/test_planner/test_global_planner.py

### Implementation for User Story 3

- [ ] T046 [P] [US3] Create poi_sampler.py module in robot_sf/planner/
- [ ] T047 [US3] Implement POISampler.__init__() with seeded RNG in robot_sf/planner/poi_sampler.py
- [ ] T048 [US3] Implement POISampler.sample() with random strategy in robot_sf/planner/poi_sampler.py
- [ ] T049 [US3] Add nearest and farthest strategies to POISampler.sample() in robot_sf/planner/poi_sampler.py
- [ ] T050 [US3] Implement via_pois parameter handling in GlobalPlanner.plan() in robot_sf/planner/global_planner.py
- [ ] T051 [US3] Add multi-segment planning for via_pois (plan each segment separately) in robot_sf/planner/global_planner.py
- [ ] T052 [US3] Implement GlobalPlanner.plan_multi_goal() with nearest-neighbor TSP heuristic in robot_sf/planner/global_planner.py
- [ ] T053 [US3] Add validation for via_pois IDs against map POIs in robot_sf/planner/global_planner.py
- [ ] T054 [US3] Run tests and verify all US3 tests pass

**Checkpoint**: Planner supports flexible routing with POIs and multiple goals

---

## Phase 6: User Story 4 - Performance and Caching (Priority: P4)

**Goal**: Achieve <100ms path queries through graph caching and optimization

**Independent Test**: Measure path generation time across 100 queries, verify median <100ms and graph built only once

### Tests for User Story 4

- [ ] T055 [P] [US4] Create test_visibility_graph.py for graph caching tests in tests/test_planner/
- [ ] T056 [P] [US4] Add test verifying graph caching behavior (single build, multiple queries) in tests/test_planner/test_visibility_graph.py
- [ ] T057 [P] [US4] Add test for cache invalidation in tests/test_planner/test_visibility_graph.py
- [ ] T058 [P] [US4] Add performance benchmark test (SC-001, SC-006) in tests/test_planner/test_global_planner.py

### Implementation for User Story 4

- [ ] T059 [US4] Implement _graph_cache dictionary with map hash keys in robot_sf/planner/visibility_graph.py
- [ ] T060 [US4] Implement _get_or_build_graph() helper with cache lookup in robot_sf/planner/visibility_graph.py
- [ ] T061 [US4] Add map content hashing (md5 of obstacles repr) in robot_sf/planner/visibility_graph.py
- [ ] T062 [US4] Update GlobalPlanner.plan() to use cached graphs in robot_sf/planner/global_planner.py
- [ ] T063 [US4] Implement GlobalPlanner.invalidate_cache() method in robot_sf/planner/global_planner.py
- [ ] T064 [US4] Create path_smoother.py module in robot_sf/planner/
- [ ] T065 [US4] Implement Douglas-Peucker simplification in robot_sf/planner/path_smoother.py
- [ ] T066 [US4] Integrate path smoothing into GlobalPlanner.plan() when enabled in robot_sf/planner/global_planner.py
- [ ] T067 [US4] Add smoothing validation (collision-free check) in robot_sf/planner/path_smoother.py
- [ ] T068 [US4] Run performance tests and verify SC-001 (<100ms median) and SC-006 (<500ms build)

**Checkpoint**: All performance targets met, caching working efficiently

---

## Phase 7: Environment Integration

**Purpose**: Integrate planner with gym environment factory and unified configs

- [ ] T069 Extend RobotSimulationConfig with use_planner flag in robot_sf/gym_env/unified_config.py
- [ ] T070 Add planner_clearance_margin config field in robot_sf/gym_env/unified_config.py
- [ ] T071 Update navigation.sample_route() to delegate to planner when enabled in robot_sf/nav/navigation.py
- [ ] T072 Add planner initialization in environment factory functions in robot_sf/gym_env/environment_factory.py
- [ ] T073 [P] Create test_navigation_integration.py for env factory integration in tests/test_planner/
- [ ] T074 [P] Add test for config-driven planner usage in make_robot_env() in tests/test_planner/test_navigation_integration.py
- [ ] T075 [P] Verify backward compatibility (planner disabled by default) in tests/test_planner/test_navigation_integration.py

**Checkpoint**: Planner fully integrated with gym environments via factory pattern

---

## Phase 8: Examples and Documentation

**Purpose**: Create runnable examples and user-facing documentation

- [ ] T076 [P] Create 20_global_planner_demo.py example in examples/advanced/
- [ ] T077 [P] Create 21_poi_routing_demo.py example in examples/advanced/
- [ ] T078 [P] Update examples manifest with new planner demos in examples/examples_manifest.yaml
- [ ] T079 [P] Add planner section to docs/README.md with links to contracts and quickstart
- [ ] T080 [P] Create verification script verify_planner.sh in scripts/validation/
- [ ] T081 [P] Create benchmark_planner.py performance profiling script in scripts/
- [ ] T082 Verify quickstart.md examples are runnable and accurate
- [ ] T083 Update CHANGELOG.md with planner feature addition

**Checkpoint**: Documentation and examples complete, ready for user onboarding

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Final quality improvements and validation

- [ ] T084 [P] Add comprehensive docstrings to all public methods (Google style) in robot_sf/planner/
- [ ] T085 [P] Add type hints to all function signatures in robot_sf/planner/
- [ ] T086 Run Ruff formatter and linter on robot_sf/planner/ and tests/test_planner/
- [ ] T087 Run ty type checker and resolve any type errors in robot_sf/planner/
- [ ] T088 Verify test coverage exceeds 90% (SC-007) for robot_sf/planner/ module
- [ ] T089 [P] Add narrow passage warning detection in robot_sf/planner/global_planner.py
- [ ] T090 [P] Add path quality metrics (length, clearance) to plan() return metadata (optional enhancement)
- [ ] T091 Run all validation scripts (verify_planner.sh, benchmark_planner.py)
- [ ] T092 Verify all success criteria (SC-001 through SC-010) are met
- [ ] T093 Final smoke test: Run all examples and verify no errors

**Checkpoint**: Feature complete, all quality gates passed, ready for PR

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - US1 (Basic Path Generation) - Can start after Phase 2, no dependencies on other stories
  - US2 (Map Integration) - Can start after Phase 2, independent of US1 but benefits from US1 testing
  - US3 (Flexible Routing) - Depends on US1 (needs basic plan() method working)
  - US4 (Performance) - Depends on US1 (needs basic plan() method working)
- **Environment Integration (Phase 7)**: Depends on US1 and US2 being complete
- **Examples (Phase 8)**: Depends on US1-US4 being complete
- **Polish (Phase 9)**: Depends on all user stories being complete

### User Story Dependencies

- **US1 (P1)**: No dependencies on other stories - can start immediately after Phase 2
- **US2 (P2)**: Can start after Phase 2 - independent but benefits from US1 tests as reference
- **US3 (P3)**: Requires US1 basic plan() method - depends on T023 completion
- **US4 (P4)**: Requires US1 basic plan() method - depends on T023 completion

### Critical Path

The minimum viable product (MVP) requires:
1. Phase 1 (Setup) → Phase 2 (Foundational) → Phase 3 (US1) → Phase 7 (partial - config integration) → Phase 8 (minimal docs)

This delivers basic automated path generation capability integrated with gym environments.

### Parallel Opportunities

**Phase 1 (Setup)**:
- T003, T004, T005 can run in parallel

**Phase 2 (Foundational)**:
- T008, T010, T011, T012, T013 can run in parallel after T006-T009

**Phase 3 (US1 Tests)**:
- T014, T015, T016, T017 can all run in parallel (different test files/sections)

**Phase 4 (US2 Tests)**:
- T029, T030, T031, T032 can all run in parallel

**Phase 5 (US3 Tests)**:
- T041, T042, T043, T044, T045 can all run in parallel
- T046 (poi_sampler module creation) can run in parallel with US3 tests

**Phase 6 (US4 Tests)**:
- T055, T056, T057, T058 can all run in parallel

**Phase 7 (Environment Integration Tests)**:
- T073, T074, T075 can run in parallel after T069-T072

**Phase 8 (Examples and Documentation)**:
- T076, T077, T078, T079, T080, T081 can all run in parallel

**Phase 9 (Polish)**:
- T084, T085, T089, T090 can run in parallel
- T086, T087, T088 should run sequentially after implementation tasks

### Within Each User Story

**US1 Implementation Order**:
1. Tests first (T014-T017) - all can run in parallel
2. Then: T018 (inflate obstacles) → T019-T021 (visibility graph) can run in parallel
3. Then: T022-T027 (GlobalPlanner core) - sequential
4. Finally: T028 (verify tests pass)

**US2 Implementation Order**:
1. Tests first (T029-T032) - all can run in parallel
2. Then: T033-T037 (SVG parser) - sequential (modifying same file)
3. Then: T038 (map updates), T039 (validation) can run in parallel
4. Finally: T040 (verify tests pass)

**US3 Implementation Order**:
1. Tests first (T041-T045) - all can run in parallel
2. Then: T046-T049 (POISampler) - sequential within same module
3. Then: T050-T053 (GlobalPlanner POI support) - sequential (same file)
4. Finally: T054 (verify tests pass)

**US4 Implementation Order**:
1. Tests first (T055-T058) - all can run in parallel
2. Then: T059-T063 (caching) - sequential (same file)
3. Then: T064-T067 (smoothing) - sequential within module
4. Finally: T068 (performance verification)

---

## Parallel Example: User Story 1

```bash
# Terminal 1: Write tests (can start immediately after Phase 2)
# T014-T017 in parallel (different test cases)
code tests/test_planner/test_global_planner.py

# Terminal 2: Implement visibility graph module (can start with tests)
# T019-T021
code robot_sf/planner/visibility_graph.py

# Terminal 3: Implement obstacle inflation (can start with tests)
# T018
code robot_sf/planner/global_planner.py  # _inflate_obstacles method

# After visibility graph and inflation complete, continue sequentially:
# T022-T027 in robot_sf/planner/global_planner.py
# T028 verify tests
```

---

## Implementation Strategy

### MVP Scope (Minimum Viable Product)

**Target**: Deliver User Story 1 (Basic Path Generation) integrated with gym environments

**Includes**:
- Phase 1: Setup
- Phase 2: Foundational
- Phase 3: User Story 1 (Basic Path Generation)
- Phase 7: Minimal integration (T069-T072 only)
- Phase 8: Minimal docs (quickstart verification only)

**Estimated Effort**: 1-1.5 weeks (1 developer)

**Deliverables**:
- Working GlobalPlanner class generating collision-free paths
- Integration with make_robot_env() via use_planner config flag
- Unit tests with >85% coverage for core planner
- Basic documentation and one runnable example

### Full Feature Scope

**Target**: All user stories (US1-US4) plus complete documentation

**Estimated Effort**: 3-4 weeks (1 developer)

**Deliverables**:
- All 4 user stories implemented and tested
- POI routing and multi-goal planning
- Performance optimizations (caching, smoothing)
- Complete test suite (>90% coverage)
- Full documentation and examples
- Validation scripts and benchmarks

### Incremental Delivery Plan

1. **Week 1**: MVP (Phases 1-3 + minimal Phase 7) - Basic path generation working
2. **Week 2**: US2 (Map Integration) + US3 (Flexible Routing) - POI support added
3. **Week 3**: US4 (Performance) + complete Phase 7 - Performance targets met
4. **Week 4**: Phases 8-9 - Documentation, examples, polish

Each week delivers a working, testable increment that can be demonstrated to stakeholders.

---

## Task Summary

**Total Tasks**: 93
- Phase 1 (Setup): 5 tasks
- Phase 2 (Foundational): 8 tasks
- Phase 3 (US1 - Basic Path Generation): 15 tasks (4 test + 11 implementation)
- Phase 4 (US2 - Map Integration): 12 tasks (4 test + 8 implementation)
- Phase 5 (US3 - Flexible Routing): 14 tasks (5 test + 9 implementation)
- Phase 6 (US4 - Performance): 14 tasks (4 test + 10 implementation)
- Phase 7 (Environment Integration): 7 tasks
- Phase 8 (Examples and Documentation): 8 tasks
- Phase 9 (Polish): 10 tasks

**Tasks per User Story**:
- US1 (P1): 15 tasks (MVP critical path)
- US2 (P2): 12 tasks
- US3 (P3): 14 tasks
- US4 (P4): 14 tasks

**Parallel Opportunities**: 35 tasks marked [P] can run in parallel within their phases

**Independent Test Criteria**:
- US1: Load map, generate path, verify obstacle avoidance
- US2: Use planner output with RouteNavigator, verify tracking
- US3: Sample multiple start/goal pairs, verify all paths valid
- US4: Measure 100 queries, verify median <100ms

**MVP Scope**: Phases 1-3 + minimal 7 = ~28 tasks (1-1.5 weeks)

**Full Feature**: All 93 tasks (3-4 weeks)

---

## Format Validation ✅

All tasks follow required checklist format:
- ✅ Checkbox prefix `- [ ]`
- ✅ Task IDs (T001-T093) in execution order
- ✅ [P] markers for parallelizable tasks (35 total)
- ✅ [Story] labels for user story phases (US1-US4)
- ✅ Exact file paths in descriptions
- ✅ Clear action descriptions

**Ready for execution** - each task is specific enough for LLM completion without additional context.
