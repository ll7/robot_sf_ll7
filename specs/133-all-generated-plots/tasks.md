# Tasks: Fix Benchmark Placeholder Outputs

**I## Ph## Pha## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [x] T004 [P] Contract test generate_benchmark_plots in tests/test_visualization_plots.py
- [x] T005 [P] Contract test generate_benchmark_videos in tests/test_visualization_videos.py
- [x] T006 [P] Contract test validate_visual_artifacts in tests/test_visualization_validation.py
- [x] T007 [P] Integration test benchmark with real visualizations in tests/test_benchmark_visualization_integration.py
- [x] T008 [P] Integration test placeholder detection in tests/test_placeholder_detection.py Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [x] T004 [P] Contract test generate_benchmark_plots in tests/test_visualization_plots.py
- [x] T005 [P] Contract test generate_benchmark_videos in tests/test_visualization_videos.py
- [x] T006 [P] Contract test validate_visual_artifacts in tests/test_visualization_validation.py
- [x] T007 [P] Integration test benchmark with real visualizations in tests/test_benchmark_visualization_integration.py
- [ ] T008 [P] Integration test placeholder detection in tests/test_placeholder_detection.py: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [x] T004 [P] Contract test generate_benchmark_plots in tests/test_visualization_plots.py
- [x] T005 [P] Contract test generate_benchmark_videos in tests/test_visualization_videos.py
- [x] T006 [P] Contract test validate_visual_artifacts in tests/test_visualization_validation.py
- [ ] T007 [P] Integration test benchmark with real visualizations in tests/test_benchmark_visualization_integration.py: Design documents from `/specs/133-all-generated-plots/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: Python 3.13, Matplotlib, MoviePy, NumPy, Gymnasium, StableBaselines3
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: DB, middleware, logging
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have models?
   → All endpoints implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `robot_sf/`, `tests/` at repository root
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 3.1: Setup
- [x] T001 Create visualization module structure in robot_sf/benchmark/visualization.py
- [x] T002 Add matplotlib and moviepy dependencies to pyproject.toml if not present
- [x] T003 [P] Configure type hints for new visualization functions

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [x] T004 [P] Contract test generate_benchmark_plots in tests/test_visualization_plots.py
- [x] T005 [P] Contract test generate_benchmark_videos in tests/test_visualization_videos.py
- [ ] T006 [P] Contract test validate_visual_artifacts in tests/test_visualization_validation.py
- [ ] T007 [P] Integration test benchmark with real visualizations in tests/test_benchmark_visualization_integration.py
- [ ] T008 [P] Integration test placeholder detection in tests/test_placeholder_detection.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [x] T009 [P] VisualArtifact dataclass in robot_sf/benchmark/visualization.py
- [x] T010 [P] VisualizationError exception class in robot_sf/benchmark/visualization.py
- [ ] T011 [P] generate_benchmark_plots function in robot_sf/benchmark/visualization.py
- [ ] T012 [P] generate_benchmark_videos function in robot_sf/benchmark/visualization.py
- [ ] T013 [P] validate_visual_artifacts function in robot_sf/benchmark/visualization.py
- [ ] T014 [P] _check_dependencies helper function in robot_sf/benchmark/visualization.py
- [ ] T015 Extend benchmark orchestrator to call visualization functions in robot_sf/benchmark/orchestrator.py
- [ ] T016 Add visualization phase to benchmark pipeline in robot_sf/benchmark/orchestrator.py

## Phase 3.4: Integration
- [ ] T017 Connect visualization functions to episode data parsing in robot_sf/benchmark/visualization.py
- [ ] T018 Integrate with environment factory for video rendering in robot_sf/benchmark/visualization.py
- [ ] T019 Add error handling for missing dependencies in robot_sf/benchmark/visualization.py
- [ ] T020 Add logging for visualization generation progress in robot_sf/benchmark/visualization.py

## Phase 3.5: Polish
- [ ] T021 [P] Unit tests for data processing helpers in tests/test_visualization_helpers.py
- [ ] T022 Performance tests for visualization generation (< 60s) in tests/test_visualization_performance.py
- [ ] T023 [P] Update docs for new visualization functions in docs/benchmark.md
- [ ] T024 [P] Add troubleshooting section to docs for visualization issues in docs/benchmark.md
- [ ] T025 Update benchmark CLI help text to mention real visualizations in robot_sf/benchmark/cli.py
- [ ] T026 Run validation scripts to ensure visualizations work in scripts/validation/

## Dependencies
- Tests (T004-T008) before implementation (T009-T016)
- T009-T010 blocks T011-T014
- T011-T014 blocks T015-T016
- T015-T016 blocks T017-T020
- Implementation before polish (T021-T026)

## Parallel Example
```
# Launch T004-T008 together:
Task: "Contract test generate_benchmark_plots in tests/test_visualization_plots.py"
Task: "Contract test generate_benchmark_videos in tests/test_visualization_videos.py"
Task: "Contract test validate_visual_artifacts in tests/test_visualization_validation.py"
Task: "Integration test benchmark with real visualizations in tests/test_benchmark_visualization_integration.py"
Task: "Integration test placeholder detection in tests/test_placeholder_detection.py"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify tests fail before implementing
- Commit after each task
- Avoid: vague tasks, same file conflicts

## Task Generation Rules
*Applied during main() execution*

1. **From Contracts**:
   - Each contract file → contract test task [P]
   - Each endpoint → implementation task

2. **From Data Model**:
   - Each entity → model creation task [P]
   - Relationships → service layer tasks

3. **From User Stories**:
   - Each story → integration test [P]
   - Quickstart scenarios → validation tasks

4. **Ordering**:
   - Setup → Tests → Models → Services → Endpoints → Polish
   - Dependencies block parallel execution

## Validation Checklist
*GATE: Checked by main() before returning*

- [ ] All contracts have corresponding tests
- [ ] All entities have model tasks
- [ ] All tests come before implementation
- [ ] Parallel tasks truly independent
- [ ] Each task specifies exact file path
- [ ] No task modifies same file as another [P] task</content>
<parameter name="filePath">/Users/lennart/git/robot_sf_ll7/specs/133-all-generated-plots/tasks.md