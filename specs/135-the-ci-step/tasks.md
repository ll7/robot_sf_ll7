# Tasks: Optimize CI System Package Installation

**Input- [x] T010 [P] CIJob monitoring setup in scripts/ci_monitoring.py
- [x] T011 [P] SystemPackage validation in scripts/ci-tests/package_validation.py
- [x] T012 [P] PerformanceMetric collection in scripts/ci-tests/performance_metrics.py Design documents from `/Users/lennart/git/robot_sf_ll7/specs/135-the-ci-step/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Extract: YAML/GitHub Actions workflows, Python 3.13, Ubuntu system packages
2. Load optional design documents:
   → data-model.md: Extract entities → monitoring tasks
   → contracts/: workflow contract → validation tasks
   → research.md: Extract decisions → CI optimization tasks
3. Generate tasks by category:
   → Setup: CI environment preparation, tool installation
   → Tests: contract tests, integration tests for performance/reliability
   → Core: CI workflow modifications, caching implementation
   → Integration: performance monitoring, validation
   → Polish: documentation, cleanup
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have monitoring?
   → All scenarios validated?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **CI Workflow**: `.github/workflows/ci.yml` at repository root
- **Test Files**: `specs/135-the-ci-step/contracts/` for contract tests
- **Scripts**: Repository root for CI validation scripts

## Phase 3.1: Setup
- [x] T001 Install apt-fast and caching tools in CI environment
- [x] T002 [P] Configure CI performance monitoring scripts

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [x] T003 [P] Contract test for CI workflow interface in specs/135-the-ci-step/contracts/test_ci_workflow_contract.py
- [x] T004 [P] Integration test for performance validation in scripts/ci-tests/test_ci_performance.py
- [x] T005 [P] Integration test for reliability validation in scripts/ci-tests/test_ci_reliability.py
- [x] T006 [P] Integration test for functionality preservation in scripts/ci-tests/test_ci_functionality.py
- [x] T007 [P] Integration test for cache effectiveness in scripts/ci-tests/test_ci_cache.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [x] T008 Modify CI workflow to add package caching in .github/workflows/ci.yml
- [x] T009 Modify CI workflow to install and use apt-fast in .github/workflows/ci.yml
- [x] T010 [P] CIJob monitoring setup in scripts/ci_monitoring.py
- [x] T011 [P] SystemPackage validation in scripts/ci-tests/package_validation.py
- [x] T012 [P] PerformanceMetric collection in scripts/ci-tests/performance_metrics.py

## Phase 3.4: Integration
- [x] T013 Integrate performance monitoring with CI workflow
- [x] T014 Test CI workflow modifications end-to-end
- [x] T015 Validate package installation reliability
- [x] T016 Measure and validate performance improvements

## Phase 3.5: Polish
- [ ] T017 [P] Update CI documentation in docs/dev_guide.md
- [ ] T018 [P] Add performance monitoring to CI dashboard
- [ ] T019 Clean up temporary CI optimization scripts
- [ ] T020 Final validation of 50% performance improvement

## Dependencies
- Setup (T001-T002) before tests (T003-T007)
- Tests (T003-T007) before implementation (T008-T012)
- Core implementation (T008-T012) before integration (T013-T016)
- Integration (T013-T016) before polish (T017-T020)
- T008 blocks T009 (same file .github/workflows/ci.yml)
- T010-T012 can run in parallel (different files)

## Parallel Example
```
# Launch T003-T007 together:
Task: "Contract test for CI workflow interface in specs/135-the-ci-step/contracts/test_ci_workflow_contract.py"
Task: "Integration test for performance validation in scripts/test_ci_performance.py"
Task: "Integration test for reliability validation in scripts/test_ci_reliability.py"
Task: "Integration test for functionality preservation in scripts/test_ci_functionality.py"
Task: "Integration test for cache effectiveness in scripts/test_ci_cache.py"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify tests fail before implementing CI changes
- Commit after each task to maintain git history
- Test CI changes on feature branch before merging
- Monitor CI performance metrics throughout implementation

## Task Generation Rules
*Applied during main() execution*

1. **From Contracts**:
   - workflow-contract.md → T003 contract test task [P]
   - test_ci_workflow_contract.py → existing test file, validate it runs

2. **From Data Model**:
   - CIJob entity → T010 monitoring setup task [P]
   - SystemPackage entity → T011 validation task [P]
   - PerformanceMetric entity → T012 collection task [P]

3. **From User Stories**:
   - Performance validation scenario → T004 integration test [P]
   - Reliability validation scenario → T005 integration test [P]
   - Functionality preservation scenario → T006 integration test [P]
   - Cache effectiveness scenario → T007 integration test [P]

4. **Ordering**:
   - Setup → Tests → Core → Integration → Polish
   - CI workflow modifications sequential due to same file

## Validation Checklist
*GATE: Checked by main() before returning*

- [x] All contracts have corresponding tests (workflow-contract.md → T003)
- [x] All entities have implementation tasks (CIJob → T010, SystemPackage → T011, PerformanceMetric → T012)
- [x] All test scenarios have integration tests (4 scenarios → T004-T007)
- [x] Tests come before implementation (T003-T007 before T008-T012)
- [x] Parallel tasks are independent (different file paths)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task</content>
<parameter name="filePath">/Users/lennart/git/robot_sf_ll7/specs/135-the-ci-step/tasks.md