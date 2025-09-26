# Tasks: Type Checking Fixes

**Input**: Design documents from `/specs/138-phase-1-critical/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
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
- [ ] T001 Verify uvx ty type checker installation and configuration
- [ ] T002 [P] Run baseline type check to capture current 103 diagnostics
- [ ] T003 [P] Set up Python 3.11+ compatibility testing environment

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [ ] T004 [P] Contract test for run_type_check endpoint in tests/contract/test_type_checking_api.py
- [ ] T005 [P] Contract test for validate_compatibility function in tests/contract/test_python_compatibility.py
- [ ] T006 [P] Integration test for type checking validation workflow in tests/integration/test_type_validation.py
- [ ] T007 [P] Integration test for factory function type safety in tests/integration/test_factory_types.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [ ] T008 [P] Fix datetime.UTC imports for Python 3.11+ compatibility in affected files
- [ ] T009 [P] Address missing required arguments in factory functions
- [ ] T010 [P] Fix invalid type assignments that could cause runtime errors
- [ ] T011 [P] Update return type annotations for environment factories
- [ ] T012 [P] Fix parameter defaults in data analysis functions
- [ ] T013 [P] Resolve Gym space type issues in reinforcement learning contexts
- [ ] T014 [P] Add conditional imports for optional dependencies
- [ ] T015 [P] Fix dynamic import type issues
- [ ] T016 [P] Resolve test utility type problems
- [ ] T017 [P] Add missing type annotations throughout codebase
- [ ] T018 [P] Improve generic type usage
- [ ] T019 [P] Enhance type safety in utility functions

## Phase 3.4: Integration
- [ ] T020 Run type checking validation after Phase 1 fixes
- [ ] T021 Run type checking validation after Phase 2 fixes
- [ ] T022 Run type checking validation after Phase 3 fixes
- [ ] T023 Run type checking validation after Phase 4 fixes

## Phase 3.5: Polish
- [ ] T024 [P] Verify 80% type coverage target achieved
- [ ] T025 [P] Run full test suite to ensure no regressions
- [ ] T026 [P] Update documentation with type safety improvements
- [ ] T027 [P] Validate backward compatibility maintained

## Dependencies
- Setup (T001-T003) before tests (T004-T007)
- Tests (T004-T007) before implementation (T008-T019)
- Phase 1 fixes (T008-T010) before Phase 2 (T011-T013)
- Phase 2 before Phase 3 (T014-T016)
- Phase 3 before Phase 4 (T017-T019)
- Implementation before integration (T020-T023)
- Integration before polish (T024-T027)

## Parallel Example
```
# Launch T008-T010 together (Phase 1 critical fixes):
Task: "Fix datetime.UTC imports for Python 3.11+ compatibility in affected files"
Task: "Address missing required arguments in factory functions"
Task: "Fix invalid type assignments that could cause runtime errors"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify tests fail before implementing type fixes
- Commit after each task completion
- Run type checks after each phase to validate progress
- Maintain backward compatibility throughout

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
- [ ] No task modifies same file as another [P] task