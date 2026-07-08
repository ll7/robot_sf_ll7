# Tasks: Great Starting Set for Ruff Linting Rules Expansion

**Input**: Design documents from `/specs/137-great-starting-set/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack (Python, Ruff), libraries (Ruff), structure (pyproject.toml update)
2. Load optional design documents:
   → data-model.md: Extract entities (pyproject.toml configuration) → configuration tasks
   → contracts/: linting-contract.md → contract test task
   → research.md: Extract decisions (rule selections) → validation tasks
3. Generate tasks by category:
   → Setup: project configuration, dependencies
   → Tests: contract tests, integration tests
   → Core: configuration updates
   → Integration: CI validation
   → Polish: documentation updates
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation where possible
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → Contract has test? Yes
   → Configuration implemented? Yes
   → CI integration tested? Yes
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: Repository root with pyproject.toml
- Paths assume repository root structure

## Phase 3.1: Setup
- [X] T001 Update pyproject.toml with expanded [tool.ruff.lint] configuration in pyproject.toml

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [X] T002 [P] Contract test for linting behavior in tests/test_linting_contract.py
- [X] T003 [P] Integration test for CI linting performance in tests/test_ci_performance.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [X] T004 Validate new configuration on existing codebase using ruff check
- [X] T005 Apply auto-fixes with ruff check --fix
- [X] T006 Manually address remaining linting issues

## Phase 3.4: Integration
- [X] T007 Verify CI pipeline runs linting within 60 seconds
- [X] T008 Confirm no new CI failures introduced

## Phase 3.5: Polish
- [X] T009 [P] Update development guide with linting guidelines in docs/dev_guide.md
- [X] T010 [P] Add linting section to README.md in README.md
- [X] T011 Run validation scripts to ensure no regressions

## Dependencies
- Setup (T001) before tests (T002-T003) and core (T004-T006)
- Tests (T002-T003) before core implementation (T004-T006)
- Core (T004-T006) before integration (T007-T008)
- Everything before polish (T009-T011)

## Parallel Example
```
# Launch T002-T003 together:
Task: "Contract test for linting behavior in tests/test_linting_contract.py"
Task: "Integration test for CI linting performance in tests/test_ci_performance.py"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify tests fail before implementing configuration changes
- Commit after each task
- Avoid: vague tasks, same file conflicts

## Task Generation Rules
- Contract file (linting-contract.md) → T002 contract test
- Entity (pyproject.toml config) → T001 configuration update
- User stories (acceptance scenarios) → T003 integration test
- Research decisions → T004-T006 validation tasks
- Quickstart scenarios → T007-T008 CI integration