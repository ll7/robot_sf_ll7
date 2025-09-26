# Tasks: Consolidate Episode Schema Definitions

**Input**: Design documents from `/specs/136-consolidate-episode-schema/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: Python 3.x, JSON Schema validation, git hooks, pathlib
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
- [ ] T001 Create project structure per implementation plan
- [ ] T002 Initialize Python project with JSON Schema validation dependencies
- [ ] T003 [P] Configure linting and formatting tools

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [ ] T004 [P] Contract test for schema-loader-api.v1.json in tests/contract/test_schema_loader_api.py
- [ ] T005 [P] Contract test for git-hook-api.v1.json in tests/contract/test_git_hook_api.py
- [ ] T006 [P] Integration test schema consolidation in tests/integration/test_schema_consolidation.py
- [ ] T007 [P] Integration test runtime resolution in tests/integration/test_runtime_resolution.py
- [ ] T008 [P] Integration test git hook prevention in tests/integration/test_git_hook_prevention.py
- [ ] T009 [P] Integration test version detection in tests/integration/test_version_detection.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [ ] T010 [P] EpisodeSchema entity in robot_sf/benchmark/schemas/episode_schema.py
- [ ] T011 [P] SchemaReference entity in robot_sf/benchmark/schema_reference.py
- [ ] T012 [P] SchemaVersion entity in robot_sf/benchmark/schema_version.py
- [ ] T013 load_schema function in robot_sf/benchmark/schema_loader.py
- [ ] T014 get_schema_version function in robot_sf/benchmark/schema_loader.py
- [ ] T015 prevent_schema_duplicates function in hooks/prevent-schema-duplicates.py
- [ ] T016 Semantic versioning utilities in robot_sf/benchmark/version_utils.py
- [ ] T017 Schema validation utilities in robot_sf/benchmark/validation_utils.py

## Phase 3.4: Integration
- [ ] T018 Remove duplicate schema file from specs/120-social-navigation-benchmark-plan/contracts/
- [ ] T019 Update existing contract tests to use new schema loader
- [ ] T020 Integrate git hook with pre-commit configuration
- [ ] T021 Add schema caching to schema loader

## Phase 3.5: Polish
- [ ] T022 [P] Unit tests for schema validation in tests/unit/test_schema_validation.py
- [ ] T023 [P] Unit tests for version utilities in tests/unit/test_version_utils.py
- [ ] T024 Performance tests for schema loading (<100ms)
- [ ] T025 [P] Update docs/dev_guide.md with canonical schema location
- [ ] T026 [P] Update docs/benchmark.md with schema consolidation details
- [ ] T027 Remove duplication in schema loading code
- [ ] T028 Run quickstart.md validation scenarios

## Dependencies
- Tests (T004-T009) before implementation (T010-T017)
- T010, T011, T012 block T013, T014 (entities before functions)
- T013 blocks T019 (schema loader before test updates)
- T015 blocks T020 (hook implementation before integration)
- T016 blocks T017 (version utils before validation utils)
- Implementation before polish (T022-T028)
- T018 blocks T025, T026 (consolidation before documentation)

## Parallel Example
```
# Launch T004-T009 together (all test files are different):
Task: "Contract test for schema-loader-api.v1.json in tests/contract/test_schema_loader_api.py"
Task: "Contract test for git-hook-api.v1.json in tests/contract/test_git_hook_api.py"
Task: "Integration test schema consolidation in tests/integration/test_schema_consolidation.py"
Task: "Integration test runtime resolution in tests/integration/test_runtime_resolution.py"
Task: "Integration test git hook prevention in tests/integration/test_git_hook_prevention.py"
Task: "Integration test version detection in tests/integration/test_version_detection.py"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify tests fail before implementing
- Commit after each task
- Avoid: vague tasks, same file conflicts
- Schema consolidation requires removing duplicate file in T018
- Git hook integration requires updating .pre-commit-config.yaml in T020

## Task Generation Rules
- Each contract file → contract test task marked [P]
- Each entity in data-model → model creation task marked [P]
- Each endpoint → implementation task (not parallel if shared files)
- Each user story → integration test marked [P]
- Different files = can be parallel [P]
- Same file = sequential (no [P])</content>
<parameter name="filePath">/Users/lennart/git/robot_sf_ll7/specs/136-consolidate-episode-schema/tasks.md