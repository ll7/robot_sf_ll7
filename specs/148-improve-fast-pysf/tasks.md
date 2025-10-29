---
description: "Task list for fast-pysf integration quality improvements"
---

# Tasks: Improve fast-pysf Integration

**Input**: Design documents from `/specs/148-improve-fast-pysf/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Not explicitly requested in spec - tasks focus on integration, quality tooling, and PR review resolution

**Organization**: Tasks grouped by user story priority (P1 → P1 → P2 → P3) to enable independent implementation and testing

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions
- Repository root: `/Users/lennart/git/robot_sf_ll7/`
- Main tests: `tests/`
- fast-pysf tests: `fast-pysf/tests/`
- Configuration: `pyproject.toml`
- Documentation: `docs/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Verify baseline and prepare for test integration

- [ ] T001 [P] Verify current test baseline: Run `uv run pytest tests -v` and confirm ~43 tests pass
- [ ] T002 [P] Verify fast-pysf tests can run independently: `uv run pytest fast-pysf/tests -v` (expect 10 pass, 2 fail)
- [ ] T003 [P] Document baseline metrics in `specs/148-improve-fast-pysf/baseline_metrics.md`:
  - robot_sf test count: ~43
  - fast-pysf test count: 12 (10 pass, 2 fail)
  - Current coverage: ~91.73% for robot_sf
  - ruff status: passes for robot_sf only
  - ty status: warnings for robot_sf only

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Create test fixtures and update configuration - MUST complete before user stories

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Create test fixture directory: `mkdir -p fast-pysf/tests/test_maps/`
- [ ] T005 [P] Create valid map fixture in `fast-pysf/tests/test_maps/map_regular.json`:
  ```json
  {
    "obstacles": [[5.0, 5.0], [10.0, 10.0], [15.0, 5.0]],
    "routes": [{"id": "route1", "waypoints": [[0.0, 0.0], [20.0, 20.0]]}],
    "crowded_zones": [{"center": [10.0, 10.0], "radius": 5.0}]
  }
  ```
- [ ] T006 [P] Create invalid map fixture in `fast-pysf/tests/test_maps/invalid_json_file.json`:
  ```json
  {
    "obstacles": "not_an_array",
    "routes": [1, 2, 3]
  }
  ```
- [ ] T007 Verify fixtures resolve test failures: `uv run pytest fast-pysf/tests/test_map_loader.py -v` (expect all pass)

**Checkpoint**: Foundation ready - all 12 fast-pysf tests should pass, user story implementation can begin

---

## Phase 3: User Story 1 - Seamless Test Execution (Priority: P1) 🎯 MVP

**Goal**: Enable unified test execution with single `uv run pytest` command for both robot_sf and fast-pysf tests

**Independent Test**: Run `uv run pytest` from repository root and verify both test suites execute (55+ tests total)

### Configuration Tasks

- [ ] T008 [US1] Update pytest configuration in `pyproject.toml`:
  - Locate `[tool.pytest.ini_options]` section (around line 97)
  - Change `testpaths = ["tests"]` to `testpaths = ["tests", "fast-pysf/tests"]`
  - Add marker: `markers = ["fast_pysf: marks tests from fast-pysf subtree"]`

- [ ] T009 [US1] Update coverage configuration in `pyproject.toml`:
  - Locate `[tool.coverage.run]` section (around line 124)
  - Add `"fast-pysf/pysocialforce"` to `source` list
  - Add `"fast-pysf/examples/*"` to `omit` patterns

### Verification Tasks

- [ ] T010 [US1] Test unified execution: `uv run pytest -v` (expect 55+ tests, all pass)
- [ ] T011 [US1] Test selective execution: `uv run pytest tests -v` (expect ~43 tests, robot_sf only)
- [ ] T012 [US1] Test fast-pysf only: `uv run pytest fast-pysf/tests -v` (expect 12 tests)
- [ ] T013 [US1] Test headless mode: `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest -v`
- [ ] T014 [US1] Test parallel execution: `uv run pytest -n auto -v` (expect same results, faster)
- [ ] T015 [US1] Verify coverage includes fast-pysf: Check `coverage.json` for `fast-pysf/pysocialforce/` entries

### Documentation Tasks

- [ ] T016 [P] [US1] Update `docs/dev_guide.md` testing section:
  - Add "Unified Test Suite" subsection under "Testing strategy"
  - Document `uv run pytest` runs both robot_sf and fast-pysf tests
  - Update test count from ~43 to 55+
  - Add example: "Run only fast-pysf tests: `uv run pytest fast-pysf/tests`"

- [ ] T017 [P] [US1] Update `README.md` quick start section:
  - Update test command examples to reflect unified suite
  - Add note about fast-pysf test integration
  - Update expected test count in examples

- [ ] T018 [P] [US1] Create `fast-pysf/tests/test_maps/README.md`:
  - Document map fixture JSON schema
  - Explain purpose of valid vs invalid fixtures
  - Provide examples for adding new test maps

**Checkpoint**: User Story 1 complete - unified test execution working ✅

**Success Criteria Verification**:
- [ ] SC-001: Test count increases from ~43 to 55+ ✅
- [ ] SC-002: Fast-pysf tests complete in <60 seconds ✅
- [ ] SC-003: Zero test failures (all fixtures created) ✅

---

## Phase 4: User Story 2 - Resolved Code Quality Issues (Priority: P1)

**Goal**: Systematically address all 24 PR #236 review comments with documented resolution status

**Independent Test**: Verify all high-priority issues resolved and quality gates pass

### High-Priority Issues (7 total - 4-6 hours)

- [ ] T019 [P] [US2] Fix unreachable print in `fast-pysf/pysocialforce/map_config.py:81`:
  - Investigate conditional logic around line 81
  - Either remove unreachable code or fix condition
  - Verify with: `uv run ruff check fast-pysf/pysocialforce/map_config.py`

- [ ] T020 [P] [US2] Verify test result in `fast-pysf/tests/unittest/test_forces.py:44` (TestObstacleForce):
  - Review test at line 44
  - Add explicit assertion or verification comment
  - Run test: `uv run pytest fast-pysf/tests/unittest/test_forces.py::TestObstacleForce -v`

- [ ] T021 [US2] Implement or document empty reset in `fast-pysf/pysocialforce/ped_behavior.py:114`:
  - Review PedBehavior.reset() method
  - Either add reset logic or document why empty is correct
  - Consider upstream coordination if state management needed

- [ ] T022 [P] [US2] Extract magic numbers in `fast-pysf/pysocialforce/map_osm_converter.py:36`:
  - Replace hardcoded 1350 and 4.08 with named constants
  - Add constants at module level with descriptive names
  - Example: `OSM_SCALE_FACTOR = 1350` and `COORDINATE_OFFSET = 4.08`

- [ ] T023 [US2] Resolve TODO for scale factor in `fast-pysf/pysocialforce/map_osm_converter.py:35`:
  - Investigate hardcoded scale factor TODO
  - Either make configurable via parameter or document decision
  - Update function signature if made configurable

- [ ] T024 [P] [US2] Clarify ortho_vec documentation in `fast-pysf/pysocialforce/forces.py:397`:
  - Review ortho_vec function docstring
  - Add clear explanation of vector direction (clockwise vs counter-clockwise)
  - Include mathematical formula or visual description

- [ ] T025 [P] [US2] Verify polygon closing logic in `fast-pysf/pysocialforce/map_loader_svg.py:96`:
  - Review polygon closing question at line 96
  - Add comment explaining behavior (auto-close or explicit)
  - Verify with test case if available

### Medium-Priority Issues (10 total - 2-3 hours)

- [ ] T026 [P] [US2] Fix broken CI badge in `fast-pysf/README.md:3`:
  - Update badge URL or remove if repository moved
  - Verify badge displays correctly on GitHub

- [ ] T027 [P] [US2] Remove redundant empty check in `fast-pysf/pysocialforce/forces.py:770`:
  - Review unnecessary conditional at line 770
  - Simplify logic if check is truly redundant

- [ ] T028 [P] [US2] Fix copy-paste docstring in `fast-pysf/examples/example06.py:2`:
  - Update docstring to match actual example content
  - Ensure description is accurate

- [ ] T029 [P] [US2] Fill placeholder migration metadata in `docs/SUBTREE_MIGRATION.md:246-247`:
  - Replace `[TODO: Date]` and `[TODO: Commit]` with actual values
  - Verify migration history accuracy

- [ ] T030 [P] [US2] Update GitHub Actions versions in `.github/workflows/ci.yml:15-23`:
  - Upgrade actions/checkout from v2 to v4
  - Upgrade actions/setup-python from v2 to v5
  - Test CI pipeline after upgrade

- [ ] T031 [P] [US2] Fix spelling: "Fucntion" → "Function" in relevant file
- [ ] T032 [P] [US2] Fix spelling: "Tehere" → "There" in relevant file
- [ ] T033 [P] [US2] Fix spelling: "verices" → "vertices" in relevant file
- [ ] T034 [P] [US2] Fix spelling: "pedstrains" → "pedestrians" in relevant file
- [ ] T035 [P] [US2] Fix spelling: "approximetly" → "approximately" in relevant file
- [ ] T036 [P] [US2] Fix spelling: "thh" → "the" in relevant file

### Low-Priority Issues (7 total - 1-2 hours)

- [ ] T037 [P] [US2] Remove commented code in `fast-pysf/pysocialforce/scene.py:90-91`:
  - Delete commented lines if no longer needed
  - Or add explanatory comment if kept for reference

- [ ] T038 [P] [US2] Fix unused loop variable in example files (5 instances):
  - Replace unused `_` with descriptive name or use `enumerate()` if index needed
  - Files: various examples in `fast-pysf/examples/`

- [ ] T039 [P] [US2] Remove duplicate simulator assignment in `fast-pysf/examples/example09.py:19`:
  - Consolidate duplicate variable assignments
  - Ensure no side effects

- [ ] T040 [P] [US2] Improve import style in `fast-pysf/pysocialforce/simulator.py:15`:
  - Make import statement consistent with project style
  - Group imports appropriately

- [ ] T041 [P] [US2] Replace wildcard import in `fast-pysf/tests/unittest/TestObstacleForce.py:2`:
  - Change `from module import *` to explicit imports
  - Improves clarity and IDE support

- [ ] T042 [P] [US2] Fix markdown indentation in `.specify/prompts/generate_issue.prompt.md:87-88`:
  - Correct list indentation
  - Verify markdown renders correctly

### Tracking and Verification

- [ ] T043 [US2] Update resolution status in `specs/148-improve-fast-pysf/pr236_review_comments.md`:
  - Mark each resolved comment with status "Resolved"
  - Add resolution note and commit SHA for each fix
  - Calculate resolution metrics (24/24 tracked, high-priority complete)

- [ ] T044 [US2] Run quality gates to verify all fixes:
  - `uv run ruff check . --fix` (auto-fix what's possible)
  - `uv run ruff format .` (format all code)
  - `uv run ruff check .` (verify clean)
  - `uv run pylint robot_sf --errors-only` (check robot_sf)
  - `uvx ty check . --exit-zero` (type check)
  - `uv run pytest` (all tests pass)

**Checkpoint**: User Story 2 complete - all PR review comments addressed ✅

**Success Criteria Verification**:
- [ ] SC-004: All 24 comments tracked (100%) ✅
- [ ] SC-005: Zero high-priority issues open ✅

---

## Phase 5: User Story 3 - Unified Code Quality Standards (Priority: P2)

**Goal**: Extend ruff, ty, and coverage to fast-pysf directory with same standards as robot_sf

**Independent Test**: Run quality gates on fast-pysf and verify they execute without configuration errors

### Ruff Configuration

- [ ] T045 [US3] Configure ruff for fast-pysf in `pyproject.toml`:
  - Locate `[tool.ruff.lint.per-file-ignores]` section (around line 80)
  - Add per-file ignores for fast-pysf:
    ```toml
    "fast-pysf/**/*.py" = [
        "C901",    # Complex function (relax initially)
        "PLR0912", # Too many branches
        "PLR0913", # Too many arguments
        "PLR0915", # Too many statements
    ]
    ```
  - Document rationale: "Gradual adoption - complexity rules relaxed for fast-pysf subtree"

- [ ] T046 [US3] Run ruff on fast-pysf: `uv run ruff check fast-pysf/ --fix`
  - Apply auto-fixes
  - Document remaining issues in `specs/148-improve-fast-pysf/ruff_baseline.md`

- [ ] T047 [US3] Format fast-pysf code: `uv run ruff format fast-pysf/`
  - Apply consistent formatting
  - Commit formatted code

### Type Checking Configuration

- [ ] T048 [US3] Configure ty/pyright for fast-pysf in `pyproject.toml`:
  - Add or update `[tool.pyright]` section:
    ```toml
    [tool.pyright]
    include = ["robot_sf", "fast-pysf/pysocialforce"]
    exclude = ["**/tests", "**/test_*.py", "**/examples"]
    typeCheckingMode = "basic"
    reportMissingImports = true
    reportMissingTypeStubs = false
    reportUnusedImport = true
    reportUnusedVariable = true
    ```

- [ ] T049 [US3] Run type check on fast-pysf: `uvx ty check fast-pysf/pysocialforce --exit-zero`
  - Establish baseline of type errors
  - Document in `specs/148-improve-fast-pysf/type_baseline.md`:
    - Total type errors
    - Breakdown by category (missing types, Any usage, etc.)
    - Files with most issues

### Coverage Configuration (Already done in US1, verify)

- [ ] T050 [US3] Verify coverage tracking for fast-pysf:
  - Run: `uv run pytest fast-pysf/tests --cov=fast-pysf/pysocialforce --cov-report=term-missing`
  - Check coverage percentage (target ≥70%)
  - Document in `specs/148-improve-fast-pysf/coverage_baseline.md`

### Quality Gate Integration

- [ ] T051 [US3] Update VS Code task "Check Code Quality" to include fast-pysf:
  - Locate `.vscode/tasks.json` or create if missing
  - Ensure ruff checks entire repo (already does with `.`)
  - Optionally add separate task for fast-pysf only

- [ ] T052 [US3] Update CI workflow `.github/workflows/ci.yml`:
  - Verify ruff job checks entire repo
  - Verify type check job includes fast-pysf
  - Verify test job runs unified suite
  - Add coverage report for fast-pysf (if not already present)

### Documentation

- [ ] T053 [P] [US3] Update `docs/dev_guide.md` quality gates section:
  - Add note that ruff/ty now cover fast-pysf
  - Document per-file ignores for fast-pysf
  - Update expected quality gate runtime (may increase slightly)

- [ ] T054 [P] [US3] Create `fast-pysf/QUALITY_STANDARDS.md`:
  - Document ruff rules applied
  - Document type checking expectations
  - Document coverage target (≥70%)
  - Explain numba compatibility constraints
  - Link to robot_sf quality standards

**Checkpoint**: User Story 3 complete - quality tooling unified ✅

**Success Criteria Verification**:
- [ ] SC-006: Ruff passes for fast-pysf ✅
- [ ] SC-008: Quality gates run in <5 minutes ✅
- [ ] SC-013: Fast-pysf coverage ≥70% ✅

---

## Phase 6: User Story 4 - Improved Type Annotations (Priority: P3)

**Goal**: Add type annotations to fast-pysf public APIs and reduce `Any` usage by 25%

**Independent Test**: Compare type error count before/after and verify 25% reduction in undefined/Any types

### Baseline and Planning

- [ ] T055 [US4] Analyze current type annotation state:
  - Run: `uvx ty check fast-pysf/pysocialforce --exit-zero > specs/148-improve-fast-pysf/type_errors_before.txt`
  - Count total type errors
  - Identify files with most `Any` types
  - Prioritize public API functions

- [ ] T056 [US4] Create annotation plan in `specs/148-improve-fast-pysf/annotation_plan.md`:
  - List target functions (public APIs first)
  - Identify numba-decorated functions (handle carefully)
  - Estimate effort per file
  - Set success metric: 25% reduction in undefined types

### Public API Annotations

- [ ] T057 [P] [US4] Add type hints to `fast-pysf/pysocialforce/forces.py` public functions:
  - Functions: `desired_force`, `social_force`, `obstacle_force`, `ortho_vec`
  - Use NumPy array types: `np.ndarray` or `npt.NDArray[np.float64]`
  - Add return type annotations
  - Preserve numba `@njit` compatibility

- [ ] T058 [P] [US4] Add type hints to `fast-pysf/pysocialforce/simulator.py` Simulator class:
  - Annotate `__init__`, `step`, `step_once` methods
  - Use proper types for `peds`, `groups`, `obstacles`
  - Annotate `State` dataclass if present

- [ ] T059 [P] [US4] Add type hints to `fast-pysf/pysocialforce/map_loader.py` functions:
  - Functions: `load_map`, parsing helpers
  - Use proper return types for MapDefinition
  - Annotate file paths as `str | Path`

- [ ] T060 [P] [US4] Add type hints to `fast-pysf/pysocialforce/scene.py` Scene class:
  - Annotate initialization parameters
  - Type pedestrian and obstacle lists
  - Use protocols for duck-typed interfaces if needed

### Replace Any Types

- [ ] T061 [US4] Identify and replace `Any` types:
  - Search for `Any` imports and usage
  - Replace with specific types where possible
  - Use `Protocol` for duck-typed interfaces
  - Use `TypeVar` for generic functions
  - Document remaining `Any` usage with rationale

### Verification

- [ ] T062 [US4] Verify type annotations:
  - Run: `uvx ty check fast-pysf/pysocialforce --exit-zero > specs/148-improve-fast-pysf/type_errors_after.txt`
  - Compare error counts (should reduce by ≥25%)
  - Verify no new runtime errors: `uv run pytest fast-pysf/tests -v`

- [ ] T063 [US4] Verify numba compatibility:
  - Run tests with numba functions: `uv run pytest fast-pysf/tests/test_forces.py -v`
  - Check for numba compilation warnings
  - Ensure JIT-compiled functions still work

### Documentation

- [ ] T064 [P] [US4] Document type annotation strategy in `fast-pysf/TYPE_ANNOTATIONS.md`:
  - Explain numba compatibility approach
  - List fully annotated modules
  - Document remaining gaps
  - Provide examples for future contributors

- [ ] T065 [P] [US4] Update `specs/148-improve-fast-pysf/annotation_plan.md` with results:
  - Record before/after type error counts
  - Calculate reduction percentage
  - Document lessons learned
  - Suggest next steps for complete coverage

**Checkpoint**: User Story 4 complete - type annotations improved ✅

**Success Criteria Verification**:
- [ ] SC-007: Type errors reduced by ≥25% ✅

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final improvements affecting multiple user stories

- [ ] T066 [P] Update `CHANGELOG.md` with all changes:
  - Add entry for fast-pysf integration under "Changed" section
  - List PR #236 review comment resolutions under "Fixed" section
  - Document quality tooling extension under "Added" section
  - Document type annotation improvements under "Changed" section

- [ ] T067 [P] Update `docs/dev_guide.md` with comprehensive changes:
  - Update test suite section (US1)
  - Update quality gates section (US3)
  - Add troubleshooting for fast-pysf tests
  - Link to fast-pysf quality standards

- [ ] T068 [P] Create migration guide in `docs/FAST_PYSF_INTEGRATION.md`:
  - Explain git subtree approach
  - Document test integration
  - Document quality tooling integration
  - Provide examples for common workflows
  - Link to quickstart.md

- [ ] T069 Run full validation suite:
  - Execute: `./scripts/validation/test_basic_environment.sh`
  - Execute: `./scripts/validation/test_model_prediction.sh`
  - Execute: `./scripts/validation/test_complete_simulation.sh`
  - All should pass without errors

- [ ] T070 Performance validation:
  - Run: `time uv run pytest` (should complete in <5 minutes)
  - Run: `time uv run pytest fast-pysf/tests` (should complete in <60 seconds)
  - Document in `specs/148-improve-fast-pysf/performance_results.md`

- [ ] T071 Verify quickstart guide: Follow `specs/148-improve-fast-pysf/quickstart.md` step-by-step
  - Ensure all commands work as documented
  - Fix any discrepancies
  - Add troubleshooting entries if needed

- [ ] T072 Create feature summary in `specs/148-improve-fast-pysf/SUMMARY.md`:
  - List all completed user stories
  - Document success criteria met
  - Provide before/after metrics
  - Include lessons learned
  - Suggest future improvements

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (Phase 4)**: Depends on Foundational (Phase 2) - Independent of US1 (can run in parallel)
- **User Story 3 (Phase 5)**: Depends on US1 (needs test integration) - Independent of US2
- **User Story 4 (Phase 6)**: Depends on US3 (needs quality tooling) - Independent of US1/US2
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

```
Foundational (T004-T007)
    ├─> US1 (T008-T018) - Test Integration [P1]
    │   └─> US3 (T045-T054) - Quality Tooling [P2]
    │       └─> US4 (T055-T065) - Type Annotations [P3]
    │
    └─> US2 (T019-T044) - PR Review Resolution [P1] (independent, can parallel with US1)
```

### Suggested Execution Order

**Sprint 1 (Week 1)**: MVP - User Story 1
1. Complete Setup (T001-T003) - 1 hour
2. Complete Foundational (T004-T007) - 2 hours
3. Complete US1: Test Integration (T008-T018) - 4 hours
   - **Checkpoint**: Unified test execution working

**Sprint 2 (Week 2)**: Critical Quality
4. Complete US2: PR Review Resolution (T019-T044) - 7-11 hours
   - **Checkpoint**: All review comments addressed

**Sprint 3 (Week 3)**: Quality Standards
5. Complete US3: Quality Tooling (T045-T054) - 3-4 hours
   - **Checkpoint**: Quality gates unified

**Sprint 4 (Week 4)**: Enhancement
6. Complete US4: Type Annotations (T055-T065) - 4-6 hours
   - **Checkpoint**: Type safety improved

**Sprint 5 (Final)**: Polish
7. Complete Polish (T066-T072) - 2-3 hours
   - **Checkpoint**: Feature complete and documented

### Parallel Opportunities

**Within Setup (Phase 1)**: All tasks can run in parallel
- T001 (verify robot_sf tests) || T002 (verify fast-pysf tests) || T003 (document baseline)

**Within Foundational (Phase 2)**: T005 and T006 can run in parallel
- T005 (create valid fixture) || T006 (create invalid fixture)

**Within US1 (Phase 3)**: Documentation tasks can run in parallel
- T016 (update dev_guide.md) || T017 (update README.md) || T018 (create fixture README)

**Within US2 (Phase 4)**: Most fix tasks can run in parallel (different files)
- All T019-T042 marked [P] can run concurrently (23 parallel tasks!)

**Within US3 (Phase 5)**: Documentation tasks can run in parallel
- T053 (update dev_guide) || T054 (create quality standards doc)

**Within US4 (Phase 6)**: Annotation tasks can run in parallel (different files)
- T057 (forces.py) || T058 (simulator.py) || T059 (map_loader.py) || T060 (scene.py)

**Within Polish (Phase 7)**: Most documentation tasks can run in parallel
- T066 (CHANGELOG) || T067 (dev_guide) || T068 (migration guide)

**Cross-Story Parallelism** (if team capacity allows):
- US1 and US2 are independent - can work in parallel after Foundational complete
- Different developers can work on different stories simultaneously

---

## Parallel Example: User Story 2 (PR Review Resolution)

```bash
# High-priority fixes (can launch together):
Task T019: Fix unreachable print in map_config.py
Task T020: Verify test result in TestObstacleForce.py
Task T022: Extract magic numbers in map_osm_converter.py
Task T024: Clarify ortho_vec documentation in forces.py
Task T025: Verify polygon closing in map_loader_svg.py

# Medium-priority fixes (can launch together):
Task T026: Fix CI badge
Task T027: Remove redundant check
Task T028: Fix copy-paste docstring
Task T029: Fill migration metadata
Task T030: Update GitHub Actions versions
Task T031-T036: Fix all 6 spelling errors

# Low-priority fixes (can launch together):
Task T037: Remove commented code
Task T038: Fix unused loop variables
Task T039: Remove duplicate assignment
Task T040: Improve import style
Task T041: Replace wildcard import
Task T042: Fix markdown indentation

# Total: 23 tasks can run in parallel within US2!
```

---

## Parallel Example: User Story 1 (Test Integration)

```bash
# Configuration tasks are sequential (same file):
Task T008: Update pytest config in pyproject.toml
Task T009: Update coverage config in pyproject.toml

# Verification tasks can run in parallel (different test invocations):
Task T010: Test unified execution
Task T011: Test selective robot_sf only
Task T012: Test fast-pysf only
Task T013: Test headless mode
Task T014: Test parallel execution

# Documentation tasks can run in parallel (different files):
Task T016: Update docs/dev_guide.md
Task T017: Update README.md
Task T018: Create fast-pysf/tests/test_maps/README.md
```

---

## Task Summary

**Total Tasks**: 72
- **Setup**: 3 tasks (1 hour)
- **Foundational**: 4 tasks (2 hours)
- **User Story 1 (P1)**: 11 tasks (4 hours)
- **User Story 2 (P1)**: 26 tasks (7-11 hours)
- **User Story 3 (P2)**: 10 tasks (3-4 hours)
- **User Story 4 (P3)**: 11 tasks (4-6 hours)
- **Polish**: 7 tasks (2-3 hours)

**Total Estimated Effort**: 23-32 hours

**Parallel Opportunities**: 45 tasks marked [P] (62.5% parallelizable)

**Critical Path** (sequential dependencies):
1. Setup → Foundational (3 hours)
2. US1 Configuration (1 hour)
3. US3 Configuration (1 hour)
4. US4 Baseline → Annotation → Verification (4 hours)

**Minimum Time** (with unlimited parallelism): ~9 hours (critical path only)
**Realistic Time** (1-2 developers): 18-25 hours (4-5 working days)

**MVP Scope** (User Story 1 only):
- Tasks: T001-T018 (18 tasks)
- Effort: 7 hours
- Deliverable: Unified test execution working

**Success Criteria Coverage**:
- SC-001 ✅ (US1: T010)
- SC-002 ✅ (US1: T010, T070)
- SC-003 ✅ (Foundational: T007)
- SC-004 ✅ (US2: T043)
- SC-005 ✅ (US2: T019-T025)
- SC-006 ✅ (US3: T046)
- SC-007 ✅ (US4: T062)
- SC-008 ✅ (US3: T051, T070)
- SC-009 ✅ (US1: T016-T018, Polish: T066-T068)
- SC-010 ✅ (US3: T052)
- SC-011 ✅ (Foundational: T003, US3: T054)
- SC-012 ✅ (Polish: T072)
- SC-013 ✅ (US3: T050)

---

## Implementation Strategy

**Approach**: Incremental delivery with independent testing per user story

**Phase 1 (MVP)**: Focus on User Story 1 only
- Delivers immediate value: unified test execution
- Independently testable: `uv run pytest` works
- Low risk: configuration changes only
- Quick win: ~7 hours to completion

**Phase 2 (Quality)**: Address User Story 2 in parallel with US1 deployment
- Can start before US1 is fully documented
- Independent of test integration
- High parallel potential: 23 tasks marked [P]

**Phase 3 (Standards)**: Build on US1, add quality tooling
- Requires test integration foundation
- Natural extension of existing quality gates
- Medium effort: ~3-4 hours

**Phase 4 (Enhancement)**: Optional type annotation improvements
- Lowest priority: P3
- Can be deferred or done incrementally
- Enhances developer experience but not critical

**Deployment Gates**:
- After US1: Unified tests must pass (SC-001, SC-002, SC-003)
- After US2: All high-priority review comments resolved (SC-004, SC-005)
- After US3: Quality gates unified and fast (SC-006, SC-008, SC-013)
- After US4: Type errors reduced ≥25% (SC-007)

**Rollback Strategy**:
- US1: Revert pyproject.toml changes
- US2: Review comments tracked, partial resolution acceptable
- US3: Per-file ignores allow gradual adoption
- US4: Type annotations are additive, can be reverted per-file
