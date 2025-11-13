# Tasks: Code Coverage Monitoring and Quality Tracking

**Input**: Design documents from `/specs/145-add-pytest-cov/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Smoke tests and validation included for each user story to verify functionality

**Organization**: Tasks are grouped by user story (P1→P4) to enable independent implementation and testing

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and dependency setup for coverage tooling

- [X] T001 [P] Add pytest-cov dependency to pyproject.toml dependencies list
- [X] T002 [P] Create robot_sf/coverage_tools/ directory structure with __init__.py
- [X] T003 [P] Create scripts/coverage/ directory for CLI entry points
- [X] T004 [P] Create tests/coverage_tools/ directory for coverage tool tests
- [X] T005 Update .gitignore with coverage output directories (htmlcov/, .coverage, coverage.json, coverage.xml)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core configuration and library infrastructure that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T006 Configure coverage collection in pyproject.toml [tool.coverage.run] section (source, omit, parallel)
- [X] T007 Configure coverage reporting in pyproject.toml [tool.coverage.report] section (precision, show_missing, exclude_lines)
- [X] T008 Configure coverage output formats in pyproject.toml [tool.coverage.html] and [tool.coverage.json]
- [X] T009 Add coverage options to pyproject.toml [tool.pytest.ini_options] for automatic collection
- [X] T010 [P] Implement robot_sf/coverage_tools/report_formatter.py with terminal/JSON/markdown formatters
- [X] T011 Add coverage fixtures to tests/conftest.py for test isolation

**Checkpoint**: Foundation ready - coverage collection configured, formatters available, user stories can proceed

---

## Phase 3: User Story 1 - Run Tests with Coverage Information (Priority: P1) 🎯 MVP

**Goal**: Developers run standard test commands and automatically see coverage data without workflow changes

**Independent Test**: Run `uv run pytest tests` and verify coverage summary appears in terminal with overall % and per-module breakdown, HTML report generated in htmlcov/

### Validation for User Story 1

- [X] T012 [US1] Create validation script scripts/validation/test_coverage_collection.sh to verify coverage runs
- [X] T013 [US1] Add smoke test in tests/coverage_tools/test_smoke_coverage.py to verify coverage collection doesn't break tests

### Implementation for User Story 1

- [X] T014 [P] [US1] Verify pytest addopts in pyproject.toml includes --cov=robot_sf --cov-report=term-missing --cov-report=html --cov-report=json
- [X] T015 [P] [US1] Update .vscode/tasks.json with "Run Tests with Coverage" task using standard pytest command
- [X] T016 [P] [US1] Update .vscode/tasks.json with "Open Coverage Report" task using cross-platform script (scripts/coverage/open_coverage_report.py)
- [X] T017 [US1] Test coverage collection with parallel execution (pytest -n auto) and verify .coverage files merge correctly
- [X] T018 [US1] Update docs/dev_guide.md with coverage workflow section explaining automatic collection
- [X] T019 [US1] Verify coverage data excludes test files, examples, scripts per omit configuration

**Checkpoint**: P1 Complete - Developers can run tests and see coverage automatically. This is the MVP!

---

## Phase 4: User Story 2 - Receive Coverage Warnings in CI/CD (Priority: P2)

**Goal**: CI/CD pipeline compares coverage against baseline and generates non-blocking warnings on decreases

**Independent Test**: Create a PR that reduces coverage in a test file, verify CI passes with warning annotations identifying the decrease

### Implementation for User Story 2

- [X] T020 [P] [US2] Implement robot_sf/coverage_tools/baseline_comparator.py with load_baseline(), compare(), generate_warning() methods
- [X] T021 [P] [US2] Create scripts/coverage/compare_coverage.py CLI with --current, --baseline, --threshold, --format arguments
- [X] T022 [US2] Add CoverageSnapshot, CoverageBaseline, CoverageDelta entity classes to robot_sf/coverage_tools/baseline_comparator.py
- [X] T023 [US2] Implement GitHub Actions annotation format output in baseline_comparator.py
- [X] T024 [US2] Implement terminal format output for local baseline comparisons
- [X] T025 [US2] Implement JSON format output for programmatic consumption
- [X] T026 [US2] Add coverage job to .github/workflows/ci.yml after existing test job
- [X] T027 [US2] Configure GitHub Actions cache in ci.yml for coverage/.coverage-baseline.json with key coverage-baseline-${{ github.base_ref }}
- [X] T028 [US2] Add compare coverage step in ci.yml that runs scripts/coverage/compare_coverage.py with continue-on-error: true
- [X] T029 [US2] Add baseline update step in ci.yml for main branch pushes to copy coverage.json to coverage/.coverage-baseline.json
- [X] T030 [US2] Create tests/coverage_tools/test_baseline_comparator.py with unit tests for comparison logic
- [ ] T031 [US2] Test CI integration by creating intentional coverage decrease in test PR
- [X] T032 [US2] Update docs/coverage_guide.md (new file) with CI integration section and warning interpretation

**Checkpoint**: P2 Complete - CI warns on coverage decreases without failing builds

---

## Phase 5: User Story 3 - Identify Coverage Gaps Automatically (Priority: P3)

**Goal**: Developers can run a command to see ranked list of coverage gaps with actionable recommendations

**Independent Test**: Run `uv run python scripts/coverage/analyze_gaps.py` and verify output shows top 10 gaps ranked by priority score with file paths, coverage %, and recommendations

### Implementation for User Story 3

- [ ] T033 [P] [US3] Implement robot_sf/coverage_tools/gap_analyzer.py with load_coverage(), identify_gaps(), generate_report() methods
- [ ] T034 [P] [US3] Add CoverageGap, GapAnalysisReport, GapAnalysisConfig entity classes to gap_analyzer.py
- [ ] T035 [US3] Implement gap priority scoring algorithm (uncovered_lines * location_weight + complexity_adjustment)
- [ ] T036 [US3] Configure location_weights dict in gap_analyzer.py (gym_env: 1.5, sim: 1.5, nav: 1.4, benchmark: 1.3, render: 1.2)
- [ ] T037 [US3] Implement terminal format output in report_formatter.py for gap analysis
- [ ] T038 [US3] Implement JSON format output in report_formatter.py for gap analysis
- [ ] T039 [US3] Implement markdown format output in report_formatter.py for gap analysis
- [ ] T040 [P] [US3] Create scripts/coverage/analyze_gaps.py CLI with --coverage-file, --top-n, --min-lines, --output-format, --module-filter arguments
- [ ] T041 [US3] Add recommendation generation logic to gap_analyzer.py based on file path patterns
- [ ] T042 [P] [US3] Create tests/coverage_tools/test_gap_analyzer.py with unit tests for gap identification and scoring
- [ ] T043 [P] [US3] Create tests/coverage_tools/test_report_formatter.py with tests for all output formats
- [ ] T044 [US3] Add "Coverage Gap Analysis" task to .vscode/tasks.json running analyze_gaps.py
- [ ] T045 [US3] Update docs/coverage_guide.md with gap analysis usage section and interpretation guide
- [ ] T046 [US3] Add gap analysis example to quickstart.md with sample output

**Checkpoint**: P3 Complete - Developers can identify and prioritize coverage improvements

---

## Phase 6: User Story 4 - Track Quality Metrics Over Time (Priority: P4)

**Goal**: Maintainers can view historical coverage trends and understand quality evolution over time

**Independent Test**: Run coverage collection multiple times, run `uv run python scripts/coverage/track_trends.py report`, verify output shows trend statistics and direction

### Implementation for User Story 4

- [ ] T047 [P] [US4] Implement robot_sf/coverage_tools/trend_tracker.py with load_trend(), append_snapshot(), apply_retention(), calculate_statistics() methods
- [ ] T048 [P] [US4] Add CoverageTrend, TrendConfig, TrendVisualization entity classes to trend_tracker.py
- [ ] T049 [US4] Implement JSONL storage format with one CoverageSnapshot per line in trend_tracker.py
- [ ] T050 [US4] Implement retention policy logic (prune snapshots older than retention_commits or retention_days)
- [ ] T051 [US4] Implement trend direction calculation (improving/stable/degrading based on linear regression)
- [ ] T052 [US4] Implement trend rate calculation (percentage points per week)
- [ ] T053 [P] [US4] Create scripts/coverage/track_trends.py CLI with collect, visualize, report subcommands
- [ ] T054 [US4] Implement collect subcommand with --coverage-file, --history-file, --branch, --commit arguments
- [ ] T055 [US4] Implement visualize subcommand with matplotlib line graph generation (--output-file, --plot-type, --time-range, --modules)
- [ ] T056 [US4] Implement report subcommand with --history-file, --output-format arguments
- [ ] T057 [US4] Add terminal format trend report output (trend direction, rate, date range)
- [ ] T058 [US4] Add JSON format trend report output (full statistics)
- [ ] T059 [US4] Add markdown format trend report output (table format)
- [ ] T060 [P] [US4] Create tests/coverage_tools/test_trend_tracker.py with unit tests for JSONL operations and retention
- [ ] T061 [US4] Add trend collection to .github/workflows/ci.yml for main branch (after test success)
- [ ] T062 [US4] Configure trend storage path in ci.yml (results/coverage_history.jsonl as artifact)
- [ ] T063 [US4] Update docs/coverage_guide.md with trend tracking section and visualization examples
- [ ] T064 [US4] Add trend tracking example to quickstart.md

**Checkpoint**: P4 Complete - All user stories implemented, full coverage monitoring suite available

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, testing, and final integration across all user stories

- [ ] T065 [P] Create comprehensive docs/coverage_guide.md with all features, troubleshooting, and best practices
- [ ] T066 [P] Update docs/README.md to link to coverage_guide.md in appropriate section
- [ ] T067 [P] Create examples/coverage_example.py demonstrating programmatic use of coverage_tools modules
- [ ] T068 Run all validation scripts to ensure coverage collection, CI integration, gap analysis, and trends work end-to-end
- [ ] T069 [P] Add CHANGELOG.md entry for pytest-cov integration feature
- [ ] T070 [P] Update pyproject.toml project version if needed (following semantic versioning)
- [ ] T071 Verify all constitutional principles still satisfied (run through constitution checklist)
- [ ] T072 Run quality gates: Install Dependencies → Ruff: Format and Fix → Check Code Quality → Type Check → Run Tests
- [ ] T073 Final smoke test: Complete workflow from test run → gap analysis → trend collection → CI warning simulation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion (T001-T005) - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational (T006-T011) - Can start after foundation ready
- **User Story 2 (Phase 4)**: Depends on Foundational - Can start in parallel with US1 (uses same coverage.json output)
- **User Story 3 (Phase 5)**: Depends on Foundational - Can start in parallel with US1/US2 (uses same coverage.json output)
- **User Story 4 (Phase 6)**: Depends on Foundational - Can start in parallel with US1/US2/US3
- **Polish (Phase 7)**: Depends on desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Independent - only depends on Foundational phase
- **User Story 2 (P2)**: Independent - does not depend on US1, can run in parallel
- **User Story 3 (P3)**: Independent - does not depend on US1/US2, can run in parallel
- **User Story 4 (P4)**: Independent - does not depend on US1/US2/US3, can run in parallel

### Within Each User Story

- US1: Configuration → VS Code tasks → Validation → Documentation
- US2: Library modules [P] → CLI script → CI workflow → Tests [P] → Documentation
- US3: Library modules [P] → CLI script → Tests [P] → VS Code task → Documentation
- US4: Library modules [P] → CLI script → Visualization → CI integration → Tests [P] → Documentation

### Parallel Opportunities

**Setup Phase (all can run in parallel)**:
- T001, T002, T003, T004, T005 (5 tasks)

**Foundational Phase (configuration tasks can run in parallel)**:
- T006-T009 (configuration files) can run in parallel
- T010 (report formatter) can run in parallel with T006-T009
- T011 (test fixtures) depends on nothing, can be parallel

**User Story 2**:
- T020, T021, T022 (library modules and CLI) can run in parallel
- T023-T025 (output formats) can run in parallel
- T030 (tests) can run in parallel with implementation

**User Story 3**:
- T033, T034, T040 (library module and CLI) can run in parallel
- T037-T039 (output formats) can run in parallel
- T042, T043 (tests) can run in parallel with implementation

**User Story 4**:
- T047, T048, T053 (library module and CLI) can run in parallel
- T057-T059 (output formats) can run in parallel
- T060 (tests) can run in parallel with implementation

**Polish Phase**:
- T065, T066, T067, T069, T070 (documentation and metadata) can run in parallel

---

## Parallel Example: User Story 2 (CI Integration)

```bash
# Launch library modules in parallel:
Task T020: "Implement robot_sf/coverage_tools/baseline_comparator.py"
Task T021: "Create scripts/coverage/compare_coverage.py CLI"
Task T022: "Add CoverageSnapshot, CoverageBaseline, CoverageDelta entities"

# Then output formats in parallel:
Task T023: "Implement GitHub Actions annotation format"
Task T024: "Implement terminal format output"
Task T025: "Implement JSON format output"

# Tests can run in parallel with implementation:
Task T030: "Create tests/coverage_tools/test_baseline_comparator.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

**Why**: Delivers immediate value with minimal scope - developers see coverage automatically

1. Complete Phase 1: Setup (T001-T005) - ~30 minutes
2. Complete Phase 2: Foundational (T006-T011) - ~1-2 hours
3. Complete Phase 3: User Story 1 (T012-T019) - ~2-3 hours
4. **STOP and VALIDATE**: Run tests, verify coverage appears, check HTML report
5. **Demo**: Show developers coverage working automatically
6. Estimated total: **~4-6 hours for working MVP**

### Incremental Delivery (Recommended)

**Why**: Each user story adds value independently, can be deployed/tested separately

1. **MVP Release**: US1 only (automatic coverage collection)
   - Developers start seeing coverage immediately
   - Value: Visibility into test coverage
2. **V2 Release**: Add US2 (CI warnings)
   - PRs now get coverage feedback
   - Value: Prevent coverage regression
3. **V3 Release**: Add US3 (gap analysis)
   - Developers can identify improvement areas
   - Value: Prioritized testing guidance
4. **V4 Release**: Add US4 (trend tracking)
   - Maintainers see long-term quality trends
   - Value: Project health monitoring

### Parallel Team Strategy

With 2-3 developers after foundation completes:

1. **Team completes Setup + Foundational together** (T001-T011)
2. **Then split**:
   - Developer A: User Story 1 (T012-T019) - 2-3 hours
   - Developer B: User Story 2 (T020-T032) - 4-5 hours
   - Developer C: User Story 3 (T033-T046) - 4-5 hours
3. **Later (after US1-3)**:
   - Developer D: User Story 4 (T047-T064) - 5-6 hours
4. **Finally together**: Polish phase (T065-T073)

**Total parallel time**: ~1-2 days with 3 developers vs ~3-4 days sequential

---

## Task Count Summary

- **Setup**: 5 tasks
- **Foundational**: 6 tasks (CRITICAL - blocks all stories)
- **User Story 1 (P1)**: 8 tasks (MVP)
- **User Story 2 (P2)**: 13 tasks
- **User Story 3 (P3)**: 14 tasks
- **User Story 4 (P4)**: 18 tasks
- **Polish**: 9 tasks
- **Total**: 73 tasks

### Parallel Opportunities

- Setup: 5 parallel tasks
- Foundational: 4-5 parallel tasks
- US1: 3 parallel tasks
- US2: 7 parallel tasks
- US3: 6 parallel tasks
- US4: 7 parallel tasks
- Polish: 5 parallel tasks
- **Total parallel opportunities**: ~37 tasks (~50%)

### Suggested MVP Scope

**Minimum Viable Product**: Phase 1 + Phase 2 + Phase 3 (User Story 1 only)
- **Tasks**: T001-T019 (19 tasks)
- **Time estimate**: 4-6 hours
- **Value delivered**: Automatic coverage collection and reporting
- **Independent test**: Run `uv run pytest tests`, see coverage summary, open htmlcov/index.html

---

## Notes

- [P] tasks = different files, no dependencies, can run in parallel
- [Story] label (US1, US2, US3, US4) maps task to specific user story for traceability
- Each user story is independently completable and testable
- Commit after each task or logical group of related tasks
- Stop at any checkpoint to validate story independently
- All library code in robot_sf/coverage_tools/ follows library-first principle (Principle XI)
- All library code uses Loguru for logging (Principle XII); CLI scripts may use print()
- Configuration centralized in pyproject.toml (Principle IV)
- Documentation required for all new public surfaces (Principle VIII)
- Constitutional compliance verified in Polish phase (T071)
