# Feature Specification: Improve fast-pysf Integration

**Feature Branch**: `148-improve-fast-pysf`  
**Created**: October 29, 2025  
**Status**: Draft  
**Input**: User description: "Improve the fast-pysf integration. Streamline the setup process and the dependency management for fast-pysf. Consider what can be integrated in robot-sf from fast-pysf. Before changing anything, make sure the tests run. 1) Add the fast-pysf tests to the pytest configuration. Make sure they run properly. 2) Create a list of all review comments from pr 236. Create an Implementation plan and finally resolve the plan. 3) Activate ruff, ty and test coverage for fast-pysf. 4) Fix as many annotations as possible while ensuring the code runs as intended."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Seamless Test Execution (Priority: P1)

As a developer working on robot_sf_ll7, I want all tests (including fast-pysf tests) to run with a single pytest command, so that I can validate the entire codebase without manual intervention.

**Why this priority**: This is foundational for developer workflow and CI/CD reliability. Without unified test execution, developers might miss integration issues and the CI pipeline remains fragmented.

**Independent Test**: Run `uv run pytest` from the repository root and observe that both robot_sf tests (`tests/`) and fast-pysf tests (`fast-pysf/tests/`) execute successfully with proper reporting.

**Acceptance Scenarios**:

1. **Given** a fresh clone of robot_sf_ll7, **When** developer runs `uv run pytest`, **Then** all tests from both `tests/` and `fast-pysf/tests/` execute and pass
2. **Given** existing pytest configuration, **When** fast-pysf tests are added to config, **Then** test discovery includes both test suites without conflicts
3. **Given** headless CI environment, **When** tests run, **Then** fast-pysf GUI-dependent tests are properly skipped or run headless

---

### User Story 2 - Resolved Code Quality Issues (Priority: P1)

As a code reviewer, I want all PR #236 review comments addressed systematically, so that the codebase meets quality standards before merging.

**Why this priority**: Technical debt from unresolved review comments compounds over time and blocks PR approval. Addressing these ensures code maintainability and correctness.

**Independent Test**: Run through the implementation checklist where each of the 24 review comments has a corresponding fix or documented decision, verified by re-running quality gates (ruff, pylint, ty).

**Acceptance Scenarios**:

1. **Given** 24 PR review comments, **When** implementation plan is created, **Then** each comment is categorized (critical/high/medium/low) with a specific action item
2. **Given** high-priority issues (unreachable code, TODOs, magic numbers), **When** fixes are applied, **Then** code behaves identically but with improved clarity and correctness
3. **Given** spelling errors and formatting issues, **When** corrections are made, **Then** linters and formatters pass without warnings
4. **Given** upstream issues in fast-pysf/, **When** fixes are attempted, **Then** changes are either applied locally or documented for upstream contribution

---

### User Story 3 - Unified Code Quality Standards (Priority: P2)

As a maintainer, I want fast-pysf code to follow the same quality standards as robot_sf code (ruff, ty, coverage), so that the entire codebase is consistent and maintainable.

**Why this priority**: Consistent tooling reduces cognitive load for developers and ensures uniform code quality. This is important but can follow after basic test integration.

**Independent Test**: Run quality gates (`ruff check fast-pysf/`, `ty check fast-pysf/`, `pytest --cov=fast-pysf fast-pysf/tests/`) and observe that they execute without errors and generate coverage reports.

**Acceptance Scenarios**:

1. **Given** ruff configuration, **When** extended to fast-pysf/, **Then** linting and formatting apply to subtree code
2. **Given** type checking setup, **When** ty runs on fast-pysf/, **Then** type errors are reported (or baseline is established)
3. **Given** coverage configuration, **When** fast-pysf tests run, **Then** coverage data is collected and reported separately from main robot_sf coverage

---

### User Story 4 - Improved Type Annotations (Priority: P3)

As a developer using IDE autocomplete, I want improved type annotations in fast-pysf code, so that I get better IntelliSense and catch type errors earlier.

**Why this priority**: Type safety improves developer experience but is lower priority than functional correctness. This can be incremental work after core issues are resolved.

**Independent Test**: Run `ty check fast-pysf/` and verify that type annotation coverage improves (measured by reduction in `Any` types and missing annotations) while all tests still pass.

**Acceptance Scenarios**:

1. **Given** fast-pysf code with missing type hints, **When** annotations are added, **Then** ty reports fewer issues
2. **Given** code with `Any` types, **When** more specific types are added, **Then** type safety improves without breaking functionality
3. **Given** numba-decorated functions, **When** annotations are added, **Then** JIT compilation still works correctly

---

### Edge Cases

- What happens when fast-pysf tests fail but robot_sf tests pass?
  - Test suite should report failure with clear indication of which subset failed
- How does system handle dependencies unique to fast-pysf (pygame, matplotlib)?
  - These should be optional extras or clearly documented as required for full test suite
- What if upstream fast-pysf changes conflict with local fixes?
  - Document subtree merge strategy and maintain changelog of local modifications
- How to handle headless test environments for GUI-dependent fast-pysf tests?
  - Use environment variables (`DISPLAY=`, `SDL_VIDEODRIVER=dummy`) to run headless

## Requirements *(mandatory)*

### Functional Requirements

**Test Integration**
- **FR-001**: System MUST execute fast-pysf tests when running `uv run pytest` from repository root
- **FR-002**: System MUST report fast-pysf test results separately from robot_sf test results for debugging clarity
- **FR-003**: System MUST support running fast-pysf tests in headless mode for CI environments
- **FR-004**: System MUST preserve existing test execution behavior for robot_sf tests (`tests/`, `test_pygame/`)

**PR Review Resolution**
- **FR-005**: System MUST address all high-priority review comments (unreachable code, TODO verifications, magic numbers, documentation uncertainties)
- **FR-006**: System MUST fix all spelling errors and formatting issues in review comments
- **FR-007**: System MUST document decisions for review comments that are deferred or require upstream coordination
- **FR-008**: System MUST create an implementation plan categorizing all 24 review comments by priority and action

**Code Quality Tooling**
- **FR-009**: System MUST run ruff linting and formatting on fast-pysf/ directory
- **FR-010**: System MUST run type checking (ty) on fast-pysf/ directory
- **FR-011**: System MUST collect test coverage for fast-pysf tests
- **FR-012**: System MUST integrate fast-pysf quality checks into existing quality gate workflow

**Type Annotations**
- **FR-013**: System MUST add type annotations to public APIs in fast-pysf code without breaking functionality
- **FR-014**: System MUST preserve numba JIT compilation compatibility when adding type hints
- **FR-015**: System MUST reduce use of `Any` type in favor of specific type annotations

**Dependency Management**
- **FR-016**: System MUST document which fast-pysf dependencies are optional vs required
- **FR-017**: System MUST integrate fast-pysf dependencies into main `pyproject.toml` or document separate management strategy
- **FR-018**: System MUST ensure fast-pysf tests can run with minimal dependencies (core only)

**Integration Points**
- **FR-019**: System MUST document which fast-pysf modules are directly imported by robot_sf code
- **FR-020**: System MUST identify opportunities to consolidate duplicated functionality between robot_sf and fast-pysf
- **FR-021**: System MUST maintain FastPysfWrapper as the primary integration point

### Key Entities

- **TestSuite**: Collection of tests with metadata (location, dependencies, execution mode)
  - Attributes: path, test count, coverage percentage, runtime
  - Types: robot_sf tests, fast-pysf tests, GUI tests

- **ReviewComment**: Individual feedback item from PR #236
  - Attributes: id, author, priority (critical/high/medium/low), file path, line number, category (code quality, documentation, functionality)
  - Status: open, in-progress, resolved, deferred

- **QualityCheck**: Automated code quality validation
  - Attributes: tool (ruff, ty, pylint, coverage), target directory, pass/fail status, output
  - Types: linting, type checking, test coverage

- **DependencySpec**: Package requirement definition
  - Attributes: name, version constraint, optional flag, source (robot_sf vs fast-pysf)
  - Categories: core, dev, optional extras

## Success Criteria *(mandatory)*

### Measurable Outcomes

**Test Execution**
- **SC-001**: Running `uv run pytest` executes all fast-pysf tests in addition to existing robot_sf tests (verified by test count increase from ~43 to 60+)
- **SC-002**: Fast-pysf tests complete in under 60 seconds in headless mode
- **SC-003**: Zero test failures after integration (all pre-existing failures documented and tracked separately)

**Code Quality**
- **SC-004**: All 24 PR #236 review comments have documented resolution status (100% tracking)
- **SC-005**: Zero high-priority review comment issues remain open after implementation
- **SC-006**: Ruff linting passes for fast-pysf/ with zero errors (warnings documented)
- **SC-007**: Type checking baseline established for fast-pysf/ with measurable improvement (reduce undefined types by 25%)

**Developer Experience**
- **SC-008**: Single quality gate command runs all checks (robot_sf + fast-pysf) in under 5 minutes
- **SC-009**: Documentation updated with clear instructions for running fast-pysf tests
- **SC-010**: CI pipeline includes fast-pysf quality checks without increasing total runtime beyond 10%

**Integration**
- **SC-011**: Dependency list consolidated or clearly documented for both robot_sf and fast-pysf
- **SC-012**: At least 3 integration opportunities identified and documented for future work
- **SC-013**: Fast-pysf test coverage data available in coverage reports (minimum 70% coverage)
