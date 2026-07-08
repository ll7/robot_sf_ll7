# Data Model: fast-pysf Integration Quality Improvements

**Feature**: 148-improve-fast-pysf  
**Date**: 2025-10-29  
**Phase**: 1 (Design & Contracts)

## Overview

This document defines the key entities, relationships, and state transitions for the fast-pysf integration quality improvements feature. The data model focuses on test infrastructure, quality check results, and review comment tracking.

## Entity Definitions

### 1. TestSuite

**Purpose**: Represents a collection of test cases with execution metadata and results.

**Attributes**:
- `name`: string (e.g., "robot_sf", "fast-pysf")
- `path`: string (absolute path to test directory)
- `test_count`: integer (number of collected tests)
- `passed`: integer (number of passing tests)
- `failed`: integer (number of failing tests)
- `skipped`: integer (number of skipped tests)
- `duration`: float (total execution time in seconds)
- `coverage_percent`: float | null (code coverage percentage, if measured)
- `status`: enum ["not_run", "running", "passed", "failed", "error"]

**Relationships**:
- Has many `TestCase` entities
- Belongs to `TestConfiguration`
- Produces `TestReport`

**Validation Rules**:
- `test_count` = `passed` + `failed` + `skipped`
- `duration` >= 0
- `coverage_percent` in range [0.0, 100.0] or null
- `path` must exist and be readable

**State Transitions**:
```
not_run → running → [passed | failed | error]
         ↓
      skipped (if pytest collection phase fails)
```

---

### 2. TestCase

**Purpose**: Individual test function with execution result and diagnostic information.

**Attributes**:
- `id`: string (pytest node ID, e.g., "fast-pysf/tests/test_forces.py::test_desired_force")
- `name`: string (test function name)
- `file_path`: string (relative path from repo root)
- `line_number`: integer (test definition line)
- `suite`: string (parent suite name: "robot_sf" or "fast-pysf")
- `status`: enum ["passed", "failed", "skipped", "xfailed", "xpassed"]
- `duration`: float (execution time in seconds)
- `failure_message`: string | null (assertion error or exception message)
- `failure_traceback`: string | null (full traceback if failed)
- `markers`: list[string] (pytest markers, e.g., ["slow", "integration"])

**Relationships**:
- Belongs to `TestSuite`
- May reference `TestFixture` (map files, config objects)

**Validation Rules**:
- `id` must be unique within test run
- `duration` >= 0
- `failure_message` and `failure_traceback` non-null only if status = "failed"
- `file_path` must exist

**State Transitions**:
```
collected → [passed | failed | skipped | xfailed | xpassed]
```

---

### 3. TestConfiguration

**Purpose**: Pytest and coverage configuration settings affecting test execution.

**Attributes**:
- `testpaths`: list[string] (directories to search for tests)
- `python_files`: list[string] (test file patterns, default: ["test_*.py"])
- `python_classes`: list[string] (test class patterns, default: ["Test*"])
- `python_functions`: list[string] (test function patterns, default: ["test_*"])
- `addopts`: list[string] (additional pytest options)
- `coverage_source`: list[string] (packages to measure coverage)
- `coverage_omit`: list[string] (patterns to exclude from coverage)
- `min_coverage`: float (minimum coverage threshold, e.g., 70.0)

**Relationships**:
- Configures `TestSuite` execution
- Defined in `pyproject.toml`

**Validation Rules**:
- `testpaths` must contain at least one valid directory
- `min_coverage` in range [0.0, 100.0]
- All paths relative to repository root

**Example**:
```toml
[tool.pytest.ini_options]
testpaths = ["tests", "fast-pysf/tests"]
addopts = ["-v", "--cov=robot_sf", "--cov=fast-pysf/pysocialforce"]

[tool.coverage.run]
source = ["robot_sf", "fast-pysf/pysocialforce"]
omit = ["*/tests/*", "*/examples/*"]
```

---

### 4. QualityCheck

**Purpose**: Result of automated code quality analysis (ruff, ty, coverage).

**Attributes**:
- `tool`: enum ["ruff_lint", "ruff_format", "type_check", "coverage"]
- `target`: string (file path or directory checked)
- `status`: enum ["passed", "failed", "warning", "skipped"]
- `issue_count`: integer (number of issues found)
- `issues`: list[QualityIssue] (detailed issue records)
- `execution_time`: float (check duration in seconds)
- `timestamp`: datetime (when check was run)

**Relationships**:
- Contains many `QualityIssue` entities
- Aggregated in `QualityReport`

**Validation Rules**:
- `issue_count` = len(`issues`)
- `status` = "passed" if `issue_count` == 0, else "failed" or "warning"
- `execution_time` >= 0

**State Transitions**:
```
not_run → running → [passed | failed | warning]
```

---

### 5. QualityIssue

**Purpose**: Specific code quality violation detected by tooling.

**Attributes**:
- `tool`: enum ["ruff", "ty", "pylint"]
- `file_path`: string (relative path from repo root)
- `line_number`: integer | null
- `column_number`: integer | null
- `rule_code`: string (e.g., "F401", "C901", "type-error")
- `rule_name`: string (human-readable rule name)
- `severity`: enum ["error", "warning", "info"]
- `message`: string (issue description)
- `suggestion`: string | null (auto-fix suggestion if available)
- `fixable`: boolean (can be auto-fixed)

**Relationships**:
- Belongs to `QualityCheck`
- May reference `ReviewComment` if related to PR feedback

**Validation Rules**:
- `file_path` must exist
- `line_number` > 0 if not null
- `severity` = "error" implies `fixable` may be false
- `suggestion` non-null only if `fixable` = true

**Example**:
```json
{
  "tool": "ruff",
  "file_path": "fast-pysf/pysocialforce/forces.py",
  "line_number": 397,
  "rule_code": "D400",
  "rule_name": "first-line-should-end-with-period",
  "severity": "warning",
  "message": "First line should end with a period",
  "suggestion": "Add period to docstring first line",
  "fixable": true
}
```

---

### 6. ReviewComment

**Purpose**: GitHub PR review comment tracking and resolution status.

**Attributes**:
- `comment_id`: string (GitHub comment ID or sequential number)
- `pr_number`: integer (GitHub PR number, e.g., 236)
- `author`: string (reviewer username)
- `file_path`: string (file referenced in comment)
- `line_number`: integer | null (specific line if applicable)
- `category`: enum ["functionality", "maintainability", "quality", "documentation"]
- `priority`: enum ["high", "medium", "low"]
- `description`: string (comment text)
- `status`: enum ["open", "in_progress", "resolved", "wont_fix", "upstream"]
- `resolution_note`: string | null (how it was resolved)
- `related_commit`: string | null (commit SHA resolving the comment)
- `effort_estimate`: float (hours estimated to resolve)
- `actual_effort`: float | null (hours actually spent)

**Relationships**:
- May relate to `QualityIssue` (if comment flagged code quality problem)
- Tracked in `ReviewCommentRegistry`

**Validation Rules**:
- `pr_number` > 0
- `line_number` > 0 if not null
- `status` = "resolved" implies `resolution_note` and `related_commit` non-null
- `actual_effort` <= `effort_estimate` * 2 (flag if significantly over-budget)

**State Transitions**:
```
open → in_progress → [resolved | wont_fix | upstream]
  ↓
(can reopen if regression detected)
```

---

### 7. TestFixture

**Purpose**: Test data files and configuration objects used by test cases.

**Attributes**:
- `name`: string (fixture identifier, e.g., "map_regular")
- `type`: enum ["json_file", "config_object", "mock_data"]
- `file_path`: string | null (path to fixture file if file-based)
- `schema_version`: string | null (for versioned fixtures)
- `content_hash`: string (SHA256 hash for integrity verification)
- `dependencies`: list[string] (other fixtures this depends on)
- `used_by_tests`: list[string] (test IDs using this fixture)

**Relationships**:
- Used by `TestCase` entities
- Stored in `fast-pysf/tests/test_maps/` or `tests/fixtures/`

**Validation Rules**:
- `file_path` must exist if `type` = "json_file"
- `content_hash` must match actual file content
- `schema_version` follows semantic versioning if present

**Example**:
```json
{
  "name": "map_regular",
  "type": "json_file",
  "file_path": "fast-pysf/tests/test_maps/map_regular.json",
  "schema_version": "1.0.0",
  "content_hash": "a3b5c7...",
  "dependencies": [],
  "used_by_tests": [
    "fast-pysf/tests/test_map_loader.py::test_load_map"
  ]
}
```

---

## Entity Relationships Diagram

```
TestConfiguration
    ↓ configures
TestSuite (1:N) ──has──> TestCase (N)
    ↓ produces              ↓ uses
TestReport              TestFixture (N)
    ↓ includes
QualityCheck (N) ──contains──> QualityIssue (N)
    ↓ may relate to
ReviewComment (N)
    ↓ tracked in
ReviewCommentRegistry
```

---

## State Management

### Test Execution Lifecycle

1. **Collection Phase**:
   - Pytest discovers tests based on `TestConfiguration.testpaths`
   - Creates `TestCase` entities for each discovered test
   - Validates `TestFixture` availability

2. **Execution Phase**:
   - `TestSuite.status` transitions to "running"
   - Each `TestCase` executes and records result
   - Coverage measurement runs concurrently

3. **Reporting Phase**:
   - `TestReport` aggregates results
   - Coverage percentage computed
   - `TestSuite.status` set to final state

### Quality Check Lifecycle

1. **Initialization**:
   - Tool configuration loaded from `pyproject.toml`
   - Target files/directories identified

2. **Execution**:
   - `QualityCheck.status` = "running"
   - Tool scans code and generates `QualityIssue` records

3. **Resolution**:
   - Auto-fixable issues resolved via `ruff check --fix`
   - Manual issues tracked and linked to `ReviewComment` if applicable

### Review Comment Resolution Lifecycle

1. **Import** (from PR #236):
   - Parse review comments from GitHub API
   - Categorize and prioritize
   - Create `ReviewComment` entities

2. **Triage**:
   - Assign priority based on impact
   - Estimate effort
   - Set status = "open"

3. **Resolution**:
   - Developer works on fix
   - Status = "in_progress"
   - Commit created
   - Status = "resolved" with `resolution_note`

4. **Validation**:
   - Related `QualityCheck` re-run
   - Ensure no regression in tests

---

## Data Persistence

### Configuration Files

- **pyproject.toml**: Authoritative source for `TestConfiguration` and quality tool settings
- **pr236_review_comments.md**: Human-readable tracking of `ReviewComment` entities
- **pytest cache**: Stores test results for --lf (last failed) and --ff (failed first) modes

### Runtime Artifacts

- **coverage.json**: Coverage data for programmatic access
- **htmlcov/**: HTML coverage report for developer browsing
- **.pytest_cache/**: Test execution metadata
- **.ruff_cache/**: Ruff analysis cache for incremental checking

### Documentation Artifacts

- **research.md**: Decisions and rationale (Phase 0 output)
- **data-model.md**: This file (Phase 1 output)
- **contracts/**: API contracts (if pytest plugins needed)
- **quickstart.md**: Developer onboarding guide

---

## Validation & Integrity

### Test Suite Validation

- Total test count must increase (from ~43 to 55+)
- Zero new test failures introduced
- Fast-pysf test duration < 60 seconds (SC-002)
- Coverage for fast-pysf ≥ 70% (SC-013)

### Quality Check Validation

- Ruff must pass for both robot_sf and fast-pysf
- Type errors reduced by 25% (SC-007)
- No critical security issues (ruff rule S)

### Review Comment Validation

- All 24 comments tracked (SC-004)
- High-priority comments resolved (SC-005)
- Medium/low priority documented in tasks.md

---

## Next Steps

**Phase 1 Continuation**:
1. ✅ Data model defined (this document)
2. ⏳ Generate API contracts in `contracts/` (pytest plugin interface if needed)
3. ⏳ Create `quickstart.md` (developer onboarding)
4. ⏳ Update agent context
5. ⏳ Re-evaluate Constitution Check
