# API Contracts: fast-pysf Integration Quality Improvements

**Feature**: 148-improve-fast-pysf  
**Date**: 2025-10-29  
**Phase**: 1 (Design & Contracts)

## Overview

This document defines the API contracts and interfaces for the fast-pysf integration quality improvements. Since this feature focuses on test infrastructure and quality tooling (not runtime APIs), contracts are primarily configuration-based rather than programmatic interfaces.

## Contract Types

1. **Configuration Contracts**: pyproject.toml schema for pytest, ruff, coverage
2. **Test Execution Contracts**: pytest CLI interface and exit codes
3. **Quality Tool Contracts**: ruff/ty command interfaces and output formats
4. **Fixture Contracts**: Test map JSON schema

---

## 1. pytest Configuration Contract

### Interface

**Location**: `pyproject.toml` → `[tool.pytest.ini_options]`

**Schema**:
```toml
[tool.pytest.ini_options]
# REQUIRED: Test discovery paths (MUST include both suites)
testpaths = ["tests", "fast-pysf/tests"]

# REQUIRED: Test file/function patterns
python_files = ["test_*.py"]
python_functions = ["test_*"]

# OPTIONAL: Additional pytest options
addopts = [
    "-v",                          # Verbose output
    "--tb=short",                  # Short traceback format
    "--cov=robot_sf",              # Coverage for robot_sf
    "--cov=fast-pysf/pysocialforce", # Coverage for fast-pysf
    "--cov-report=html",           # HTML coverage report
    "--cov-report=json",           # JSON coverage data
    "--cov-report=term-missing",   # Terminal report with missing lines
]

# OPTIONAL: Markers for test categorization
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests requiring multiple components",
    "fast_pysf: marks tests from fast-pysf subtree",
]
```

**Contract Guarantees**:
- Running `uv run pytest` MUST execute tests from both `tests/` and `fast-pysf/tests/`
- Running `uv run pytest tests` MUST execute only robot_sf tests
- Running `uv run pytest fast-pysf/tests` MUST execute only fast-pysf tests
- Exit code 0 = all tests passed, exit code 1 = at least one failure

**Breaking Changes**:
- Removing a testpath requires migration guide
- Changing python_files pattern may hide existing tests (BREAKING)

---

## 2. Coverage Configuration Contract

### Interface

**Location**: `pyproject.toml` → `[tool.coverage.run]` and `[tool.coverage.report]`

**Schema**:
```toml
[tool.coverage.run]
# REQUIRED: Source packages to measure
source = [
    "robot_sf",
    "fast-pysf/pysocialforce"  # ADDED in this feature
]

# REQUIRED: Patterns to exclude from coverage
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
    "fast-pysf/examples/*",  # ADDED in this feature
]

# OPTIONAL: Parallel mode for multi-worker testing
parallel = true

[tool.coverage.report]
# OPTIONAL: Precision for percentage display
precision = 2

# OPTIONAL: Exclude lines matching these patterns
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

**Contract Guarantees**:
- Coverage percentage MUST include both robot_sf and fast-pysf code
- HTML report MUST separate packages by directory
- JSON output MUST be parsable with Python's json module
- Coverage ≥ 70% for fast-pysf/pysocialforce (SC-013)

**Output Format** (coverage.json excerpt):
```json
{
  "meta": {
    "version": "7.0.0",
    "timestamp": "2025-10-29T12:00:00",
    "branch_coverage": false
  },
  "files": {
    "fast-pysf/pysocialforce/forces.py": {
      "summary": {
        "covered_lines": 85,
        "num_statements": 120,
        "percent_covered": 70.83,
        "missing_lines": 35,
        "excluded_lines": 0
      }
    }
  },
  "totals": {
    "covered_lines": 9892,
    "num_statements": 10768,
    "percent_covered": 91.87
  }
}
```

---

## 3. Ruff Configuration Contract

### Interface

**Location**: `pyproject.toml` → `[tool.ruff]`

**Schema**:
```toml
[tool.ruff]
# Target Python version
target-version = "py313"
line-length = 100

[tool.ruff.lint]
# Rule categories to enable (see research.md for rationale)
select = [
    "F",    # Pyflakes (errors)
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "B",    # flake8-bugbear
    "S",    # flake8-bandit (security)
    "UP",   # pyupgrade (modernization)
    "I",    # isort (import sorting)
    # Complexity rules enabled but relaxed for fast-pysf
    "C90",  # mccabe complexity
    "PLR",  # pylint refactor
]

# Per-file ignores for gradual adoption
[tool.ruff.lint.per-file-ignores]
"fast-pysf/**/*.py" = [
    "C901",   # Complex function (relax initially)
    "PLR0912", # Too many branches (relax initially)
    "PLR0913", # Too many arguments (relax initially)
    "PLR0915", # Too many statements (relax initially)
]

[tool.ruff.format]
# Formatting settings (applied to all code)
quote-style = "double"
indent-style = "space"
```

**Contract Guarantees**:
- `uv run ruff check .` MUST check both robot_sf/ and fast-pysf/
- `uv run ruff format .` MUST format both codebases
- Security issues (S rule category) MUST NOT be ignored
- Exit code 0 = no violations, exit code 1 = violations found

**Output Format** (ruff check --output-format=json):
```json
[
  {
    "code": "F401",
    "message": "'numpy' imported but unused",
    "location": {
      "row": 10,
      "column": 8
    },
    "filename": "fast-pysf/pysocialforce/forces.py",
    "noqa_row": 10,
    "fix": {
      "message": "Remove unused import: `numpy`",
      "edits": [...]
    }
  }
]
```

---

## 4. Type Checking Contract (ty/pyright)

### Interface

**Command**: `uvx ty check . --exit-zero`

**Schema** (pyproject.toml → [tool.pyright]):
```toml
[tool.pyright]
# Include fast-pysf in type checking
include = ["robot_sf", "fast-pysf/pysocialforce"]

# Exclude test files and examples
exclude = [
    "**/tests",
    "**/test_*.py",
    "**/examples",
]

# Type checking mode (basic = less strict, standard = default)
typeCheckingMode = "basic"

# Report settings
reportMissingImports = true
reportMissingTypeStubs = false  # Many packages lack stubs
reportUnusedImport = true
reportUnusedVariable = true

# numba compatibility
# Note: Functions with @njit may have limited type inference
```

**Contract Guarantees**:
- Type errors MUST be reported for public APIs
- `--exit-zero` flag ALLOWS warnings but reports errors
- Type error count reduction ≥ 25% (SC-007)
- numba-decorated functions MAY have relaxed checking

**Output Format** (JSON excerpt):
```json
{
  "version": "1.1.0",
  "time": "2025-10-29T12:00:00",
  "generalDiagnostics": [
    {
      "file": "fast-pysf/pysocialforce/forces.py",
      "severity": "error",
      "message": "Type of parameter 'config' is partially unknown",
      "range": {
        "start": {"line": 50, "character": 10},
        "end": {"line": 50, "character": 16}
      },
      "rule": "reportUnknownParameterType"
    }
  ],
  "summary": {
    "filesAnalyzed": 250,
    "errorCount": 15,
    "warningCount": 42,
    "informationCount": 0
  }
}
```

---

## 5. Test Fixture Contract (Map JSON Schema)

### Interface

**Location**: `fast-pysf/tests/test_maps/*.json`

**Schema**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "MapDefinition",
  "type": "object",
  "properties": {
    "obstacles": {
      "type": "array",
      "items": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 2,
        "maxItems": 2
      },
      "description": "List of [x, y] obstacle coordinates"
    },
    "routes": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "waypoints": {
            "type": "array",
            "items": {
              "type": "array",
              "items": {"type": "number"},
              "minItems": 2,
              "maxItems": 2
            }
          }
        },
        "required": ["id", "waypoints"]
      }
    },
    "crowded_zones": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "center": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2
          },
          "radius": {"type": "number", "minimum": 0}
        },
        "required": ["center", "radius"]
      }
    }
  },
  "required": []
}
```

**Example Valid Fixture** (`map_regular.json`):
```json
{
  "obstacles": [
    [5.0, 5.0],
    [10.0, 10.0],
    [15.0, 5.0]
  ],
  "routes": [
    {
      "id": "route1",
      "waypoints": [
        [0.0, 0.0],
        [20.0, 20.0]
      ]
    }
  ],
  "crowded_zones": [
    {
      "center": [10.0, 10.0],
      "radius": 5.0
    }
  ]
}
```

**Example Invalid Fixture** (`invalid_json_file.json`):
```json
{
  "obstacles": [
    [1, 2, 3]  // Invalid: array must have exactly 2 elements
  ],
  "routes": "not_an_array"  // Invalid: must be array
}
```

**Contract Guarantees**:
- Map loader MUST parse valid fixtures without error
- Map loader MUST raise JSONDecodeError for malformed JSON
- Map loader MUST validate array dimensions (2D coordinates)
- Missing keys (obstacles, routes, crowded_zones) MUST default to empty arrays

---

## 6. Test Execution Contract

### Interface

**Command**: `uv run pytest [options] [paths]`

**Exit Codes**:
```
0 = All tests passed
1 = At least one test failed
2 = Test execution was interrupted (KeyboardInterrupt)
3 = Internal pytest error
4 = pytest command line usage error
5 = No tests collected
```

**Output Formats**:

**Terminal (default)**:
```
======================== test session starts =========================
platform darwin -- Python 3.13.1, pytest-8.3.4, pluggy-1.5.0
rootdir: /Users/lennart/git/robot_sf_ll7
configfile: pyproject.toml
plugins: xdist-3.8.0, cov-7.0.0
collected 55 items

tests/test_gym_env.py::test_environment_creation PASSED        [  1%]
...
fast-pysf/tests/test_forces.py::test_desired_force PASSED      [ 50%]
...

===================== 55 passed in 12.34s =========================
```

**JSON (--json-report)**:
```json
{
  "created": 1698598800.0,
  "duration": 12.34,
  "exitcode": 0,
  "root": "/Users/lennart/git/robot_sf_ll7",
  "tests": [
    {
      "nodeid": "tests/test_gym_env.py::test_environment_creation",
      "outcome": "passed",
      "duration": 0.15,
      "call": {
        "duration": 0.12,
        "outcome": "passed"
      }
    }
  ],
  "summary": {
    "passed": 55,
    "failed": 0,
    "total": 55
  }
}
```

**Contract Guarantees**:
- Headless mode MUST work: `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest`
- Parallel execution MUST work: `uv run pytest -n auto`
- Selective execution MUST work: `uv run pytest -m "not slow"`
- Coverage MUST be collected automatically (no separate command needed)

---

## 7. Review Comment Tracking Contract

### Interface

**Location**: `specs/148-improve-fast-pysf/pr236_review_comments.md`

**Schema** (Markdown table format):
```markdown
| ID | File | Line | Priority | Category | Status | Resolution |
|----|------|------|----------|----------|--------|------------|
| 1  | map_config.py | 81 | High | Functionality | Resolved | Removed unreachable print |
| 2  | TestObstacleForce.py | 44 | High | Quality | In Progress | Adding assertion |
```

**Status Values**:
- `Open`: Not started
- `In Progress`: Work underway
- `Resolved`: Fix committed and verified
- `Won't Fix`: Decision to skip (with rationale)
- `Upstream`: Reported to fast-pysf upstream

**Contract Guarantees**:
- All 24 PR #236 comments MUST be tracked (SC-004)
- High-priority comments MUST be resolved before feature completion (SC-005)
- Medium/low priority MUST be documented in tasks.md for follow-up
- Resolution notes MUST include commit SHA when applicable

---

## Contract Validation

### Pre-Merge Checklist

- [ ] pytest configuration includes both testpaths
- [ ] Coverage configuration includes fast-pysf source
- [ ] Ruff checks pass for both robot_sf and fast-pysf
- [ ] Type checking runs without fatal errors
- [ ] Test fixtures exist and are valid JSON
- [ ] All 24 review comments tracked with status
- [ ] Test execution produces expected exit codes
- [ ] Coverage ≥ 70% for fast-pysf code

### Regression Prevention

- Configuration changes MUST NOT break existing workflows
- Test count MUST increase (not decrease)
- Coverage MUST NOT decrease for robot_sf
- Quality checks MUST NOT introduce new failures in robot_sf

---

## Next Steps

**Phase 1 Continuation**:
1. ✅ Data model defined (data-model.md)
2. ✅ API contracts documented (this file)
3. ⏳ Create `quickstart.md` (developer onboarding)
4. ⏳ Update agent context
5. ⏳ Re-evaluate Constitution Check
