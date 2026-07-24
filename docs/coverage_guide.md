# Coverage Guide

**Purpose**: Guide to the implemented pytest-cov coverage collection, reporting, and baseline comparison workflow in robot_sf_ll7.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Automatic Collection](#automatic-collection)
- [Baseline Comparison](#baseline-comparison)
- [CI/CD Integration](#cicd-integration)
- [Coverage Reports](#coverage-reports)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Overview

The robot_sf_ll7 project uses `pytest-cov` and Python's `coverage` library for code coverage measurement. Coverage collection is opt-in for local test runs and runs automatically on non-PR CI events (main branch, merge queue, manual dispatch).

**Design Principles**:
- **Explicit opt-in**: Default local pytest runs skip coverage collection for maximum speed; explicit wrapper opt-in (`ROBOT_SF_PYTEST_COVERAGE=1`) or non-PR CI runs collect coverage data.
- **Informative**: Multiple report formats for different use cases (terminal, HTML, JSON).
- **CI-friendly**: Non-PR CI runs enforce an absolute 85.0% coverage floor and report advisory baseline warnings.
- **Developer-focused**: Cross-platform HTML report helper and VS Code tasks for local workflow.

## Quick Start

```bash
# Run tests normally (no coverage by default for fast execution)
uv run pytest tests

# View HTML report (cross-platform)
uv run python scripts/coverage/open_coverage_report.py

# Or use VS Code task: "Open Coverage Report"

# Run tests with coverage collection (local opt-in)
ROBOT_SF_PYTEST_COVERAGE=1 scripts/dev/run_tests_parallel.sh tests

# Compare with baseline (local)
uv run python scripts/coverage/compare_coverage.py \
  --current output/coverage/coverage.json \
  --baseline output/coverage/.coverage-baseline.json \
  --format terminal
```

## Coverage Collection

Coverage tooling is configured in `pyproject.toml`, and `pytest-cov` ships in the `dev`
dependency group. The default pytest invocation does not collect coverage — pass `--cov`
explicitly or set `ROBOT_SF_PYTEST_COVERAGE=1` with `scripts/dev/run_tests_parallel.sh`. The base pytest options are:

```toml
[tool.pytest.ini_options]
addopts = [
    "--import-mode=importlib",
    "--durations=10",
]
testpaths = ["tests", "fast-pysf/tests"]
```

**What gets measured by the canonical command and CI**:
- Only the `robot_sf/` package. The wrapper and non-PR CI pass `--cov=robot_sf`; pytest-cov uses
  that command-line source selection instead of the broader `source = ["robot_sf",
  "fast-pysf/pysocialforce"]` value in `pyproject.toml`.
- `fast-pysf/pysocialforce` is therefore not included in the local wrapper report, the non-PR CI
  shards, or the 85.0% coverage gate.
- The configured omit patterns still exclude test files (`*/tests/*`, `*/test_*`,
  `fast-pysf/tests/*`), `tests/pygame/*`, examples (`examples/*`, `fast-pysf/examples/*`),
  `scripts/*`, `*/conftest.py`, and `*/__pycache__/*` whenever their configured source list is used.

**Output artifacts** (under the canonical `output/coverage/` root):
- `output/coverage/.coverage` - SQLite database (raw data)
- `output/coverage/coverage.json` - JSON export for tooling
- `output/coverage/htmlcov/index.html` - Interactive HTML report

All artifacts are gitignored and regenerated on each test run.

## Baseline Comparison

The baseline comparison system detects coverage regressions across commits/branches.

### Local Usage

```bash
# Create baseline from current coverage
mkdir -p output/coverage
cp output/coverage/coverage.json output/coverage/.coverage-baseline.json

# Make changes, then collect the canonical coverage snapshot
ROBOT_SF_PYTEST_COVERAGE=1 scripts/dev/run_tests_parallel.sh tests

# Compare
uv run python scripts/coverage/compare_coverage.py \
  --current output/coverage/coverage.json \
  --baseline output/coverage/.coverage-baseline.json \
  --format terminal \
  --threshold 1.0

# Output formats
--format terminal   # Human-readable terminal output
--format github     # GitHub Actions annotations
--format json       # Machine-readable JSON

# Fail on coverage decrease (CI mode)
--fail-on-decrease  # Exit code 1 if coverage drops
```

### Understanding Comparison Output

**Terminal format**:
```
⚠️  Coverage Decreased
Overall: 85.5% → 83.2% (-2.3%)

Files with decreased coverage:
  robot_sf/gym_env/environment.py: 90.0% → 85.0% (-5.0%)
  robot_sf/sim/simulator.py: 75.0% → 70.0% (-5.0%)
```

**GitHub format** (used in CI):
```
::warning file=robot_sf/gym_env/environment.py::Coverage decreased from 90.0% to 85.0% (-5.0%)
```

**JSON format** (for tooling):
```json
{
  "has_decrease": true,
  "overall_change": -2.3,
  "files_decreased": [
    {
      "file": "robot_sf/gym_env/environment.py",
      "before": 90.0,
      "after": 85.0,
      "change": -5.0
    }
  ]
}
```

## CI/CD Integration

Coverage collection and enforcement run automatically in CI (`.github/workflows/ci.yml`) with the following architecture:

1. **Fast-feedback sharding**: The `fast-feedback` job distributes pytest execution across four runners (`PYTEST_SHARD_COUNT: 4`, `PYTEST_SHARD_INDEX: 1..4`).
   - On **pull request** events, coverage collection is disabled (`ROBOT_SF_PYTEST_COVERAGE: 0`) to keep PR feedback fast.
   - On **non-PR** events (`main`, `merge_group`, `workflow_dispatch`), coverage is enabled (`ROBOT_SF_PYTEST_COVERAGE: 1`, `ROBOT_SF_SHARD_INCLUDE_SLOW: 1`) using the high-performance CPython 3.12+ `sys.monitoring` backend (`COVERAGE_CORE: sysmon`). Each shard writes to its own database (`output/coverage/.coverage.<shard>`) and uploads an artifact (`coverage-shard-<shard>`).
2. **Coverage Gate**: On non-PR events, after all `fast-feedback` shards pass, the `coverage-gate` job executes:
   - Downloads all shard databases (`coverage-shard-*`).
   - Combines them: `uv run coverage combine output/coverage`.
   - Exports JSON and HTML reports: `uv run coverage json` and `uv run coverage html`.
   - Enforces the **absolute coverage floor** (fails CI if overall coverage drops below 85.0%):
     ```bash
     uv run python scripts/coverage/compare_coverage.py \
       --current output/coverage/coverage.json \
       --absolute-only \
       --format github \
       --minimum-total 85.0
     ```
   - Compares with the main baseline if available (`--threshold 1.0`, non-blocking advisory warnings).
   - Updates and caches `output/coverage/.coverage-baseline.json` on `main` branch pushes.
   - Uploads combined coverage artifacts (`coverage.json`, `htmlcov/`, `.coverage`).

```yaml
# Reusable pipeline summary for non-PR coverage execution in .github/workflows/ci.yml:
coverage-gate:
  runs-on: ubuntu-latest
  needs: fast-feedback
  if: ${{ github.event_name != 'pull_request' && needs.fast-feedback.result == 'success' }}
  steps:
    - name: Download coverage shards
      uses: actions/download-artifact@v4
      with:
        pattern: coverage-shard-*
        path: output/coverage
        merge-multiple: true

    - name: Combine coverage shards
      run: |
        uv run coverage combine output/coverage
        uv run coverage json
        uv run coverage html

    - name: Enforce absolute coverage floor
      run: >-
        uv run python scripts/coverage/compare_coverage.py
        --current output/coverage/coverage.json
        --absolute-only
        --format github
        --minimum-total 85.0

    - name: Restore coverage baseline
      uses: actions/cache@v4
      with:
        path: output/coverage/.coverage-baseline.json
        key: coverage-baseline-${{ github.ref_name }}
        restore-keys: |
          coverage-baseline-main

    - name: Compare coverage with baseline
      continue-on-error: true
      run: |
        if [ -f output/coverage/.coverage-baseline.json ]; then
          uv run python scripts/coverage/compare_coverage.py \
            --current output/coverage/coverage.json \
            --baseline output/coverage/.coverage-baseline.json \
            --format github \
            --threshold 1.0
        fi

    - name: Update coverage baseline (main branch only)
      if: github.ref == 'refs/heads/main' && success()
      run: |
        mkdir -p output/coverage
        cp output/coverage/coverage.json output/coverage/.coverage-baseline.json

    - name: Save coverage baseline (main branch only)
      if: github.ref == 'refs/heads/main' && success()
      uses: actions/cache/save@v4
      with:
        path: output/coverage/.coverage-baseline.json
        key: coverage-baseline-${{ github.ref_name }}-${{ github.sha }}
```

### Cache Strategy

- **Key**: `coverage-baseline-{branch_name}-{sha}` for main branch pushes.
- **Restore keys**: `coverage-baseline-main` fallback if branch baseline is missing.
- **Update**: Only main branch pushes update the baseline.
- **Retention**: GitHub Actions default (90 days).

## Coverage Reports

### Terminal Summary

Automatically printed after test runs:

```
Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
robot_sf/gym_env/environment.py           150     15  90.00%  42-45, 89-92
robot_sf/sim/simulator.py                 200     50  75.00%  10-20, 150-180
---------------------------------------------------------------------
TOTAL                                   10605    876  91.73%
```

- **Stmts**: Total executable lines
- **Miss**: Uncovered lines
- **Cover**: Percentage covered
- **Missing**: Line numbers not executed

### HTML Report

Interactive web interface with:
- File-by-file coverage breakdown
- Highlighted source code (green = covered, red = missed)
- Branch coverage details
- Sortable columns

Access via: `uv run python scripts/coverage/open_coverage_report.py` (opens `output/coverage/htmlcov/index.html`) or use the VS Code task "Open Coverage Report".

### JSON Export

Machine-readable format for tooling:

```json
{
  "meta": {
    "version": "7.11.0",
    "timestamp": "2025-01-12T10:30:00"
  },
  "files": {
    "robot_sf/gym_env/environment.py": {
      "summary": {
        "covered_lines": 135,
        "num_statements": 150,
        "percent_covered": 90.0,
        "missing_lines": 15
      }
    }
  },
  "totals": {
    "covered_lines": 9729,
    "num_statements": 10605,
    "percent_covered": 91.73
  }
}
```

## Configuration

All coverage defaults live in `pyproject.toml`. The listed `source` array is a coverage.py default;
the canonical local wrapper and non-PR CI override it with `--cov=robot_sf`, so their reports measure
only `robot_sf`.

```toml
[tool.coverage.run]
source = ["robot_sf", "fast-pysf/pysocialforce"]

# Omit test files and non-library code from coverage measurement
omit = [
    "*/tests/*",
    "*/test_*",
    "tests/pygame/*",
    "examples/*",
    "scripts/*",
    "fast-pysf/tests/*",
    "fast-pysf/examples/*",
    "*/conftest.py",
    "*/__pycache__/*",
]
parallel = true
branch = true
data_file = "output/coverage/.coverage"

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
skip_empty = false
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]

[tool.coverage.html]
directory = "output/coverage/htmlcov"
show_contexts = false

[tool.coverage.json]
output = "output/coverage/coverage.json"
show_contexts = false
```

### Customization

**Exclude specific code from coverage**:
```python
def debug_helper():  # pragma: no cover
    """Only used in interactive debugging"""
    breakpoint()
```

**Adjust thresholds**:
```bash
# Stricter threshold (fail if coverage drops >0.5%)
--threshold 0.5

# Looser threshold (fail if coverage drops >2.0%)
--threshold 2.0
```

## Troubleshooting

### Coverage data not generated

**Symptom**: No `output/coverage/coverage.json` or `output/coverage/htmlcov/` after tests
**Solution**: Ensure the canonical wrapper runs from repository root with coverage enabled
```bash
cd robot_sf_ll7
ROBOT_SF_PYTEST_COVERAGE=1 scripts/dev/run_tests_parallel.sh tests
```

### Baseline comparison fails

**Symptom**: `FileNotFoundError: output/coverage/.coverage-baseline.json`
**Solution**: Create baseline first
```bash
ROBOT_SF_PYTEST_COVERAGE=1 scripts/dev/run_tests_parallel.sh tests
mkdir -p output/coverage
cp output/coverage/coverage.json output/coverage/.coverage-baseline.json
```

### Coverage percentage seems wrong

**Symptom**: Very low coverage (< 5%) despite writing tests
**Solution**: Check the command-line `--cov` source selection matches your package name. The
canonical wrapper intentionally measures `robot_sf` only.
```toml
source = ["robot_sf"]  # Must match actual package directory
```

### Parallel execution issues

**Symptom**: Coverage data missing with `pytest -n auto`
**Solution**: Ensure `parallel = true` in `pyproject.toml`
```toml
[tool.coverage.run]
parallel = true  # Required for pytest-xdist
```

### CI warnings not appearing

**Symptom**: No GitHub annotations despite coverage decrease
**Solution**: Check `--format github` is used in CI workflow
```yaml
run: |
  uv run python scripts/coverage/compare_coverage.py \
    --format github  # Not: terminal
```

### HTML report shows wrong files

**Symptom**: Reports include test files or submodules
**Solution**: Adjust `omit` patterns in `pyproject.toml`
```toml
omit = [
    "tests/*",      # All test files
    "fast-pysf/*",  # Submodule
    "*/conftest.py" # Pytest fixtures
]
```

### Browser won't open automatically

**Symptom**: VS Code task or script doesn't open browser
**Solution**: Use the cross-platform opener script
```bash
# Cross-platform (works on macOS, Linux, Windows)
uv run python scripts/coverage/open_coverage_report.py

# Manual fallback (platform-specific)
# macOS: open htmlcov/index.html
# Linux: xdg-open htmlcov/index.html
# Windows: start htmlcov/index.html
```

The `open_coverage_report.py` script uses Python's `webbrowser` module with proper file:// URL handling for reliable cross-platform browser launching.

## Advanced Usage

### Combining with coverage thresholds

```bash
# Fail if total coverage < 90%
uv run pytest tests --cov-fail-under=90

# Fail if any file < 80%
uv run pytest tests --cov-fail-under=80 --cov-branch
```

### Branch coverage

Enable branch coverage to track conditional execution:
```toml
[tool.coverage.run]
branch = true  # Measure if/else branches taken
```

### Coverage for specific tests

```bash
# Only measure coverage for integration tests
uv run pytest tests/test_gymnasium_env_contracts.py --cov=robot_sf.gym_env

# Measure coverage for single module
uv run pytest tests --cov=robot_sf.benchmark
```

### Programmatic access

```python
from robot_sf.coverage_tools.baseline_comparator import (
    CoverageSnapshot,
    compare,
    generate_warning
)

# Load current coverage
current = CoverageSnapshot.from_coverage_json("output/coverage/coverage.json")

# Compare with baseline
delta = compare(
    current_path="output/coverage/coverage.json",
    baseline_path="output/coverage/.coverage-baseline.json",
    threshold=1.0,
)

# Generate warning
if delta.has_decrease:
    warning = generate_warning(delta, format_type="terminal")
    print(warning)
```

## Descoped Coverage Ideas

The current coverage tooling stops at automatic pytest-cov collection, local HTML/JSON/terminal
reports, and baseline comparison warnings. The following ideas were listed in the original
`specs/145-add-pytest-cov/tasks.md` roadmap, but were not implemented and are no longer active
planned work as of the conservative issue #3349 descope decision:

- **Gap Analysis** (formerly User Story 3): no gap-analysis coverage CLI or module exists.
- **Trend Tracking** (formerly User Story 4): no trend-tracking coverage CLI, historical JSONL
  store, or coverage visualization workflow exists.
- **Automated prioritization**: Coverage tooling does not rank files or suggest what to test next.
- **Coverage badges**: README or project badges are not generated by the coverage workflow.

Treat those items as deferred historical scope, not as a promise or implementation roadmap. A future
change may reopen one of them only with a fresh issue, acceptance criteria, and validation plan.

## Related Documentation

- [Development Guide](dev_guide.md) - Coverage workflow overview
- [Testing Strategy](dev_guide.md#testing-strategy-three-test-suites) - Test suite organization
- [CI/CD Expectations](dev_guide.md#cicd-expectations) - Quality gates and pipeline

## External References

- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [coverage.py documentation](https://coverage.readthedocs.io/)
- [GitHub Actions cache documentation](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
