# Coverage Guide

**Purpose**: Comprehensive guide to coverage collection, baseline tracking, gap analysis, and trend monitoring in robot_sf_ll7.

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

The robot_sf_ll7 project uses `pytest-cov` for automatic code coverage measurement. Coverage is collected during every test run and reported in multiple formats (terminal, HTML, JSON).

**Design Principles**:
- **Non-intrusive**: Coverage collection is automatic, no extra commands needed
- **Informative**: Multiple report formats for different use cases
- **CI-friendly**: Non-blocking baseline warnings prevent coverage regressions
- **Developer-focused**: VS Code tasks and HTML reports for local workflow

## Quick Start

```bash
# Run tests (coverage collected automatically)
uv run pytest tests

# View HTML report (cross-platform)
uv run python scripts/coverage/open_coverage_report.py

# Or use VS Code task: "Open Coverage Report"

# Compare with baseline (local)
uv run python scripts/coverage/compare_coverage.py \
  --current output/coverage/coverage.json \
  --baseline output/coverage/.coverage-baseline.json \
  --format terminal
```

## Automatic Collection

Coverage is enabled by default through `pyproject.toml` configuration:

```toml
[tool.pytest.ini_options]
addopts = [
    "--cov=robot_sf",           # Measure coverage for robot_sf package
    "--cov-report=term",         # Terminal summary
    "--cov-report=html",         # Interactive HTML report
    "--cov-report=json",         # Machine-readable JSON
]
```

**What gets measured**:
- All code in `robot_sf/` package
- Excludes: tests, examples, scripts, fast-pysf submodule

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

# Make changes, run tests
uv run pytest tests

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

Coverage comparison and publishing run automatically in CI using a reusable sequence of steps:

```yaml
# 1. Tests run with automatic coverage collection
- name: Unit tests
  run: uv run pytest -q -n auto

# 2. Restore baseline from cache
- name: Restore coverage baseline
  uses: actions/cache@v4
  with:
    path: output/coverage/.coverage-baseline.json
    key: coverage-baseline-${{ github.ref_name }}

# 3. Compare (non-blocking)
- name: Compare coverage
  continue-on-error: true  # Warning only, doesn't fail CI
  run: |
    uv run python scripts/coverage/compare_coverage.py \
      --current output/coverage/coverage.json \
      --baseline output/coverage/.coverage-baseline.json \
      --format github

# 4. Update baseline on main
- name: Update baseline
  if: github.ref == 'refs/heads/main'
  run: |
    mkdir -p output/coverage
    cp output/coverage/coverage.json output/coverage/.coverage-baseline.json

# 5. Upload artifacts
- name: Upload coverage
  uses: actions/upload-artifact@v4
  with:
    path: |
      output/coverage/coverage.json
      output/coverage/htmlcov/
      output/coverage/.coverage
```

### Cache Strategy

- **Key**: `coverage-baseline-{branch}-{sha}` for main, `coverage-baseline-{branch}` for PRs
- **Restore keys**: Falls back to main branch baseline if PR baseline missing
- **Update**: Only main branch pushes update the baseline
- **Retention**: GitHub Actions default (90 days)

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

All coverage settings live in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["robot_sf"]              # Package to measure
omit = [
    "tests/*",                     # Exclude test code
    "examples/*",                  # Exclude demos
    "scripts/*",                   # Exclude CLI tools
    "fast-pysf/*",                 # Exclude submodule
]
parallel = true                    # Support pytest-xdist

[tool.coverage.report]
precision = 2                      # Decimal places (91.73%)
show_missing = true                # Show missing line numbers
exclude_lines = [
    "pragma: no cover",            # Exclude marked lines
    "def __repr__",                # Exclude __repr__ methods
    "raise AssertionError",        # Exclude defensive assertions
    "if __name__ == .__main__.:",  # Exclude script entry points
]

[tool.coverage.html]
directory = "output/coverage/htmlcov"  # HTML output directory

[tool.coverage.json]
output = "output/coverage/coverage.json"  # JSON output file
show_contexts = false                   # Don't include test names
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
**Solution**: Ensure pytest runs from repository root with coverage enabled
```bash
cd robot_sf_ll7
uv run pytest tests  # Not: pytest tests (uses wrong Python)
```

### Baseline comparison fails

**Symptom**: `FileNotFoundError: output/coverage/.coverage-baseline.json`
**Solution**: Create baseline first
```bash
uv run pytest tests
mkdir -p output/coverage
cp output/coverage/coverage.json output/coverage/.coverage-baseline.json
```

### Coverage percentage seems wrong

**Symptom**: Very low coverage (< 5%) despite writing tests
**Solution**: Check `[tool.coverage.run] source` matches your package name
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
uv run pytest tests/test_gym_env.py --cov=robot_sf.gym_env

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

## Future Enhancements

Planned features (not yet implemented):
- **Gap Analysis** (User Story 3): Identify largest coverage gaps with priority scoring
- **Trend Tracking** (User Story 4): Historical coverage tracking with visualization
- **Automated prioritization**: Suggest which files to test next
- **Coverage badges**: README badges showing current coverage percentage

See `specs/145-add-pytest-cov/tasks.md` for implementation roadmap.

## Related Documentation

- [Development Guide](dev_guide.md) - Coverage workflow overview
- [Testing Strategy](dev_guide.md#testing-strategy-three-test-suites) - Test suite organization
- [CI/CD Expectations](dev_guide.md#cicd-expectations) - Quality gates and pipeline

## External References

- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [coverage.py documentation](https://coverage.readthedocs.io/)
- [GitHub Actions cache documentation](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
