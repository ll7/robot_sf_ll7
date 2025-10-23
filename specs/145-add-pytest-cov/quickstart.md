# Quick Start: Code Coverage Monitoring

**Feature**: pytest-cov integration for robot_sf  
**Date**: 2025-10-23  
**Audience**: Developers contributing to robot_sf

## Purpose

This guide helps you get started with code coverage monitoring in robot_sf. Coverage tracking is now integrated into the standard testing workflow with minimal configuration required.

---

## TL;DR - For the Impatient

```bash
# Run tests with coverage (automatic with default pytest)
uv run pytest tests

# View detailed HTML report
open htmlcov/index.html

# Identify coverage gaps
uv run python scripts/coverage/analyze_gaps.py

# Track trends over time
uv run python scripts/coverage/track_trends.py collect
```

---

## Installation

Coverage tools are installed automatically with the project dependencies.

```bash
# One-time setup (if you haven't already)
git submodule update --init --recursive
uv sync

# Verify coverage is installed
uv run pytest --version
# Should show pytest 8.3.3+ with pytest-cov plugin
```

---

## Basic Usage

### Run Tests with Coverage

Coverage collection is now **automatic** when you run tests:

```bash
# Standard test command (coverage included)
uv run pytest tests
```

**Output**: Terminal shows coverage summary after tests complete:
```
---------- coverage: platform darwin, python 3.11.7 -----------
Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
robot_sf/__init__.py                       12      0   100%
robot_sf/gym_env/__init__.py               8      2    75%   45-46
robot_sf/gym_env/env_config.py           190    103    46%   23-45, 67-91, ...
robot_sf/sim/simulator.py                182     87    52%   10-25, 45-78, ...
---------------------------------------------------------------------
TOTAL                                  12203   3969    67%
```

### VS Code Integration

Use VS Code tasks for convenient access:

1. **Cmd+Shift+P** → "Tasks: Run Task"
2. Select "Run Tests with Coverage"
3. View results in terminal
4. Select "Open Coverage Report" to view HTML report in browser

### View Detailed HTML Report

After running tests with coverage:

```bash
# macOS
open htmlcov/index.html

# Linux
xdg-open htmlcov/index.html

# Windows
start htmlcov/index.html
```

The HTML report provides:
- ✅ File-by-file coverage breakdown
- ✅ Syntax-highlighted source with covered/uncovered lines
- ✅ Drill-down navigation by module
- ✅ Coverage percentage for each file and function

---

## Coverage Gap Analysis

Identify the largest coverage gaps to prioritize testing efforts:

```bash
# Default: top 10 gaps in terminal
uv run python scripts/coverage/analyze_gaps.py

# Top 20 gaps
uv run python scripts/coverage/analyze_gaps.py --top-n 20

# Filter to specific module
uv run python scripts/coverage/analyze_gaps.py --module-filter robot_sf.gym_env

# Save as JSON for processing
uv run python scripts/coverage/analyze_gaps.py --output-format json --output-file gaps.json

# Generate Markdown report
uv run python scripts/coverage/analyze_gaps.py --output-format markdown > gaps.md
```

**Example Output**:
```
Coverage Gap Analysis Report
Generated: 2025-10-23 14:30:00
Overall Coverage: 67.45%
Gaps Identified: 23

Top 10 Coverage Gaps:
======================

1. robot_sf/gym_env/env_config.py
   Coverage: 45.8% | Uncovered: 103 lines | Priority: 154.5
   Missing ranges: 23-45, 67-91, 120-150
   Recommendation: Add unit tests for configuration validation methods

2. robot_sf/sim/simulator.py
   Coverage: 52.3% | Uncovered: 87 lines | Priority: 130.5
   Missing ranges: 10-25, 45-78, 120-135
   Recommendation: Add integration tests for simulation edge cases
```

Use this to decide which files need test coverage improvements first.

---

## Historical Trend Tracking

Track how coverage changes over time:

### Collect a Snapshot

```bash
# Collect current coverage snapshot
uv run python scripts/coverage/track_trends.py collect

# Specify custom history file location
uv run python scripts/coverage/track_trends.py collect \
  --history-file results/my_coverage_history.jsonl
```

**Note**: This happens automatically in CI on the main branch.

### Visualize Trends

```bash
# Generate line graph of overall coverage over time
uv run python scripts/coverage/track_trends.py visualize \
  --output-file coverage_trend.png

# Last 30 days only
uv run python scripts/coverage/track_trends.py visualize \
  --output-file recent_trend.png \
  --time-range last_30_days

# Module-specific heatmap
uv run python scripts/coverage/track_trends.py visualize \
  --output-file module_heatmap.png \
  --plot-type heatmap \
  --modules robot_sf.gym_env,robot_sf.sim,robot_sf.benchmark
```

### Trend Report

```bash
# Terminal summary of trends
uv run python scripts/coverage/track_trends.py report

# JSON export for analysis
uv run python scripts/coverage/track_trends.py report \
  --output-format json \
  --output-file trend_stats.json
```

---

## CI/CD Integration

Coverage is automatically collected in GitHub Actions CI:

### For Pull Requests

1. **Tests run with coverage** on your PR branch
2. **Coverage compared** against main branch baseline
3. **Warnings generated** if coverage decreases by >1%
4. **Build still passes** (warnings don't fail the build)
5. **Coverage report uploaded** as CI artifact

**View Coverage on Your PR**:
- Check the "Actions" tab on your PR
- Download "coverage-report" artifact
- Unzip and open `index.html` in browser

### For Main Branch

1. **Coverage collected** after tests pass
2. **Baseline updated** for future PR comparisons
3. **Trend snapshot** added to history (if configured)

---

## Configuration

Coverage is configured in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["robot_sf"]  # Only measure robot_sf/ library code
omit = [
    "*/tests/*",        # Exclude test files
    "*/examples/*",     # Exclude examples
    "*/scripts/*",      # Exclude scripts
]
parallel = true         # Support parallel test execution

[tool.coverage.report]
precision = 2           # Two decimal places for percentages
show_missing = true     # Show line numbers of uncovered code
```

**To customize**:
1. Edit `pyproject.toml` `[tool.coverage.*]` sections
2. Re-run tests to apply changes
3. No additional files needed (pytest auto-detects config)

---

## Common Workflows

### Before Submitting a PR

```bash
# 1. Run tests with coverage
uv run pytest tests

# 2. Check if you've improved coverage
# (Look for files you changed in the coverage report)

# 3. If coverage decreased, identify gaps
uv run python scripts/coverage/analyze_gaps.py

# 4. Add tests for high-priority gaps
# (Focus on files you modified)

# 5. Re-run to verify improvement
uv run pytest tests
```

### Investigating Low Coverage in a Module

```bash
# 1. Run tests with coverage
uv run pytest tests

# 2. Open HTML report
open htmlcov/index.html

# 3. Navigate to the module with low coverage

# 4. Identify uncovered lines (highlighted in red)

# 5. Write tests to cover those lines

# 6. Verify coverage increased
uv run pytest tests
```

### Setting Up Trend Tracking Locally

```bash
# 1. Collect initial baseline
uv run python scripts/coverage/track_trends.py collect

# 2. Make code/test changes

# 3. Collect new snapshot
uv run python scripts/coverage/track_trends.py collect

# 4. View trend over your development session
uv run python scripts/coverage/track_trends.py visualize \
  --output-file my_dev_trend.png

# 5. Generate summary
uv run python scripts/coverage/track_trends.py report
```

---

## Understanding Coverage Metrics

### Coverage Percentage
- **Definition**: (Covered lines / Total executable lines) × 100
- **Good**: 70%+ for core modules (gym_env, sim, benchmark)
- **Acceptable**: 50%+ for utilities and helpers
- **Needs Work**: <50% indicates significant testing gaps

### Missing Lines
- **Red highlighting** in HTML report shows uncovered lines
- **Line ranges** in terminal output (e.g., "23-45, 67-91")
- **Priority**: Focus on uncovered lines in critical code paths first

### Priority Score (Gap Analysis)
- **Higher = More Important**: Gaps in core modules with many uncovered lines
- **Formula**: (Uncovered lines × Location weight) + Complexity adjustment
- **Use**: Focus testing efforts on highest-priority gaps first

---

## Troubleshooting

### Coverage Data Not Generated

**Problem**: No coverage.json or .coverage file after running tests.

**Solutions**:
```bash
# 1. Verify pytest-cov is installed
uv run pytest --version  # Should list pytest-cov plugin

# 2. Re-sync dependencies
uv sync

# 3. Run with explicit coverage flags
uv run pytest tests --cov=robot_sf
```

### HTML Report Not Found

**Problem**: `htmlcov/` directory doesn't exist.

**Solutions**:
```bash
# 1. Ensure HTML report generation is enabled
uv run pytest tests --cov=robot_sf --cov-report=html

# 2. Check if directory was gitignored but not created
ls -la | grep htmlcov
```

### Coverage Seems Too Low

**Problem**: Coverage percentage unexpectedly low.

**Investigation**:
```bash
# 1. Check which files are being measured
grep "source =" pyproject.toml
# Should show: source = ["robot_sf"]

# 2. Verify tests are actually running
uv run pytest tests -v
# Should show test discovery and execution

# 3. Check if parallel mode is causing issues
# Edit pyproject.toml: parallel = false
# Re-run tests
```

### Gap Analysis Shows No Gaps

**Problem**: Gap analysis reports no gaps despite low coverage.

**Solutions**:
```bash
# 1. Lower minimum uncovered lines threshold
uv run python scripts/coverage/analyze_gaps.py --min-lines 1

# 2. Check if coverage.json exists and is recent
ls -lh coverage.json

# 3. Verify gap analyzer is reading correct file
uv run python scripts/coverage/analyze_gaps.py --coverage-file coverage.json
```

### CI Coverage Warnings on Unrelated Changes

**Problem**: PR gets coverage warnings for files you didn't change.

**Explanation**: Someone else's earlier PR may have decreased coverage and set a lower baseline.

**Solution**: 
- Check which files lost coverage in the warning message
- If you can easily add tests for those files, do so
- Otherwise, acknowledge the warning and document in PR description
- The build will still pass (warnings are non-blocking)

---

## Advanced Usage

### Branch Coverage (Optional)

Enable branch coverage to track if/else branches:

```toml
# pyproject.toml
[tool.coverage.run]
branch = true  # Enable branch coverage
```

**Trade-off**: More detailed coverage data but 15-20% slower test runs.

### Excluding Specific Lines

Exclude lines from coverage measurement:

```python
# Option 1: Inline pragma
def debug_only_function():  # pragma: no cover
    print("This is only for debugging")

# Option 2: Block exclusion
if TYPE_CHECKING:  # Automatically excluded
    from typing import SomeType
```

### Coverage for Specific Tests

Run coverage for a subset of tests:

```bash
# Single test file
uv run pytest tests/gym_env/test_env_config.py --cov=robot_sf

# Specific test function
uv run pytest tests/gym_env/test_env_config.py::test_basic_config --cov=robot_sf

# Test marker
uv run pytest -m "not slow" --cov=robot_sf
```

---

## Best Practices

1. **Run coverage regularly**: Not just before PRs; integrate into daily workflow
2. **Focus on quality, not percentage**: 100% coverage doesn't guarantee bug-free code
3. **Prioritize critical paths**: Core navigation logic > utility helpers
4. **Test behavior, not lines**: Aim to test edge cases, not just achieve coverage
5. **Use gap analysis**: Let tooling identify where tests are most needed
6. **Track trends**: Monitor coverage over time to catch regressions early
7. **Don't game metrics**: Adding meaningless assertions to increase coverage helps no one

---

## Next Steps

- **Read the full documentation**: `docs/coverage_guide.md` (coming soon)
- **Check existing coverage**: `open htmlcov/index.html`
- **Identify gaps to address**: `uv run python scripts/coverage/analyze_gaps.py`
- **Write tests for high-priority gaps**: Focus on files you're actively working on
- **Monitor trends**: Set up local trend tracking to see your progress

---

## Getting Help

- **Development guide**: `docs/dev_guide.md` (coverage section)
- **CLI contracts**: `specs/145-add-pytest-cov/contracts/cli-contracts.md`
- **Data model**: `specs/145-add-pytest-cov/data-model.md`
- **GitHub Issues**: Report bugs or request features
- **PR reviews**: Ask maintainers about coverage expectations

---

**Happy testing! 🧪 Every line covered is a potential bug prevented. 🐛→✅**
