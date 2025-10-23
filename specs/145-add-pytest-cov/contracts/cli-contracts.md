# Coverage Tools CLI Contracts

This document defines the command-line interfaces for coverage analysis tools.

## Gap Analysis CLI

**Command**: `uv run python scripts/coverage/analyze_gaps.py`

**Purpose**: Identify and rank test coverage gaps by file priority and uncovered line count.

### Arguments

```bash
analyze_gaps.py [OPTIONS]
```

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--coverage-file` | path | No | `coverage.json` | Path to coverage JSON file |
| `--top-n` | int | No | `10` | Number of top gaps to report |
| `--min-lines` | int | No | `5` | Minimum uncovered lines to qualify as gap |
| `--output-format` | choice | No | `terminal` | Output format: `terminal`, `json`, `markdown` |
| `--output-file` | path | No | stdout | File to write report (defaults to terminal) |
| `--module-filter` | string | No | None | Filter to specific module (e.g., `robot_sf.gym_env`) |

### Output Formats

#### Terminal Format (default)
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

...
```

#### JSON Format
```json
{
  "generated_at": "2025-10-23T14:30:00Z",
  "snapshot_timestamp": "2025-10-23T14:25:00Z",
  "overall_coverage": 67.45,
  "total_gaps_identified": 23,
  "top_gaps": [
    {
      "file_path": "robot_sf/gym_env/env_config.py",
      "coverage_percent": 45.8,
      "uncovered_lines": 103,
      "missing_line_ranges": [[23, 45], [67, 91], [120, 150]],
      "priority_score": 154.5,
      "location_weight": 1.5,
      "module_path": "robot_sf.gym_env",
      "recommendation": "Add unit tests for configuration validation methods"
    }
  ],
  "summary_stats": {
    "total_uncovered_lines": 1234,
    "avg_gap_priority": 87.3,
    "core_module_gaps": 8
  }
}
```

#### Markdown Format
```markdown
# Coverage Gap Analysis Report

**Generated**: 2025-10-23 14:30:00  
**Overall Coverage**: 67.45%  
**Gaps Identified**: 23

## Top 10 Coverage Gaps

| Rank | File | Coverage | Uncovered Lines | Priority | Recommendation |
|------|------|----------|----------------|----------|----------------|
| 1 | robot_sf/gym_env/env_config.py | 45.8% | 103 | 154.5 | Add unit tests for configuration validation |
| 2 | robot_sf/sim/simulator.py | 52.3% | 87 | 130.5 | Add integration tests for simulation edge cases |
...
```

### Exit Codes

- `0`: Success
- `1`: Invalid arguments
- `2`: Coverage file not found
- `3`: Invalid coverage data format

---

## Trend Tracking CLI

**Command**: `uv run python scripts/coverage/track_trends.py`

**Purpose**: Track coverage metrics over time and visualize trends.

### Subcommands

#### collect
Collect a new coverage snapshot and append to trend history.

```bash
track_trends.py collect [OPTIONS]
```

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--coverage-file` | path | No | `coverage.json` | Path to coverage JSON file |
| `--history-file` | path | No | `results/coverage_history.jsonl` | Path to trend history JSONL |
| `--branch` | string | No | current branch | Git branch name |
| `--commit` | string | No | current commit | Git commit SHA |

**Output**: Appends snapshot to history file, prints confirmation

**Exit Codes**:
- `0`: Success
- `1`: Invalid arguments
- `2`: Coverage file not found
- `3`: History file write error

#### visualize
Generate trend visualizations from history.

```bash
track_trends.py visualize [OPTIONS]
```

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--history-file` | path | No | `results/coverage_history.jsonl` | Path to trend history JSONL |
| `--output-file` | path | Yes | - | Path to save visualization (PNG/PDF/SVG) |
| `--plot-type` | choice | No | `line` | Plot type: `line`, `heatmap`, `comparison` |
| `--time-range` | choice | No | `all` | Time range: `all`, `last_30_days`, `last_100_commits` |
| `--modules` | string | No | `all` | Comma-separated module paths to plot |
| `--title` | string | No | Auto-generated | Graph title |

**Output**: Saves visualization to file, prints path

**Exit Codes**:
- `0`: Success
- `1`: Invalid arguments
- `2`: History file not found
- `3`: Insufficient data for visualization

#### report
Generate textual trend report.

```bash
track_trends.py report [OPTIONS]
```

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--history-file` | path | No | `results/coverage_history.jsonl` | Path to trend history JSONL |
| `--output-format` | choice | No | `terminal` | Output format: `terminal`, `json`, `markdown` |
| `--output-file` | path | No | stdout | File to write report |

**Output**: Trend statistics and analysis

---

## Baseline Comparison CLI

**Command**: `uv run python scripts/coverage/compare_coverage.py`

**Purpose**: Compare current coverage against baseline and generate warnings.

### Arguments

```bash
compare_coverage.py [OPTIONS]
```

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--current` | path | Yes | - | Path to current coverage JSON |
| `--baseline` | path | Yes | - | Path to baseline coverage JSON |
| `--threshold` | float | No | `1.0` | Coverage decrease threshold (%) for warnings |
| `--format` | choice | No | `terminal` | Output format: `terminal`, `github-actions`, `json` |
| `--fail-on-decrease` | flag | No | False | Exit with code 1 on coverage decrease (overrides non-blocking) |

### Output Formats

#### Terminal Format (default)
```
Coverage Comparison Report
==========================
Baseline: main (commit a1b2c3d, 2025-10-20)
Current:  PR #123 (commit e5f6789, 2025-10-23)

Overall Coverage: 65.15% → 67.45% (+2.30%) ✓

Module Changes:
  robot_sf.gym_env: 77.5% → 72.3% (-5.2%) ⚠
  robot_sf.sim:     64.7% → 65.8% (+1.1%) ✓
  robot_sf.benchmark: 58.2% → 58.2% (0.0%)

Files with Significant Changes:
  robot_sf/gym_env/env_config.py: -8.4% (degraded)
  robot_sf/sim/simulator.py: +3.2% (improved)

Status: ✓ PASSED (overall coverage improved)
```

#### GitHub Actions Format
```
::warning file=robot_sf/gym_env/env_config.py,line=1,title=Coverage Decreased::Coverage decreased from 54.2% to 45.8% (-8.4%)
::notice title=Coverage Summary::Overall coverage: 65.15% → 67.45% (+2.30%)
```

#### JSON Format
```json
{
  "baseline": {
    "commit": "a1b2c3d",
    "branch": "main",
    "timestamp": "2025-10-20T12:00:00Z",
    "coverage_overall": 65.15
  },
  "current": {
    "commit": "e5f6789",
    "branch": "145-add-pytest-cov",
    "timestamp": "2025-10-23T14:30:00Z",
    "coverage_overall": 67.45
  },
  "delta": {
    "overall_change": 2.30,
    "module_changes": {
      "robot_sf.gym_env": -5.2,
      "robot_sf.sim": 1.1
    },
    "degraded_files": [
      ["robot_sf/gym_env/env_config.py", -8.4]
    ],
    "improved_files": [
      ["robot_sf/sim/simulator.py", 3.2]
    ]
  },
  "status": "passed",
  "warnings": [
    "robot_sf.gym_env module coverage decreased by 5.2%"
  ]
}
```

### Exit Codes

- `0`: Success (coverage maintained or improved, or warnings only)
- `1`: Coverage decreased below threshold (only if `--fail-on-decrease`)
- `2`: Invalid arguments
- `3`: Coverage files not found or invalid

---

## Configuration File Contract

Coverage configuration in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["robot_sf"]
omit = [
    "*/tests/*",
    "*/test_pygame/*",
    "*/examples/*",
    "*/scripts/*",
    "*/__pycache__/*",
    "*/fast-pysf/*"
]
parallel = true
branch = false  # Line coverage only (branch coverage optional)

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@(abc\\.)?abstractmethod"
]

[tool.coverage.html]
directory = "htmlcov"

[tool.coverage.json]
output = "coverage.json"

[tool.pytest.ini_options]
# Coverage options added to default pytest invocation
addopts = [
    "--cov=robot_sf",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=json"
]
```

Optional gap analysis config in `pyproject.toml`:

```toml
[tool.robot_sf.coverage.gap_analysis]
top_n_gaps = 10
min_uncovered_lines = 5
location_weights = {
    "robot_sf.gym_env" = 1.5,
    "robot_sf.sim" = 1.5,
    "robot_sf.nav" = 1.4,
    "robot_sf.benchmark" = 1.3,
    "robot_sf.render" = 1.2
}

[tool.robot_sf.coverage.trends]
storage_path = "results/coverage_history.jsonl"
retention_commits = 100
retention_days = 90
collection_branch = "main"
auto_collect = true
```

---

## GitHub Actions Integration

Coverage job in `.github/workflows/ci.yml`:

```yaml
coverage:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install uv
        uv sync
    
    - name: Run tests with coverage
      run: |
        uv run pytest tests \
          --cov=robot_sf \
          --cov-report=json \
          --cov-report=term
    
    - name: Cache baseline coverage
      id: cache-baseline
      uses: actions/cache@v4
      with:
        path: .coverage-baseline.json
        key: coverage-baseline-${{ github.base_ref || 'main' }}
    
    - name: Compare coverage
      if: github.event_name == 'pull_request'
      run: |
        uv run python scripts/coverage/compare_coverage.py \
          --current coverage.json \
          --baseline .coverage-baseline.json \
          --threshold 1.0 \
          --format github-actions
      continue-on-error: true  # Never fail the build
    
    - name: Update baseline
      if: github.ref == 'refs/heads/main'
      run: |
        cp coverage.json .coverage-baseline.json
    
    - name: Upload coverage HTML
      uses: actions/upload-artifact@v4
      with:
        name: coverage-report
        path: htmlcov/
```

---

## VS Code Tasks

New tasks in `.vscode/tasks.json`:

```json
{
  "label": "Run Tests with Coverage",
  "type": "shell",
  "command": "uv run pytest tests --cov=robot_sf --cov-report=term-missing --cov-report=html",
  "group": "test",
  "problemMatcher": []
},
{
  "label": "Coverage Gap Analysis",
  "type": "shell",
  "command": "uv run python scripts/coverage/analyze_gaps.py --output-format=terminal",
  "group": "test",
  "problemMatcher": []
},
{
  "label": "Open Coverage Report",
  "type": "shell",
  "command": "open htmlcov/index.html",
  "group": "test",
  "problemMatcher": []
}
```

---

## Summary

These contracts define:
- ✅ **CLI interfaces** for all coverage tools with clear arguments and outputs
- ✅ **Output formats** supporting terminal, JSON, and GitHub Actions
- ✅ **Configuration contracts** in pyproject.toml
- ✅ **CI integration** with non-blocking warning patterns
- ✅ **VS Code tasks** for developer convenience
- ✅ **Exit codes** for proper error handling
- ✅ **Extensibility** through optional configuration sections

All contracts align with:
- Non-intrusive defaults (coverage auto-collected)
- Warning-only CI (exit 0 even on decrease)
- Multiple output formats for different consumers
- Clear error handling and validation
