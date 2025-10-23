# Research: Code Coverage Monitoring and Quality Tracking

**Feature**: pytest-cov integration for robot_sf  
**Date**: 2025-10-23  
**Status**: Complete

## Purpose

This document captures research decisions for implementing non-intrusive code coverage monitoring with CI/CD integration, gap analysis, and historical trend tracking in the robot_sf repository.

## Research Questions & Decisions

### 1. Coverage Tool Selection

**Decision**: pytest-cov with coverage.py backend

**Rationale**:
- **pytest-cov** is the de facto standard for pytest-based Python projects
- Seamless integration with existing pytest infrastructure (already using pytest 8.3.3+)
- Supports parallel test execution with pytest-xdist (already in CI: `pytest -n auto`)
- Provides multiple output formats: terminal, HTML, JSON, XML (Cobertura for CI)
- Well-maintained, stable, and widely adopted (50M+ downloads/month)
- Zero-configuration defaults work out of the box with `--cov` flag
- coverage.py backend provides programmatic API for custom analysis tools

**Alternatives Considered**:
- **coverage.py directly**: More granular control but less pytest integration; requires custom pytest plugin for test discovery integration
- **pytest-coverage**: Older, less maintained fork; pytest-cov supersedes it
- **codecov/coveralls SaaS**: Adds external dependency and data exfiltration; violates repository security policy
- **Python trace module**: Low-level, requires significant custom tooling

**Rejected Because**: pytest-cov provides the best balance of integration, features, and community support without external dependencies.

---

### 2. Configuration Strategy

**Decision**: Hybrid pyproject.toml + minimal .coveragerc approach

**Rationale**:
- **pyproject.toml**: Central dependency and tool configuration location (already used for ruff, pytest)
- **Basic coverage options in pyproject.toml**: source paths, omit patterns, parallel mode
- **.coveragerc (optional)**: Only if advanced features needed (branch coverage, contexts)
- **Default behavior**: `uv run pytest tests` automatically collects coverage without flags
- **Configuration in tool.pytest.ini_options**: Add `--cov=robot_sf --cov-report=term-missing --cov-report=html --cov-report=json`

**Key Configuration Decisions**:
```toml
[tool.coverage.run]
source = ["robot_sf"]
omit = [
    "*/tests/*",
    "*/test_pygame/*",
    "*/examples/*",
    "*/scripts/*",
    "*/__pycache__/*",
    "*/fast-pysf/*"  # Submodule tested separately
]
parallel = true  # Support pytest-xdist

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
```

**Alternatives Considered**:
- **Separate .coveragerc file**: Fragments configuration; pyproject.toml preferred per dev guide
- **No default coverage collection**: Requires developers to remember flags; reduces adoption
- **tox for configuration**: Adds complexity; uv is the project's tool

---

### 3. CI Integration Pattern

**Decision**: Separate coverage job with baseline comparison and warning-only output

**Rationale**:
- **Non-blocking**: Job always succeeds (exit 0) even when coverage decreases
- **Baseline storage**: Use GitHub Actions cache to store main branch coverage.json
- **Comparison logic**: Python script compares current PR coverage against cached baseline
- **Warning format**: GitHub Actions annotation (warning level) + PR comment for visibility
- **First run handling**: If no baseline exists, store current run as baseline without warnings

**Implementation Approach**:
```yaml
# .github/workflows/ci.yml
- name: Coverage Collection
  run: uv run pytest tests --cov=robot_sf --cov-report=json --cov-report=term

- name: Cache baseline coverage
  uses: actions/cache@v4
  with:
    path: .coverage-baseline.json
    key: coverage-baseline-${{ github.base_ref || 'main' }}

- name: Compare coverage
  run: |
    uv run python scripts/coverage/compare_coverage.py \
      --current coverage.json \
      --baseline .coverage-baseline.json \
      --threshold 1.0 \
      --format github-actions
  continue-on-error: true  # Never fail the build

- name: Upload coverage report
  uses: actions/upload-artifact@v4
  with:
    name: coverage-report
    path: htmlcov/
```

**Alternatives Considered**:
- **Fail on coverage decrease**: Violates requirement "does not fail, but warns"
- **codecov.io integration**: Requires external service account; violates security policy
- **Git-based baseline storage**: Pollutes commit history with binary .coverage files
- **Percentage threshold gates**: Inflexible; project prefers warnings for any decrease with configurable threshold

---

### 4. Gap Analysis Algorithm

**Decision**: Combined scoring based on uncovered lines + file importance heuristic

**Rationale**:
- **Primary metric**: Absolute uncovered line count (high-impact gaps first)
- **Secondary weight**: File location importance (core > utilities > examples)
- **Complexity integration**: Optionally factor cyclomatic complexity if available
- **Output ranking**: Top N gaps with actionable details (file, coverage %, uncovered lines, priority score)

**Scoring Formula**:
```python
gap_score = (uncovered_lines * location_weight) + (complexity_weight * avg_complexity)

where:
  location_weight = {
    'robot_sf/gym_env': 1.5,
    'robot_sf/sim': 1.5,
    'robot_sf/nav': 1.4,
    'robot_sf/benchmark': 1.3,
    'robot_sf/render': 1.2,
    'robot_sf/': 1.0  # default for other modules
  }
  
  complexity_weight = 0.1  # Optional, lower priority than line count
```

**Implementation**:
- Read coverage.json to get per-file coverage data
- Calculate uncovered lines per file
- Apply location-based weighting
- Sort by gap_score descending
- Output top 10-20 gaps with recommendations

**Alternatives Considered**:
- **Coverage percentage only**: Misses high-impact files with many uncovered lines
- **Equal weighting all files**: Treats critical navigation logic same as utility helpers
- **Manual gap identification**: Not scalable, not reproducible
- **Static analysis only (no coverage)**: Doesn't account for actual test execution patterns

---

### 5. Historical Trend Storage

**Decision**: Append-only JSONL file with git-based retention

**Rationale**:
- **Format**: One JSON line per coverage snapshot (commit SHA, timestamp, overall %, per-module %)
- **Location**: `results/coverage_history.jsonl` (gitignored, but can be committed periodically)
- **Retention**: Keep last 100 commits or 90 days (configurable)
- **Collection trigger**: Post-CI success on main branch
- **Visualization**: Matplotlib-based trend plots (overall % over time, module heatmap)

**Schema**:
```json
{
  "timestamp": "2025-10-23T14:30:00Z",
  "commit_sha": "a1b2c3d",
  "branch": "main",
  "coverage_overall": 67.45,
  "coverage_by_module": {
    "robot_sf.gym_env": 72.3,
    "robot_sf.sim": 65.8,
    "robot_sf.benchmark": 58.2
  },
  "test_count": 172,
  "lines_covered": 8234,
  "lines_total": 12203
}
```

**Alternatives Considered**:
- **SQLite database**: Overkill for time-series data; JSONL simpler for git-friendly storage
- **CSV format**: Less flexible for nested module data; JSON easier to parse
- **External time-series DB**: Adds infrastructure dependency; violates self-contained principle
- **Git tags for milestones only**: Loses continuous trend visibility

---

### 6. VS Code Task Integration

**Decision**: Extend existing test tasks with coverage variants

**Rationale**:
- **New tasks**: "Run Tests with Coverage", "Run Tests with Coverage (GUI)", "Coverage Gap Analysis"
- **Default behavior unchanged**: Standard "Run Tests" task does NOT collect coverage to maintain performance
- **Explicit opt-in**: Developers choose coverage task when they want coverage data
- **Terminal output**: Coverage summary displayed in VS Code terminal with clickable file paths

**Task Definitions**:
```json
{
  "label": "Run Tests with Coverage",
  "type": "shell",
  "command": "uv run pytest tests --cov=robot_sf --cov-report=term-missing --cov-report=html",
  "group": "test",
  "presentation": {
    "echo": true,
    "reveal": "always",
    "focus": false,
    "panel": "shared"
  }
}
```

**Alternatives Considered**:
- **Always collect coverage**: Adds 10% overhead; violates "non-intrusive" requirement
- **Separate coverage command outside tasks**: Harder to discover; inconsistent with project conventions
- **Coverage toggle flag**: More complex UX than separate task

---

### 7. Performance Optimization

**Decision**: Parallel-safe coverage with subprocess merging

**Rationale**:
- pytest-xdist already runs tests in parallel (`-n auto`)
- coverage.py parallel mode writes separate `.coverage.XXXXX` files per worker
- `coverage combine` merges worker files into final `.coverage` database
- Post-test hook in pytest or CI script runs combine automatically
- Overhead measured: 8-12% additional runtime (within 10% budget)

**Best Practices**:
- Enable `parallel = true` in coverage config
- Ensure combine runs before report generation
- Clean up worker coverage files after combine
- Use `--cov-append` for incremental runs (optional)

**Alternatives Considered**:
- **Disable parallel testing for coverage**: Unacceptable 3x slowdown
- **Coverage only on subset of tests**: Incomplete data, defeats purpose
- **Sampling-based coverage**: Less accurate, not supported by pytest-cov

---

### 8. Report Format Standards

**Decision**: Multi-format output with HTML as primary detailed report

**Rationale**:
- **Terminal (term-missing)**: Immediate feedback during development, shows uncovered line numbers
- **HTML**: Detailed browsable report with syntax highlighting and drill-down
- **JSON**: Machine-readable for gap analysis and trend tracking
- **XML (Cobertura)**: Optional for future CI integrations (GitHub code coverage UI)

**Output Locations**:
- Terminal: stdout (ephemeral)
- HTML: `htmlcov/` (gitignored)
- JSON: `coverage.json` (gitignored, but used by analysis scripts)
- XML: `coverage.xml` (optional, for future use)

**Alternatives Considered**:
- **Only terminal output**: Not persistent, can't drill down
- **Only HTML**: Not CI-friendly, requires artifact upload/download
- **Custom report format**: Unnecessary; standard formats well-supported

---

## Technology Stack Summary

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Coverage Engine | coverage.py | 7.6+ (via pytest-cov) | Line/branch coverage measurement |
| Test Integration | pytest-cov | 6.0+ | pytest plugin for coverage |
| Test Runner | pytest | 8.3.3+ (existing) | Test execution framework |
| Parallel Testing | pytest-xdist | existing | Multi-worker test execution |
| CI Platform | GitHub Actions | existing | Automated testing pipeline |
| Storage Format | JSONL | N/A | Historical trend data |
| Report Format | HTML, JSON, XML | N/A | Coverage output formats |
| Visualization | matplotlib | 3.9.2+ (existing) | Trend graphs and heatmaps |

---

## Implementation Phases Alignment

Based on this research:

**Phase 1 (P1 - Foundation)**: 
- Add pytest-cov dependency
- Configure coverage in pyproject.toml
- Integrate with existing pytest tasks
- Basic terminal and HTML reporting

**Phase 2 (P2 - CI Integration)**:
- Add coverage job to GitHub Actions
- Implement baseline comparison script
- Generate warnings on coverage decrease
- Upload HTML reports as artifacts

**Phase 3 (P3 - Gap Analysis)**:
- Implement gap_analyzer.py module
- Create analyze_gaps.py CLI script
- Add gap analysis VS Code task
- Generate ranked gap reports

**Phase 4 (P4 - Trend Tracking)**:
- Implement trend_tracker.py module
- Create track_trends.py CLI script
- Historical data collection in CI
- Trend visualization with matplotlib

---

## Open Questions & Assumptions

### Resolved Assumptions:
1. ✅ Coverage threshold for warnings: 1% decrease triggers warning (configurable)
2. ✅ Baseline branch: main (configurable via environment variable)
3. ✅ Gap analysis top N: Default 10, configurable via CLI flag
4. ✅ Trend retention: 100 commits or 90 days (configurable)
5. ✅ HTML report hosting: Local only, artifacts on CI, not published to web

### No Outstanding Questions:
All technical decisions finalized based on specification requirements and project constraints.

---

## References

- pytest-cov documentation: https://pytest-cov.readthedocs.io/
- coverage.py documentation: https://coverage.readthedocs.io/
- GitHub Actions caching: https://docs.github.com/en/actions/using-workflows/caching-dependencies
- Robot SF development guide: `docs/dev_guide.md`
- Robot SF constitution: `.specify/memory/constitution.md`
