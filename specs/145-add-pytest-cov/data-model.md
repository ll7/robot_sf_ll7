# Data Model: Code Coverage Monitoring and Quality Tracking

**Feature**: pytest-cov integration for robot_sf  
**Date**: 2025-10-23  
**Status**: Complete

## Purpose

This document defines the data structures for coverage collection, baseline comparison, gap analysis, and historical trend tracking.

## Core Entities

### 1. CoverageSnapshot

Represents coverage data from a single test run.

**Fields**:
- `timestamp`: ISO 8601 datetime string (when coverage was collected)
- `commit_sha`: Git commit hash (40-char hex string)
- `branch`: Git branch name (string)
- `coverage_overall`: Overall coverage percentage (float, 0.0-100.0, precision 2)
- `coverage_by_module`: Dict mapping module path to coverage percentage
  - Key: Module path string (e.g., "robot_sf.gym_env")
  - Value: Coverage percentage (float, 0.0-100.0)
- `coverage_by_file`: Dict mapping file path to detailed coverage data
  - Key: Relative file path (e.g., "robot_sf/gym_env/env_config.py")
  - Value: FileCoverage object (see below)
- `test_count`: Number of tests executed (int)
- `lines_covered`: Total lines covered (int)
- `lines_total`: Total executable lines (int)
- `execution_time_seconds`: Test suite runtime (float)

**Relationships**:
- Contains multiple FileCoverage instances
- Part of CoverageTrend time series
- Compared against CoverageBaseline

**Validation Rules**:
- `coverage_overall` must be between 0.0 and 100.0
- `lines_covered` <= `lines_total`
- `coverage_overall` ≈ (lines_covered / lines_total) * 100 (within 0.1%)
- `timestamp` must be valid ISO 8601 format
- `commit_sha` must be 40 hex characters (if provided)

**State Transitions**:
N/A (immutable snapshot)

**Example**:
```json
{
  "timestamp": "2025-10-23T14:30:00Z",
  "commit_sha": "a1b2c3d4e5f6789012345678901234567890abcd",
  "branch": "main",
  "coverage_overall": 67.45,
  "coverage_by_module": {
    "robot_sf.gym_env": 72.3,
    "robot_sf.sim": 65.8,
    "robot_sf.benchmark": 58.2
  },
  "test_count": 172,
  "lines_covered": 8234,
  "lines_total": 12203,
  "execution_time_seconds": 185.4
}
```

---

### 2. FileCoverage

Detailed coverage information for a single source file.

**Fields**:
- `file_path`: Relative path from repository root (string)
- `coverage_percent`: Coverage percentage for this file (float, 0.0-100.0)
- `lines_covered`: Number of covered lines (int)
- `lines_total`: Number of executable lines (int)
- `missing_lines`: List of uncovered line numbers (list of int)
- `excluded_lines`: List of excluded line numbers (list of int, optional)
- `branch_coverage_percent`: Branch coverage percentage (float, optional)

**Relationships**:
- Belongs to one CoverageSnapshot
- Analyzed by GapAnalyzer to produce CoverageGap

**Validation Rules**:
- `lines_covered` <= `lines_total`
- `coverage_percent` ≈ (lines_covered / lines_total) * 100
- `missing_lines` are within file line count
- No overlap between `missing_lines` and line numbers in [1, lines_total]

**Example**:
```json
{
  "file_path": "robot_sf/gym_env/env_config.py",
  "coverage_percent": 45.8,
  "lines_covered": 87,
  "lines_total": 190,
  "missing_lines": [23, 24, 45, 67, 89, 90, 91, ...],
  "excluded_lines": [1, 2, 150],
  "branch_coverage_percent": 38.5
}
```

---

### 3. CoverageBaseline

Reference coverage data from the main/default branch for comparison.

**Fields**:
- `snapshot`: CoverageSnapshot from baseline branch
- `baseline_branch`: Name of baseline branch (e.g., "main")
- `created_at`: When baseline was captured (ISO 8601)
- `cache_key`: Unique identifier for baseline cache (string)
- `is_valid`: Whether baseline is current (bool)

**Relationships**:
- References one CoverageSnapshot
- Compared against current CoverageSnapshot by BaselineComparator

**Validation Rules**:
- `snapshot` must be a valid CoverageSnapshot
- `baseline_branch` must match repository's default branch
- `is_valid` = True only if baseline commit is ancestor of current HEAD

**State Transitions**:
- Created: New baseline established (first run or cache miss)
- Refreshed: Updated when baseline branch coverage changes
- Invalidated: Marked invalid when baseline branch moves ahead significantly

---

### 4. CoverageDelta

Represents the difference between current coverage and baseline.

**Fields**:
- `overall_change`: Overall coverage percentage change (float, can be negative)
- `module_changes`: Dict mapping module path to coverage change
  - Key: Module path (string)
  - Value: Coverage change (float, can be negative)
- `file_changes`: Dict mapping file path to coverage change
  - Key: File path (string)
  - Value: Coverage change (float, can be negative)
- `new_files`: List of files added since baseline (list of string)
- `removed_files`: List of files removed since baseline (list of string)
- `improved_files`: List of files with coverage increase (list of tuple: (path, change))
- `degraded_files`: List of files with coverage decrease (list of tuple: (path, change))

**Relationships**:
- Calculated from CoverageSnapshot (current) and CoverageBaseline
- Used by warning generation in CI

**Validation Rules**:
- Changes should be consistent: sum of file changes should approximate overall change
- `improved_files` and `degraded_files` must not overlap
- Files in `new_files` should not be in baseline snapshot

**Example**:
```json
{
  "overall_change": -2.3,
  "module_changes": {
    "robot_sf.gym_env": -5.2,
    "robot_sf.sim": +1.1
  },
  "file_changes": {
    "robot_sf/gym_env/env_config.py": -8.4,
    "robot_sf/sim/simulator.py": +3.2
  },
  "new_files": ["robot_sf/coverage_tools/gap_analyzer.py"],
  "removed_files": [],
  "improved_files": [("robot_sf/sim/simulator.py", 3.2)],
  "degraded_files": [("robot_sf/gym_env/env_config.py", -8.4)]
}
```

---

### 5. CoverageGap

Represents an identified gap in test coverage.

**Fields**:
- `file_path`: Path to file with coverage gap (string)
- `coverage_percent`: Current coverage percentage (float, 0.0-100.0)
- `uncovered_lines`: Number of uncovered lines (int)
- `missing_line_ranges`: List of uncovered line number ranges (list of tuple: (start, end))
- `priority_score`: Calculated priority for addressing gap (float, higher = more important)
- `location_weight`: Importance weight based on file location (float, 1.0-2.0)
- `module_path`: Parent module path (string, e.g., "robot_sf.gym_env")
- `recommendation`: Suggested action for addressing gap (string)

**Relationships**:
- Derived from FileCoverage by GapAnalyzer
- Multiple gaps sorted by priority in gap analysis report

**Validation Rules**:
- `priority_score` = (uncovered_lines * location_weight) + complexity_adjustment
- `uncovered_lines` > 0 (otherwise not a gap)
- `coverage_percent` < 100.0
- `missing_line_ranges` non-empty

**State Transitions**:
N/A (derived data, regenerated each analysis)

**Example**:
```json
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
```

---

### 6. CoverageTrend

Historical trend data for coverage metrics over time.

**Fields**:
- `trend_id`: Unique identifier for this trend series (string)
- `snapshots`: List of CoverageSnapshot objects in chronological order
- `start_date`: First snapshot timestamp (ISO 8601)
- `end_date`: Last snapshot timestamp (ISO 8601)
- `trend_direction`: Overall trend direction (enum: "improving", "stable", "degrading")
- `trend_rate`: Average coverage change per week (float, percentage points/week)
- `retention_commits`: Maximum commits to retain (int, default 100)
- `retention_days`: Maximum days to retain (int, default 90)

**Relationships**:
- Contains multiple CoverageSnapshot instances
- Analyzed to produce trend visualizations

**Validation Rules**:
- `snapshots` must be sorted by timestamp ascending
- `trend_direction` determined by linear regression slope:
  - "improving" if slope > +0.1% per week
  - "degrading" if slope < -0.1% per week
  - "stable" otherwise
- Only retain snapshots within retention limits

**State Transitions**:
- Extended: New snapshot appended
- Pruned: Old snapshots removed based on retention policy
- Analyzed: Trend statistics recalculated

**Example**:
```json
{
  "trend_id": "main-branch-overall",
  "snapshots": [ /* list of CoverageSnapshot objects */ ],
  "start_date": "2025-09-23T00:00:00Z",
  "end_date": "2025-10-23T14:30:00Z",
  "trend_direction": "improving",
  "trend_rate": 0.3,
  "retention_commits": 100,
  "retention_days": 90
}
```

---

## Configuration Entities

### 7. CoverageConfig

Configuration for coverage collection and reporting.

**Fields**:
- `source_paths`: List of source directories to measure (list of string)
- `omit_patterns`: List of glob patterns to exclude (list of string)
- `report_formats`: List of report formats to generate (list of enum: "term", "html", "json", "xml")
- `parallel_mode`: Enable parallel coverage collection (bool)
- `branch_coverage`: Enable branch coverage measurement (bool)
- `precision`: Decimal places for percentages (int, default 2)
- `show_missing`: Show missing line numbers in reports (bool)
- `exclude_lines`: List of regex patterns for lines to exclude (list of string)

**Source**: Loaded from `pyproject.toml` `[tool.coverage.*]` sections

**Validation Rules**:
- `source_paths` must exist and be directories
- `precision` must be 0-4
- `report_formats` must be non-empty

---

### 8. GapAnalysisConfig

Configuration for coverage gap analysis.

**Fields**:
- `top_n_gaps`: Number of top gaps to report (int, default 10)
- `min_uncovered_lines`: Minimum uncovered lines to qualify as gap (int, default 5)
- `location_weights`: Dict mapping path prefix to importance weight
- `complexity_threshold`: Minimum complexity to include in scoring (int, optional)
- `output_format`: Format for gap report (enum: "terminal", "json", "markdown")

**Validation Rules**:
- `top_n_gaps` > 0
- `min_uncovered_lines` >= 1
- Location weights must be positive floats

---

### 9. TrendConfig

Configuration for historical trend tracking.

**Fields**:
- `storage_path`: Path to JSONL file for trend data (string)
- `retention_commits`: Maximum commits to retain (int, default 100)
- `retention_days`: Maximum days to retain (int, default 90)
- `collection_branch`: Branch to collect trends from (string, default "main")
- `auto_collect`: Automatically collect on CI success (bool, default true)
- `visualization_format`: Format for trend graphs (enum: "png", "pdf", "svg")

**Validation Rules**:
- `storage_path` parent directory must exist
- Retention limits must be positive
- `collection_branch` should be a valid branch name

---

## Derived Data Structures

### 10. GapAnalysisReport

Output of gap analysis containing ranked gaps and summary.

**Fields**:
- `generated_at`: Report generation timestamp (ISO 8601)
- `snapshot_timestamp`: Coverage snapshot timestamp being analyzed (ISO 8601)
- `overall_coverage`: Overall coverage percentage (float)
- `total_gaps_identified`: Total number of gaps found (int)
- `top_gaps`: List of CoverageGap objects, sorted by priority descending (top N)
- `summary_stats`: Dict with summary statistics
  - `total_uncovered_lines`: Sum of uncovered lines across all gaps
  - `avg_gap_priority`: Average priority score
  - `core_module_gaps`: Count of gaps in core modules
- `recommendations`: List of high-level recommendations (list of string)

---

### 11. TrendVisualization

Configuration for generating trend graphs.

**Fields**:
- `trend_data`: CoverageTrend object to visualize
- `plot_type`: Type of visualization (enum: "line", "heatmap", "comparison")
- `time_range`: Time range to plot (enum: "all", "last_30_days", "last_100_commits")
- `modules_to_plot`: List of module paths to include (list of string, or "all")
- `output_path`: Path to save visualization (string)
- `title`: Graph title (string)
- `show_annotations`: Show significant events on graph (bool)

---

## Data Flow Diagrams

### Coverage Collection Flow
```
Test Execution
    ↓
coverage.py measures line execution
    ↓
pytest-cov plugin collects coverage data
    ↓
Worker coverage files (.coverage.XXXXX)
    ↓
coverage combine merges workers
    ↓
.coverage database (SQLite)
    ↓
coverage report/html/json generates outputs
    ↓
CoverageSnapshot created from coverage.json
```

### Gap Analysis Flow
```
coverage.json
    ↓
GapAnalyzer.load_coverage()
    ↓
Parse FileCoverage for each file
    ↓
Filter files with uncovered_lines > threshold
    ↓
Calculate priority_score for each gap
    ↓
Sort by priority descending
    ↓
Take top N gaps
    ↓
Generate GapAnalysisReport
    ↓
Format and output report
```

### Trend Tracking Flow
```
CI Test Success (main branch)
    ↓
TrendTracker.collect_snapshot()
    ↓
Create CoverageSnapshot from coverage.json
    ↓
Load existing CoverageTrend from JSONL
    ↓
Append new snapshot
    ↓
Apply retention policy (prune old snapshots)
    ↓
Calculate trend statistics
    ↓
Write updated JSONL
    ↓
Optional: Generate TrendVisualization
```

### Baseline Comparison Flow (CI)
```
PR Test Run
    ↓
Collect current CoverageSnapshot
    ↓
Load CoverageBaseline from cache
    ↓
BaselineComparator.compare(current, baseline)
    ↓
Calculate CoverageDelta
    ↓
Check if overall_change < -threshold
    ↓
If decrease detected:
    ↓
Generate warning message with details
    ↓
Output GitHub Actions annotation
    ↓
Post PR comment (optional)
    ↓
Exit 0 (non-blocking)
```

---

## Persistence Strategy

### CoverageSnapshot
- **Format**: JSON (in coverage.json from coverage.py)
- **Storage**: Ephemeral (gitignored), regenerated each run
- **Serialization**: Standard JSON encoding

### CoverageBaseline
- **Format**: JSON
- **Storage**: GitHub Actions cache (`.coverage-baseline.json`)
- **Lifetime**: Cache invalidation on baseline branch update
- **Key**: `coverage-baseline-${{ github.base_ref }}`

### CoverageTrend
- **Format**: JSONL (one JSON object per line)
- **Storage**: `results/coverage_history.jsonl` (gitignored, but can be committed)
- **Retention**: Prune automatically based on retention policy
- **Append-only**: New snapshots appended, old lines removed during prune

### GapAnalysisReport
- **Format**: JSON or Markdown
- **Storage**: Ephemeral (stdout or temp file)
- **Can be saved**: Optional `--output` flag to persist

---

## API Contracts (Internal Library)

While this is a library-level feature, key internal APIs for reusability:

### GapAnalyzer API
```python
class GapAnalyzer:
    def __init__(self, config: GapAnalysisConfig):
        """Initialize gap analyzer with configuration."""
    
    def load_coverage(self, coverage_json_path: str) -> CoverageSnapshot:
        """Load coverage data from JSON file."""
    
    def identify_gaps(self, snapshot: CoverageSnapshot) -> list[CoverageGap]:
        """Identify coverage gaps and return sorted by priority."""
    
    def generate_report(self, gaps: list[CoverageGap]) -> GapAnalysisReport:
        """Generate gap analysis report from identified gaps."""
    
    def format_report(self, report: GapAnalysisReport, format: str) -> str:
        """Format report for output (terminal, JSON, markdown)."""
```

### TrendTracker API
```python
class TrendTracker:
    def __init__(self, config: TrendConfig):
        """Initialize trend tracker with configuration."""
    
    def load_trend(self, storage_path: str) -> CoverageTrend:
        """Load existing trend data from JSONL file."""
    
    def append_snapshot(self, trend: CoverageTrend, snapshot: CoverageSnapshot) -> None:
        """Append new snapshot to trend."""
    
    def apply_retention(self, trend: CoverageTrend) -> None:
        """Prune old snapshots based on retention policy."""
    
    def save_trend(self, trend: CoverageTrend, storage_path: str) -> None:
        """Save trend data to JSONL file."""
    
    def calculate_statistics(self, trend: CoverageTrend) -> dict:
        """Calculate trend statistics (direction, rate, etc.)."""
```

### BaselineComparator API
```python
class BaselineComparator:
    def __init__(self, threshold: float = 1.0):
        """Initialize comparator with warning threshold."""
    
    def load_baseline(self, baseline_path: str) -> CoverageBaseline:
        """Load baseline coverage from cache file."""
    
    def compare(self, current: CoverageSnapshot, baseline: CoverageBaseline) -> CoverageDelta:
        """Compare current coverage against baseline."""
    
    def generate_warning(self, delta: CoverageDelta, format: str) -> str:
        """Generate warning message in specified format."""
    
    def should_warn(self, delta: CoverageDelta) -> bool:
        """Determine if warning should be issued based on threshold."""
```

---

## Summary

This data model provides:
- ✅ **Snapshot-based coverage tracking** with full detail preservation
- ✅ **Baseline comparison** for CI regression detection
- ✅ **Gap analysis** with priority scoring and actionable insights
- ✅ **Historical trends** with retention policy and trend statistics
- ✅ **Configuration flexibility** through typed config objects
- ✅ **Clear relationships** between entities for analysis flows
- ✅ **Validation rules** ensuring data integrity
- ✅ **Multiple output formats** for different consumption needs

All entities align with the Robot SF constitution principles:
- Reproducible (deterministic from inputs)
- Transparent (clear metrics and calculations)
- Well-documented (purpose and constraints specified)
- Self-contained (no external dependencies)
