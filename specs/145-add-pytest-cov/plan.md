# Implementation Plan: Code Coverage Monitoring and Quality Tracking

**Branch**: `145-add-pytest-cov` | **Date**: 2025-10-23 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/145-add-pytest-cov/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Add pytest-cov integration to enable non-intrusive code coverage collection, CI/CD coverage regression warnings, automated gap analysis, and historical quality metrics tracking. The implementation must integrate seamlessly with existing pytest workflows, VS Code tasks, and GitHub Actions CI without breaking builds when coverage decreases. Coverage tracking focuses on the `robot_sf/` library code, providing actionable insights for improving test quality over time while maintaining the project's reproducibility and deterministic testing principles.

## Technical Context

**Language/Version**: Python 3.11+ (verified in pyproject.toml)  
**Primary Dependencies**: pytest 8.3.3+, pytest-cov (to be added), coverage.py (via pytest-cov)  
**Storage**: Filesystem-based (.coverage database, coverage.json, HTML reports); Git cache for CI baseline  
**Testing**: pytest with parallel execution (pytest-xdist already in use per CI workflow)  
**Target Platform**: Cross-platform (Linux CI, macOS/Windows local development)  
**Project Type**: Single Python library project (`robot_sf/`) with examples, scripts, and test suites  
**Performance Goals**: Coverage collection overhead < 10% of test runtime (approximately 18-20 seconds added to ~3 minute test suite)  
**Constraints**: Non-blocking CI (warnings only), no configuration required for default use, headless execution support  
**Scale/Scope**: ~15k-20k lines in robot_sf/ library, ~170 existing tests, 3 test suites (tests/, test_pygame/, fast-pysf/tests)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Reproducible Social Navigation Research Core
✅ **PASS** - Coverage tracking enhances reproducibility by documenting which code paths are exercised during validation and benchmarking. Deterministic seeds and configurations remain unchanged. Coverage data collection does not alter test execution behavior.

### Principle II: Factory-Based Environment Abstraction
✅ **PASS** - No changes to environment factory signatures or public interfaces. Coverage monitoring is orthogonal to environment design and operates at the test execution layer.

### Principle III: Benchmark & Metrics First
✅ **PASS** - Coverage metrics complement existing quality metrics. Gap analysis and trend tracking align with the metrics-first philosophy. No changes to benchmark runner or episode schema.

### Principle IV: Unified Configuration & Deterministic Seeds
✅ **PASS** - Coverage configuration added to pyproject.toml and pytest.ini without disrupting existing config hierarchy. Seed propagation and determinism unaffected by coverage collection.

### Principle V: Minimal, Documented Baselines
✅ **PASS** - No impact on baseline planners. Coverage data collected during baseline benchmarks provides visibility into baseline code path coverage.

### Principle VI: Metrics Transparency & Statistical Rigor
✅ **PASS** - Coverage metrics are transparent (line/branch coverage percentages), deterministic (same code + same tests = same coverage), and include contextual detail (per-file, per-module breakdowns).

### Principle VII: Backward Compatibility & Evolution Gates
✅ **PASS** - Purely additive feature. No breaking changes to existing APIs, schemas, or workflows. VS Code tasks extended but not modified. CI enhanced with optional coverage step that doesn't alter existing test job contracts.

### Principle VIII: Documentation as an API Surface
✅ **PASS** - Implementation plan includes documentation requirements: quickstart guide, gap analysis usage, CI integration guide, and updates to dev_guide.md. Coverage tooling will be documented with clear purpose and assumptions.

### Principle IX: Test Coverage for Public Behavior
✅ **PASS** - Meta-requirement satisfied: this feature adds tooling to measure test coverage itself. Smoke tests will validate coverage collection, CI integration, and gap analysis functionality.

### Principle X: Scope Discipline
✅ **PASS** - Coverage monitoring directly supports the repository's testing and quality goals without expanding into out-of-scope areas. Focused exclusively on test quality visibility for the social navigation simulation platform.

### Principle XI: Library Reuse & Helper Documentation
✅ **PASS** - Gap analysis and trend tracking utilities will be implemented as reusable modules in `robot_sf/coverage_tools/` (or similar) following the library-first principle. Scripts will orchestrate these helpers. All helpers will include docstrings explaining purpose, decision rules, and side effects.

### Principle XII: Preferred Logging & Observability
✅ **PASS** - Coverage tools will use Loguru for runtime messaging (warnings about missing baselines, gap analysis progress, trend computation). No print() in library code. CLI scripts may use print for coverage summaries as they are explicit user-facing tools.

### **Overall Gate Status: ✅ PASSED**

All constitutional principles satisfied. No violations requiring justification. Feature is aligned with repository governance and quality standards.

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```
robot_sf/
├── coverage_tools/          # NEW: Coverage analysis utilities (library-first principle)
│   ├── __init__.py
│   ├── gap_analyzer.py      # Identifies and ranks coverage gaps
│   ├── trend_tracker.py     # Historical coverage metrics storage and analysis
│   ├── baseline_comparator.py  # CI coverage comparison logic
│   └── report_formatter.py  # Terminal and file report generation
│
├── gym_env/                 # Existing - no changes
├── sim/                     # Existing - no changes
├── nav/                     # Existing - no changes
├── render/                  # Existing - no changes
└── benchmark/               # Existing - no changes

scripts/
├── coverage/                # NEW: Coverage tool entry points
│   ├── analyze_gaps.py      # CLI for gap analysis
│   ├── track_trends.py      # CLI for trend visualization
│   └── compare_coverage.py  # CI helper for baseline comparison
└── [existing scripts...]

tests/
├── coverage_tools/          # NEW: Tests for coverage utilities
│   ├── test_gap_analyzer.py
│   ├── test_trend_tracker.py
│   ├── test_baseline_comparator.py
│   └── test_report_formatter.py
├── [existing test modules...]
└── conftest.py              # May add coverage fixtures

.github/
└── workflows/
    └── ci.yml               # MODIFIED: Add coverage collection and comparison step

configs/                     # Existing - no changes needed
docs/
├── coverage_guide.md        # NEW: Coverage tooling documentation
└── dev_guide.md             # MODIFIED: Add coverage workflow section

pyproject.toml               # MODIFIED: Add pytest-cov dependency and coverage config
pytest.ini or .coveragerc    # NEW: Coverage configuration (omit patterns, source)
.gitignore                   # MODIFIED: Add coverage output directories
.vscode/
└── tasks.json               # MODIFIED: Add coverage-enabled test tasks
```

**Structure Decision**: 
This follows the existing single Python project structure. Coverage analysis utilities are placed in `robot_sf/coverage_tools/` as reusable library modules (Principle XI), with orchestration scripts in `scripts/coverage/`. Tests for new functionality follow the existing pattern in `tests/coverage_tools/`. Configuration changes are minimal and localized to standard files (pyproject.toml, pytest.ini). This maintains consistency with the repository's established organization while adhering to the library-first principle.

## Complexity Tracking

*No violations - section omitted per template guidance. All constitutional gates passed.*
