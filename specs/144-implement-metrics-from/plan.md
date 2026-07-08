# Implementation Plan: Metrics from Paper 2306.16740v4

**Branch**: `144-implement-metrics-from` | **Date**: October 23, 2025 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/144-implement-metrics-from/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement 22 social navigation metrics from paper 2306.16740v4 (Table 1) to enable standardized evaluation of robot navigation algorithms. The implementation extends the existing `robot_sf/benchmark/metrics.py` module with both NHT (Navigation/Hard Task) and SHT (Social/Human-aware Task) metrics, maintaining compatibility with the current `episode.schema.v1.json` format and benchmark infrastructure.

## Technical Context

**Language/Version**: Python 3.13.1 (as per dev_guide.md)
**Primary Dependencies**: NumPy (numerical computation), pytest (testing), existing robot_sf.benchmark.metrics module
**Storage**: Append-only JSONL episode records (robot_sf/benchmark/schemas/episode.schema.v1.json)
**Testing**: pytest with unit tests in tests/ and integration tests for metric computation
**Target Platform**: Cross-platform (Linux, macOS) with headless execution support
**Project Type**: Single project - extends existing robot_sf library
**Performance Goals**: < 100ms per episode for up to 50 pedestrians; batch aggregation of 1000+ episodes in < 30 seconds
**Constraints**: 
  - Must maintain backward compatibility with episode.schema.v1.json
  - No new external dependencies beyond current project requirements
  - < 10% overhead on existing metric computation
  - Must integrate with existing EpisodeData dataclass pattern
**Scale/Scope**: 22 new metric functions (11 NHT + 11 SHT) extending existing ~15 metrics in metrics.py

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Reproducible Social Navigation Research Core
✅ **PASS** - Metrics are deterministic functions of episode data; support reproducible benchmarking

### Principle II: Factory-Based Environment Abstraction
✅ **PASS** - No changes to environment factories; metrics operate on post-episode data

### Principle III: Benchmark & Metrics First
✅ **PASS** - Core purpose is extending benchmark metrics suite with paper-standard measures

### Principle IV: Unified Configuration & Deterministic Seeds
✅ **PASS** - Metrics are pure functions; no new configuration needed beyond episode data

### Principle V: Minimal, Documented Baselines
✅ **PASS** - No new baselines; metrics evaluate existing baselines

### Principle VI: Metrics Transparency & Statistical Rigor
✅ **PASS** - Each metric documented with units, ranges, formulas; supports aggregation with CIs

### Principle VII: Backward Compatibility & Evolution Gates
✅ **PASS** - Extends existing metrics.py without breaking changes; uses existing schema

### Principle VIII: Documentation as an API Surface
⚠️ **REQUIRES ATTENTION** - Must add metric documentation to docs/benchmark.md and update docs/README.md index

### Principle IX: Test Coverage for Public Behavior
⚠️ **REQUIRES ATTENTION** - Must add unit tests for all 22 new metrics (normal + edge cases)

### Principle X: Scope Discipline
✅ **PASS** - Directly supports core social navigation evaluation; no scope creep

### Principle XI: Library Reuse & Helper Documentation
✅ **PASS** - Follows existing EpisodeData pattern; each metric function includes docstring with purpose, units, edge cases

### Principle XII: Preferred Logging & Observability
✅ **PASS** - Metrics are pure computation functions; no logging needed (return NaN for invalid cases)

**Overall Status**: ✅ CONDITIONALLY APPROVED - Proceed with documentation and testing requirements documented above

## Project Structure

### Documentation (this feature)

```
specs/144-implement-metrics-from/
├── plan.md              # This file (/speckit.plan command output)
├── spec.md              # Feature specification (completed)
├── checklists/
│   └── requirements.md  # Quality validation checklist
├── research.md          # Phase 0 output (metric computation patterns, edge case handling)
├── data-model.md        # Phase 1 output (EpisodeData extensions if needed)
├── quickstart.md        # Phase 1 output (usage examples)
├── contracts/           # Phase 1 output (metric function signatures)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```
robot_sf/
├── benchmark/
│   ├── metrics.py           # EXTEND: Add 22 new metric functions
│   ├── constants.py         # MAY EXTEND: Add metric-specific thresholds if needed
│   └── schemas/
│       └── episode.schema.v1.json  # NO CHANGE: Existing schema supports new metrics
│
tests/
├── test_metrics.py          # EXTEND: Add unit tests for new metrics
├── test_benchmark_integration.py  # MAY EXTEND: Integration tests for batch computation
└── fixtures/
    └── episode_data.py      # MAY ADD: Synthetic test cases for metric validation

docs/
├── benchmark.md             # UPDATE: Document new metrics with formulas
├── README.md                # UPDATE: Add link to metrics documentation
└── dev/
    └── issues/
        └── 144-implement-metrics-from/  # SYMLINK or reference to specs/
```

**Structure Decision**: Single project extension - all changes localized to `robot_sf/benchmark/metrics.py` with supporting tests and documentation. No new modules or packages needed; follows existing pattern of metric functions operating on `EpisodeData` dataclass.

## Complexity Tracking

*No violations - all Constitution checks passed or documented*

This feature maintains the repository's focus on social navigation metrics (Principle III) and follows established patterns for metric implementation. No additional complexity introduced beyond the inherent complexity of implementing 22 metric formulas from the paper.

---

## Phase 0: Research Completed ✅

**Artifacts Generated**:
- [research.md](./research.md) - Comprehensive research on implementation patterns

**Key Decisions**:
1. **EpisodeData Extensions**: Add two optional fields (`obstacles`, `other_agents_pos`) for collision detection
2. **Metric Categories**: Organized into 5 implementation categories by data requirements
3. **Edge Case Strategy**: Semantic-based pattern (counts→0.0, distances→NaN, ratios→limit)
4. **Computation Helpers**: Internal functions for pedestrian velocities, jerk, distance matrix
5. **Performance**: One-function-per-metric pattern with vectorized NumPy operations

**All NEEDS CLARIFICATION Resolved**: ✅
- EpisodeData sufficiency analysis complete
- Collision detection strategy defined
- Timeout/progress tracking approach established
- Velocity/jerk computation patterns specified
- Edge case handling patterns documented
- Performance optimization strategy outlined
- Integration approach confirmed

---

## Phase 1: Design & Contracts Completed ✅

**Artifacts Generated**:
1. [data-model.md](./data-model.md) - EpisodeData extensions and metric catalog
2. [contracts/metric_signatures.md](./contracts/metric_signatures.md) - API contracts for all 22 metrics
3. [quickstart.md](./quickstart.md) - User guide and usage examples
4. Agent context updated via `.specify/scripts/bash/update-agent-context.sh copilot`

**Data Model Summary**:
- Extended EpisodeData with 2 optional fields (backward compatible)
- Defined 22 metric function signatures with types
- Documented validation rules and state transitions
- Cataloged metrics by category (NHT/SHT) with units and ranges

**Contract Highlights**:
- All metrics are pure functions (deterministic, no side effects)
- Comprehensive docstrings with formulas, units, edge cases, paper references
- Clear behavioral contracts for each metric
- Type hints for all parameters and returns
- Performance guarantee: O(T*K) maximum complexity

**Constitution Re-Check After Phase 1**: ✅ APPROVED

All principles still satisfied:
- Documentation artifacts created (Principle VIII) → quickstart.md, contracts/
- Test coverage requirements documented (Principle IX) → ready for Phase 2 implementation
- Library patterns followed (Principle XI) → EpisodeData extension, helper functions
- No scope violations (Principle X)

---

## Next Steps: Phase 2 - Implementation Tasks

**Command**: Run `.specify/scripts/bash/create-tasks.sh` to generate `tasks.md`

**Expected Task Breakdown**:
1. Extend EpisodeData dataclass with optional fields
2. Implement 22 metric functions (5 categories)
3. Add internal helper functions (_compute_ped_velocities, _compute_jerk, etc.)
4. Write unit tests for all metrics (normal + edge cases)
5. Add integration tests for batch computation
6. Update documentation (docs/benchmark.md, docs/README.md)
7. Performance validation (< 100ms target)
8. Update CHANGELOG.md

**Ready for Implementation**: All design decisions made, contracts defined, edge cases documented.
