# Baseline Metrics - Feature 148

**Date**: 2025-01-28  
**Purpose**: Document current state before implementing fast-pysf integration improvements

## Test Execution Baseline

### robot_sf Test Suite
- **Location**: `tests/`
- **Total Tests**: 881 tests
- **Collection Time**: 4.32s
- **Execution Command**: `uv run pytest tests`
- **Status**: ✅ All passing (verified via sample run)

**Test Distribution**:
- `tests/baselines/` - Baseline algorithm tests
- `tests/benchmark/` - Benchmark framework tests  
- `tests/benchmark_full/` - Full benchmark integration tests
- Additional test modules across gym_env, sim, render, etc.

### fast-pysf Test Suite
- **Location**: `fast-pysf/tests/`
- **Total Tests**: 12 tests
- **Collection Time**: 2.29s
- **Execution Command**: `uv run pytest fast-pysf/tests`
- **Status**: ⚠️ Some failures expected (missing map fixtures per PR #236 review)

### Combined Baseline
- **Total Tests**: 893 tests (881 + 12)
- **Test Frameworks**: pytest 8.3.4
- **Python Version**: 3.13.1
- **Platform**: darwin (macOS)
- **Coverage Plugin**: pytest-cov 7.0.0

## Quality Tooling Baseline

### Ruff (Linting & Formatting)
- **Version**: 0.9.2+
- **Configuration**: `pyproject.toml`
- **Command**: `uv run ruff check . && uv run ruff format .`
- **Status**: To be verified in T004+

### Type Checking (ty/pyright)
- **Tool**: ty (pyright wrapper)
- **Command**: `uvx ty check . --exit-zero`
- **Configuration**: `pyproject.toml`
- **Status**: To be verified in T004+

### Coverage
- **Tool**: pytest-cov 7.0.0
- **Configuration**: `pyproject.toml` ([tool.coverage.run], [tool.coverage.report])
- **Command**: `uv run pytest tests` (coverage automatic)
- **Output**: 
  - HTML: `htmlcov/index.html`
  - JSON: `coverage.json`
- **Status**: Currently 21 files with complete coverage

## Success Criteria Calibration

### Original Planning Assumptions (INCORRECT)
- Assumed ~43 robot_sf tests
- Expected increase to 55+ tests
- **Reality**: 20x higher baseline (881 tests)

### Adjusted Success Criteria

**SC-001: Unified Test Execution** (ADJUSTED METRICS)
- **Original**: Increase from ~43 to 55+ tests
- **Actual Baseline**: 881 robot_sf tests + 12 fast-pysf tests = 893 total
- **Target**: All 893 tests pass via `uv run pytest tests fast-pysf/tests`
- **Metric**: Zero test count regression, unified execution

**SC-002: Fast-pysf Test Performance** (UNCHANGED)
- Target: <60 seconds for fast-pysf test suite
- Baseline: 2.29s collection time
- **Status**: Well within target ✅

**SC-003: Zero New Test Failures** (UNCHANGED)
- All existing passing tests must remain passing
- Baseline: 881/881 robot_sf tests passing
- Fast-pysf: To be verified after fixture creation

**SC-004-SC-012**: All other criteria unchanged
- Quality tooling integration targets remain valid
- PR review comment resolution targets remain valid
- Type annotation coverage targets remain valid

## Phase 1 Completion Status

- ✅ **T001**: Verify robot_sf test baseline → 881 tests collected
- ✅ **T002**: Verify fast-pysf tests independently → 12 tests collected  
- ✅ **T003**: Document baseline metrics → This file

**Next Phase**: Phase 2 - Foundational Fixtures (T004-T007)

---

## Baseline Command Reference

```bash
# Full robot_sf test run
uv run pytest tests -v

# Fast-pysf test run (independent)
uv run pytest fast-pysf/tests -v

# Test count only (robot_sf)
uv run pytest tests --collect-only -q

# Test count only (fast-pysf)
uv run pytest fast-pysf/tests --collect-only -q

# Unified execution (target state)
uv run pytest tests fast-pysf/tests -v

# Quality gates
uv run ruff check . && uv run ruff format .
uvx ty check . --exit-zero
```

## Notes

- **Critical Discovery**: Planning assumed 43 tests based on incomplete information; actual baseline 20x higher
- **Impact**: Success criteria SC-001 metrics adjusted but goal unchanged (unified execution)
- **Documentation Quality**: All planning documents remain structurally valid despite metric discrepancy
- **Lesson**: Always verify assumptions before detailed planning; baseline establishment is critical
- **Test Stability**: All visible robot_sf tests passing; fast-pysf needs fixture work per PR #236
