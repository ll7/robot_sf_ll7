# Research: fast-pysf Integration Quality Improvements

**Feature**: 148-improve-fast-pysf  
**Date**: 2025-10-29  
**Phase**: 0 (Research & Decision Documentation)

## Overview

This document consolidates research findings and decisions for integrating fast-pysf tests into the main pytest suite, resolving PR #236 review comments, and extending quality tooling to the fast-pysf subtree.

## Research Areas

### 1. Pytest Configuration for Subtree Integration

**Decision**: Add `fast-pysf/tests/` to pytest test paths in `pyproject.toml`

**Rationale**:
- Pytest supports multiple test discovery paths via `testpaths` configuration
- Current configuration: `testpaths = ["tests"]` (only robot_sf tests)
- Target configuration: `testpaths = ["tests", "fast-pysf/tests"]`
- This approach maintains separation while enabling unified `uv run pytest` execution
- No restructuring needed; preserves git subtree boundary

**Alternatives Considered**:
1. **Symlink approach**: Create symlink from `tests/fast_pysf/` → `fast-pysf/tests/`
   - Rejected: Breaks on Windows, complicates git subtree updates
2. **Conftest.py hook**: Use pytest collection hooks to discover fast-pysf tests
   - Rejected: Over-engineered for simple path addition
3. **Separate test command**: Keep `pytest fast-pysf/tests/` as separate invocation
   - Rejected: Doesn't meet user requirement "Make sure they run properly" (unified execution)

**Best Practices**:
- pytest documentation recommends `testpaths` for multi-directory projects
- Headless mode: Already supported via environment variables (validated in testing)
- Coverage integration: `[tool.coverage.run] source = ["robot_sf"]` should be extended to include `fast-pysf/pysocialforce`

**Implementation References**:
- pytest docs: https://docs.pytest.org/en/stable/reference/reference.html#confval-testpaths
- Current config: `pyproject.toml` lines 97-99
- Coverage config: `pyproject.toml` lines 124-139

---

### 2. Ruff Configuration for fast-pysf

**Decision**: Extend ruff to check `fast-pysf/` with selective rule enforcement

**Rationale**:
- Ruff already configured for `robot_sf/` with comprehensive rule set (security, complexity, modernization)
- fast-pysf is external subtree but should meet minimum quality standards
- Apply ruff formatting + essential error detection, defer complex refactoring rules
- Maintains compatibility with potential upstream fast-pysf contributions

**Configuration Strategy**:
1. **Formatting**: Apply ruff format to fast-pysf/ (non-breaking, auto-fixable)
2. **Error Detection**: Enable bug-risk categories (F, E, W, B, S, UP)
3. **Complexity Rules**: Initially exclude/relax (C901, PLR0912, etc.) to avoid blocking PR merge
4. **Gradual Adoption**: Address complexity warnings in follow-up iterations

**Alternatives Considered**:
1. **Separate ruff configuration**: Create `fast-pysf/pyproject.toml` with independent rules
   - Rejected: Complicates tooling, diverges from unified quality standards
2. **No ruff for fast-pysf**: Keep existing quality level
   - Rejected: Doesn't meet user requirement "Activate ruff for fast-pysf"
3. **Full robot_sf rule set**: Apply all rules immediately
   - Rejected: Too disruptive, blocks PR #236 resolution focus

**Best Practices**:
- Ruff supports per-directory rule overrides via `[tool.ruff.lint.per-file-ignores]`
- Start with format + critical errors, expand incrementally
- Document rule exceptions with rationale in `pyproject.toml` comments

**Implementation References**:
- Ruff configuration docs: https://docs.astral.sh/ruff/configuration/
- Current config: `pyproject.toml` lines 13-95
- Example per-file ignore: `"fast-pysf/**/*.py" = ["C901", "PLR0912", "PLR0915"]`

---

### 3. Type Checking (ty) for fast-pysf

**Decision**: Enable ty type checking for fast-pysf with numba compatibility considerations

**Rationale**:
- Type checking valuable for API clarity and refactoring safety
- fast-pysf uses numba `@njit` decorators requiring special handling
- mypy/pyright already handle numba through type stubs (numba-stubs package)
- ty (built on pyright) can check fast-pysf if numba types respected

**Numba Type Checking Strategy**:
1. **Decorated functions**: Accept that `@njit` functions may have limited type inference
2. **Public APIs**: Prioritize type annotations for non-JIT public interfaces
3. **Type stubs**: Rely on numba's built-in type hints where available
4. **Selective checking**: Use `# type: ignore[numba]` sparingly for unavoidable JIT conflicts

**Alternatives Considered**:
1. **Skip type checking**: Exclude fast-pysf from ty
   - Rejected: Misses opportunity for quality improvement
2. **Full strict mode**: Require 100% type coverage immediately
   - Rejected: Unrealistic given numba constraints and 4-6 hour budget
3. **Custom type checker**: Use numba-specific linter
   - Rejected: Over-engineered, ty already handles numba adequately

**Best Practices**:
- Focus on reducing `Any` types in public APIs (target: 25% reduction per spec)
- Document numba-related type limitations in docstrings
- Use `typing.Protocol` for duck-typed interfaces when appropriate
- Prioritize annotations for: forces.py public functions, simulator.py API, map loaders

**Implementation References**:
- ty (pyright) documentation: https://microsoft.github.io/pyright/
- numba typing: https://numba.readthedocs.io/en/stable/reference/types.html
- Current command: `uvx ty check . --exit-zero` (already configured)

---

### 4. Coverage Reporting for fast-pysf

**Decision**: Extend coverage source to include `fast-pysf/pysocialforce` with ≥70% target

**Rationale**:
- Coverage already enabled for robot_sf (91.73% per baseline)
- fast-pysf tests exist (12 tests) but coverage not tracked
- Target: SC-013 requires ≥70% coverage for fast-pysf code
- Provides visibility into test quality and gaps

**Configuration Changes**:
```toml
[tool.coverage.run]
source = [
    "robot_sf",
    "fast-pysf/pysocialforce"  # ADD THIS
]
omit = [
    "*/tests/*",
    "*/test_*",
    "fast-pysf/examples/*",  # ADD THIS
    # ... existing omits
]
```

**Alternatives Considered**:
1. **Separate coverage run**: Generate independent report for fast-pysf
   - Rejected: Fragments reporting, doesn't integrate with CI
2. **No coverage tracking**: Keep fast-pysf uncovered
   - Rejected: Doesn't meet SC-013 success criterion
3. **100% coverage target**: Match robot_sf's 91%+
   - Rejected: Unrealistic given test infrastructure state (2 failing tests currently)

**Best Practices**:
- Coverage HTML report separates by package (auto-grouped by source directory)
- Set coverage.json for programmatic analysis
- Document coverage baseline in research.md for tracking improvement
- Use `--cov-report term-missing` to identify uncovered lines during development

**Implementation References**:
- pytest-cov docs: https://pytest-cov.readthedocs.io/
- Current config: `pyproject.toml` lines 124-139
- Coverage guide: `docs/coverage_guide.md`

---

### 5. Test Fixture Resolution (Missing Map Files)

**Decision**: Create minimal test map fixtures in `fast-pysf/tests/test_maps/`

**Rationale**:
- 2/12 fast-pysf tests fail: `test_load_map` and `test_load_map_with_invalid_file`
- Error: `FileNotFoundError: tests/test_maps/map_regular.json`
- Tests expect fixtures in `tests/test_maps/` (relative path from fast-pysf root)
- Correct path: `fast-pysf/tests/test_maps/` (within subtree)

**Required Fixtures**:
1. `fast-pysf/tests/test_maps/map_regular.json`: Valid map with obstacles/routes
2. `fast-pysf/tests/test_maps/invalid_json_file.json`: Malformed JSON for error testing

**Fixture Schema** (based on map_loader.py inspection):
```json
{
  "obstacles": [[x1, y1], [x2, y2], ...],
  "routes": [
    {"id": "route1", "waypoints": [[x, y], [x, y]]}
  ],
  "crowded_zones": [
    {"center": [x, y], "radius": r}
  ]
}
```

**Alternatives Considered**:
1. **Mock fixtures**: Use unittest.mock to bypass file loading
   - Rejected: Doesn't test actual file parsing logic
2. **Generate at runtime**: Create fixtures programmatically in test setup
   - Rejected: Test isolation principle violated; harder to debug failures
3. **Copy from examples**: Use existing map files from fast-pysf/examples/
   - Partially accepted: Inspect examples for schema reference, create minimal fixtures

**Best Practices**:
- Keep fixtures minimal (single obstacle, single route sufficient for validation)
- Document fixture format in test_maps/README.md
- Invalid fixture should be genuinely malformed (not just missing keys)

**Implementation References**:
- Test file: `fast-pysf/tests/test_map_loader.py` lines 14, 43
- Map loader: `fast-pysf/pysocialforce/map_loader.py` lines 18-40
- Example maps: Check `fast-pysf/examples/` for schema examples

---

### 6. PR #236 Review Comments - Priority Issues

**Decision**: Resolve high-priority issues first, document medium/low in tracking doc

**High Priority Issues** (7 total - see pr236_review_comments.md):

1. **Unreachable print** (`map_config.py:81`):
   - **Fix**: Remove unreachable code or fix conditional logic
   - **Effort**: 15 minutes
   - **Risk**: Low (dead code removal)

2. **Unverified test result** (`TestObstacleForce.py:44`):
   - **Fix**: Add assertion or explicit verification comment
   - **Effort**: 30 minutes (requires understanding test intent)
   - **Risk**: Medium (may reveal actual test bug)

3. **Empty reset method** (`ped_behavior.py:114`):
   - **Fix**: Implement reset logic or document why empty
   - **Effort**: 1 hour (requires understanding state management)
   - **Risk**: Medium (may need upstream coordination)

4. **Magic numbers** (`map_osm_converter.py:36`):
   - **Fix**: Extract to named constants
   - **Effort**: 30 minutes
   - **Risk**: Low (refactoring only)

5. **Hardcoded scale factor TODO** (`map_osm_converter.py:35`):
   - **Fix**: Resolve TODO (make configurable or document decision)
   - **Effort**: 1-2 hours (may require API change)
   - **Risk**: Medium (affects public API if made configurable)

6. **ortho_vec documentation** (`forces.py:397`):
   - **Fix**: Clarify docstring with direction explanation
   - **Effort**: 30 minutes
   - **Risk**: Low (documentation only)

7. **Polygon closing question** (`map_loader_svg.py:96`):
   - **Fix**: Verify logic and add comment explaining behavior
   - **Effort**: 45 minutes
   - **Risk**: Low (verification + documentation)

**Medium/Low Priority** (17 total): Address after Phase 1 complete, tracked in tasks.md

**Alternatives Considered**:
1. **All issues at once**: Fix all 24 comments before test integration
   - Rejected: Blocks Phase 1 progress, 7-11 hour estimate
2. **Skip PR review**: Focus only on test/quality tooling
   - Rejected: User explicitly requested "Create a list of all review comments... and finally resolve the plan"
3. **Upstream only**: Report all issues to fast-pysf upstream without local fixes
   - Rejected: Some issues are local (CI badge, GitHub Actions versions) or need immediate resolution

**Best Practices**:
- Document resolution status in pr236_review_comments.md
- Create tracking issue for upstream coordination if needed
- Add CHANGELOG.md entry for user-visible fixes
- Update subtree merge documentation if coordination pattern established

---

## Phase 0 Summary

All technical unknowns from plan.md Technical Context resolved:

✅ **Pytest integration**: `testpaths` addition  
✅ **Ruff activation**: Per-directory rules with gradual adoption  
✅ **Type checking**: ty with numba compatibility strategy  
✅ **Coverage tracking**: Source extension with 70% target  
✅ **Test fixtures**: Minimal map JSON files to be created  
✅ **PR review strategy**: High-priority first, tracked resolution  

**No NEEDS CLARIFICATION items remaining**. Ready for Phase 1 (Design & Contracts).

---

## Next Steps

**Phase 1 Tasks** (from plan workflow):
1. Generate `data-model.md` (test infrastructure entities)
2. Generate API contracts in `contracts/` (pytest plugin interface if needed)
3. Create `quickstart.md` (developer onboarding for unified test workflow)
4. Update agent context via `.specify/scripts/bash/update-agent-context.sh copilot`
5. Re-evaluate Constitution Check (expected: still all pass)

**Estimated Phase 1 Duration**: 2-3 hours
